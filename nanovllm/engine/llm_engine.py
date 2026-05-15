import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    """
    初始配置，读取配置信息
    主动过滤无关参数
    """
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        # 用于张量并行TP
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        # 从1开始遍历（0是主线程），给每个 GPU 分发子进程
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # 进程和事件的引用保存
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    # 定义退出函数，清理引擎资源
    # 遍历子进程列表，确保每一个子进程退出
    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()


    # 添加新的生成请求
    # 对字符串做分词，转换为 token_ids
    # 创建一个 Sequence 对象，并添加到调度器中
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    # 让调度器决定这一轮处理哪些序列
    # 计算本轮需要处理的 token 数
    # 如果是整数，则表示 prefill 需要处理的 token 数
    # 如果是负数，则表示 decode 需要处理的 token 数
    # 调用推理模型，返回 token_ids
    # 结果交由调度器做后处理
    # 收集本轮完成的序列的 ID 和 token_ids
    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    # 判定所有请求是否都完成
    def is_finished(self):
        return self.scheduler.is_finished()


    # 输入多个 prompt，返回结果生成列表
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        # 使用 tqdm 显示进度条
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        # 统一 sampling_params 列表长度
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # 添加请求
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        # 输出结果
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        # 记录 prefill 速度和 decode 速度
        while not self.is_finished():
            # 单 step 计算
            t = perf_counter()
            output, num_tokens = self.step()
            # 通过正负更新 prefill 速度和 decode 速度
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            # 更新进度条
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        # 排序 seq 结果
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # token_id 转文本
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
