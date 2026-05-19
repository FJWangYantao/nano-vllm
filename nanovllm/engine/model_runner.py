import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    '''
    传入参数：
    config: 整体配置对象
    hf_config: 模型配置对象
    block_size: kv cache 的 block 大小
    enforce_eager: 是否强制 eager 模式，如果允许，则可以捕获 CUDA Graph 提速
    world_size: 进程数量
    rank: 当前进程 ID
    event: 多进程间同步通信事件对象
    '''
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # 建立 PyTorch 分布式通信组
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        # 设置当前 GPU
        torch.cuda.set_device(rank)
        # 设置默认数据类型
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        # 创建模型，加载权重
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        # 创建采样器
        self.sampler = Sampler()
        # 预热模型，让 CUDA kernel 先跑一遍
        # 提前分配一些临时显存
        # 记录 peak memory
        self.warmup_model()
        # 分配 KV Cache
        self.allocate_kv_cache()
        # 如果不强制 eager 模式，则捕获 CUDA Graph
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 如果 world_size > 1，则需要共享内存通信
        if self.world_size > 1:
            # rank0 创建共享内存
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            # 其他进程等待 rank0 创建共享内存
            # 连接主进程创建的共享内存
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                # 循环等待主进程通过共享内存发来的命令
                self.loop()

    # 负责释放 ModelRunner 初始化时创建的分布式资源、共享内存、CUDA Graph，以及销毁进程组。
    def exit(self):
        # 如果 world_size > 1，则需要共享内存通信
        # 退出时则释放此共享内存
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            # 所有进程都 close 只有共享内存 unlink 才能真正删除
            if self.rank == 0:
                self.shm.unlink()
        # 没启用 enforce_eager 则删除 CUDA Graph
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        # 等待 CUDA 操作完成再销毁
        torch.cuda.synchronize()
        # 销毁进程组
        dist.destroy_process_group()

    # 循环等待主进程通过共享内存发来的命令
    # 当遇到 exit 时退出循环，否则读取指令，并按照对应方法名操作
    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    # 断言多进程模式 且当前进程不是 rank0
    # 等待事件被主进程置位
    # 小端读取共享内存的前4个字节
    # 根据
    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

     # 前4个字节是长度，4：n+4是数据
    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        # 序列化方法名与参数，变为字节流
        data = pickle.dumps([method_name, *args])
        n = len(data)
        # 写入长度 后面跟着数据
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        # 通知所有进程数据已就绪
        for event in self.event:
            event.set()

    # 同步点，rank0 写指令，从进程读取这些指令
    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        # 所有进程都执行这个 method
        method = getattr(self, method_name, None)
        return method(*args)

    # 预热模型
    # 首先清空显存缓存和内存统计
    # 计算预热所用序列长度和数量
    # 模拟序列数据
    # 运行一遍前向传播
    # 再次清空显存
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    # 首先获取显存信息 空闲 峰值 当前分配
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # KV 组数，张量并行时平均分到每个 GPU
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        # 计算每个头维度
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 计算每个 block 的容量
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        # 计算可分配 block 总数
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        # 随机初始化 kvcache
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        # 遍历所有 Attention层， 存入 kvcache
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
    # 输入一组序列，构造统一的 block table 张量
    # 左对齐填充：末尾补 -1 使其与最长序列等长
    # 转为 int32 GPU 张量
    # pin_memory=True 锁页内存 + non_blocking=True 异步传输，加速 CPU→GPU 拷贝
    # 存储 kvcache 对应索引的数据结构
    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        # 初始化容器
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            # 确定当前序列处理范围
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            # 构建输入
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            # 构建 Flash Attention 长度信息
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            # 构建 slot_mapping
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        # prefix cache 判断
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        # 转为 GPU 张量，并设置上下文
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    # 准备 decode 的输入
    # 获取每个序列的最后一个 token 和位置
    # 构建 input_ids 和 positions 张量
    # 构建 context_lens，该序列当前长度
    # 计算本次新生成的 KV 应该写到 paged cache 的哪个位置
    # 转为 GPU 张量，并设置上下文，供 flash attention 前向使用
    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    # 准备参数（只有温度）
    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    # 前向传播
    # 三个条件：在 prefill 阶段，强制 eager 模式，input_ids 长度大于 512
    # 在这三个条件下，只要满足任何一个，都直接跑模型，不适用 CUDA Graph
    # 否则应用 CUDA Graph 加速
    @torch.inference_mode() # 推理模式
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # 获取当前 batch size
            bs = input_ids.size(0)
            context = get_context()
            # 获取一个 graph size，能够装下当前 batch size,但是又不超过太多，应该是一个增序
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            # 把上下文放到 graph 里面，让 graph replay,可以快速计算
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    '''
    传入参数：
    seqs: 序列列表
    is_prefill: 是否是 prefill
    返回值：
    token_ids: 生成的 token_ids 列表
    '''
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # 准备 prefill 或 decode 的输入
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # 准备采样参数，只有 rank0 进行采样，其他 GPU 不用
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        # 调用模型推理
        logits = self.run_model(input_ids, positions, is_prefill)
        # 采样
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        # 重置上下文
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        # 配置
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # [1, 2, 4, 8, 16, 32, 48, ...]
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 先录大的 graph，可以发现显存不足问题
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
           # 在 graph 上下文中执行一遍模型前向，捕获所有 GPU 操作
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 存储 graph 变量
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
