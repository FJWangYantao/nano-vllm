from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager

# prefill 完成进入 running
# 未完成则待在 waiting
# prefil 每次调度处理 token 数：取决于预算
# decode 处理数：一次调度处理一个 token
class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    # 调度完成判定：waiting队列和running队列为空
    def is_finished(self):
        return not self.waiting and not self.running

    # waiting队列append一个seq
    def add(self, seq: Sequence):
        self.waiting.append(seq)

    # 优先从waiting取seq做prefill
    # 若waiting空了且或者资源不够的时候，则从running取seq做decode
    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        # 调度的条件：waiting队列有在等待的seq，且已经调度的队列小于最大限制
        # 一次性分配block
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            # 取出队首seq，并计算剩余token预算，如果没预算，停止
            # 如果该序列还没分配 KV Cache block
            # 则查找前缀缓存物理块并为该序列分配，同时计算剩余空闲块是否够分配
            # 若不够分配，can_allocate会返回-1，此时则直接退出
            # 分配完命中缓存的物理块，则计算剩余未被分配到的token数
            # 对比剩余token数和预算，如果预算不够处理剩余token
            # 则判断是否是第一个被调度的序列，是则允许做chunked prefill
            # 如果不是，则说明这一轮已经有被做过chunked prefill了
            # 若prompt全部处理完，则送入running
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        # 执行 prefill
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        # running 队列还有 block ，本轮已调度序列未达到上限
        # 尝试分配空闲块，如果没有,则抢占进度最少的（最晚加入的）
        # 如果有则正常分配物理块
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                # 若序列里还有其他 seq 则驱逐
                # 若没其他 seq 可以驱逐，强行暂停自己
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        # 必须有已调度的序列，否则出错
        assert scheduled_seqs
        # 左端插入必须倒一次，否则插入的seq会倒序
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    # 抢占某个块的资源（释放）
    # 放到 waiting 队列
    # 释放物理块
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    # 模型推理后处理
    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            # 计算本轮推理完的块的 hash 用于前缀缓存查找
            # 更新缓存计数，清空调度计数
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # prefill 如果还未完成，则不追加 token 和生成完成 token
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            # 追加 token
            # 检测是否完成条件：
            # 1. 生成 EOS token
            # 2. token 数达到上限
            # 状态改为 finished，释放物理块，移出 running 队列
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
