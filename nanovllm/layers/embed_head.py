import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    # 分割加载权重
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        # 计算加载范围
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        # 复制到 GPU 本地
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        # 多进程
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        # 获取 embedding
        y = F.embedding(x, self.weight)
        # 同步结果
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    # 线性头并行
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        # 处理 prefill 阶段
        context = get_context()
        if context.is_prefill:
            # 获取最后一个 token 的索引
            last_indices = context.cu_seqlens_q[1:] - 1
            # 获取最后一个 token 的 embedding
            x = x[last_indices].contiguous()
        # 线性变换
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            # 只在 rank0 上收集所有 logits
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            # 从所有 GPU 收集 logits
            dist.gather(logits, all_logits, 0)
            # 在 rank0 上拼接所有 logits
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
