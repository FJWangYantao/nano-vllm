from dataclasses import dataclass, field
import time


@dataclass
class RequestMetrics:
    seq_id: int
    arrival_time: float
    start_time: float = 0.0
    first_token_time: float = 0.0
    finish_time: float = 0.0
    num_prompt_tokens: int = 0
    num_completion_tokens: int = 0
    num_cached_tokens: int = 0

    @property
    def queue_latency(self) -> float:
        if self.start_time > 0 and self.arrival_time > 0:
            return max(0.0, self.start_time - self.arrival_time)
        return 0.0

    @property
    def ttft(self) -> float:
        """Time To First Token (from arrival to first token output)"""
        if self.first_token_time > 0 and self.arrival_time > 0:
            return max(0.0, self.first_token_time - self.arrival_time)
        return 0.0

    @property
    def e2e_latency(self) -> float:
        """End-to-End Latency"""
        if self.finish_time > 0 and self.arrival_time > 0:
            return max(0.0, self.finish_time - self.arrival_time)
        return 0.0

    @property
    def tpot(self) -> float:
        """Time Per Output Token (decode phase latency per token)"""
        if self.finish_time > self.first_token_time and self.num_completion_tokens > 1:
            return (self.finish_time - self.first_token_time) / (self.num_completion_tokens - 1)
        return 0.0

    @property
    def inter_token_latency(self) -> float:
        """Average latency per generated completion token"""
        if self.finish_time > self.start_time and self.num_completion_tokens > 0:
            return (self.finish_time - self.start_time) / self.num_completion_tokens
        return 0.0


@dataclass
class StepMetrics:
    step_id: int
    timestamp: float
    is_prefill: bool
    step_latency: float
    num_seqs: int
    num_tokens: int
    num_batched_tokens: int
    waiting_seqs: int
    running_seqs: int
    free_blocks: int
    used_blocks: int
    num_preemptions: int = 0
    prefix_cache_hit_blocks: int = 0
    prefix_cache_total_blocks: int = 0


@dataclass
class EngineStatsSummary:
    total_requests: int = 0
    finished_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_prompt_tokens: int = 0
    total_steps: int = 0
    total_prefill_steps: int = 0
    total_decode_steps: int = 0
    total_preemptions: int = 0
    
    # Latencies
    avg_ttft: float = 0.0
    p50_ttft: float = 0.0
    p95_ttft: float = 0.0
    p99_ttft: float = 0.0

    avg_e2e_latency: float = 0.0
    p50_e2e_latency: float = 0.0
    p95_e2e_latency: float = 0.0
    p99_e2e_latency: float = 0.0

    avg_tpot: float = 0.0
    avg_queue_latency: float = 0.0

    # Throughput
    total_time: float = 0.0
    generation_throughput: float = 0.0  # completion tokens / s
    total_token_throughput: float = 0.0  # (prompt + completion tokens) / s
    request_throughput: float = 0.0      # requests / s

    # Cache Stats
    prefix_cache_hit_rate: float = 0.0
    kv_cache_usage_pct: float = 0.0
