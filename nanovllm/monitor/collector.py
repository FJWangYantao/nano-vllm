import time
from collections import deque
from nanovllm.monitor.metrics import RequestMetrics, StepMetrics, EngineStatsSummary


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c < len(sorted_vals):
        return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])
    return sorted_vals[f]


class MetricsCollector:
    """
    Central collector for monitoring Nano-vLLM metrics:
    - Request-level lifecycles (arrival, start, TTFT, finish, token counts)
    - Step-level metrics (step time, batch size, throughput, queues, KV blocks)
    - Prefix caching & preemption events
    """

    def __init__(self, max_history_steps: int = 10000):
        self.max_history_steps = max_history_steps
        self.step_counter = 0
        self.start_engine_time = time.perf_counter()

        # Request metrics mapping: seq_id -> RequestMetrics
        self.active_requests: dict[int, RequestMetrics] = {}
        self.finished_requests: list[RequestMetrics] = []

        # Step history
        self.step_history: deque[StepMetrics] = deque(maxlen=max_history_steps)

        # Cumulative counters
        self.total_preemptions = 0
        self.total_prefix_cache_hit_blocks = 0
        self.total_prefix_cache_lookup_blocks = 0

    def on_request_arrival(self, seq_id: int, num_prompt_tokens: int, arrival_time: float | None = None) -> RequestMetrics:
        t = arrival_time if arrival_time is not None else time.perf_counter()
        req = RequestMetrics(
            seq_id=seq_id,
            arrival_time=t,
            num_prompt_tokens=num_prompt_tokens,
        )
        self.active_requests[seq_id] = req
        return req

    def on_request_start(self, seq_id: int, start_time: float | None = None):
        if seq_id in self.active_requests:
            req = self.active_requests[seq_id]
            if req.start_time == 0.0:
                req.start_time = start_time if start_time is not None else time.perf_counter()

    def on_first_token(self, seq_id: int, first_token_time: float | None = None):
        if seq_id in self.active_requests:
            req = self.active_requests[seq_id]
            if req.first_token_time == 0.0:
                req.first_token_time = first_token_time if first_token_time is not None else time.perf_counter()

    def on_request_finish(
        self,
        seq_id: int,
        num_completion_tokens: int,
        num_cached_tokens: int = 0,
        finish_time: float | None = None,
    ):
        req = self.active_requests.pop(seq_id, None)
        if req is not None:
            req.finish_time = finish_time if finish_time is not None else time.perf_counter()
            req.num_completion_tokens = num_completion_tokens
            req.num_cached_tokens = num_cached_tokens
            if req.first_token_time == 0.0:
                req.first_token_time = req.finish_time
            self.finished_requests.append(req)

    def on_preemption(self, seq_id: int):
        self.total_preemptions += 1

    def on_prefix_cache_lookup(self, hit_blocks: int, total_blocks: int):
        self.total_prefix_cache_hit_blocks += hit_blocks
        self.total_prefix_cache_lookup_blocks += total_blocks

    def record_step(
        self,
        is_prefill: bool,
        step_latency: float,
        num_seqs: int,
        num_tokens: int,
        waiting_seqs: int,
        running_seqs: int,
        free_blocks: int,
        used_blocks: int,
        num_preemptions_in_step: int = 0,
        hit_blocks: int = 0,
        lookup_blocks: int = 0,
    ) -> StepMetrics:
        self.step_counter += 1
        metric = StepMetrics(
            step_id=self.step_counter,
            timestamp=time.perf_counter(),
            is_prefill=is_prefill,
            step_latency=step_latency,
            num_seqs=num_seqs,
            num_tokens=abs(num_tokens),
            num_batched_tokens=abs(num_tokens),
            waiting_seqs=waiting_seqs,
            running_seqs=running_seqs,
            free_blocks=free_blocks,
            used_blocks=used_blocks,
            num_preemptions=num_preemptions_in_step,
            prefix_cache_hit_blocks=hit_blocks,
            prefix_cache_total_blocks=lookup_blocks,
        )
        self.step_history.append(metric)
        return metric

    def get_summary(self) -> EngineStatsSummary:
        now = time.perf_counter()
        total_time = max(1e-6, now - self.start_engine_time)

        finished = self.finished_requests
        num_finished = len(finished)

        ttfts = [r.ttft for r in finished if r.ttft > 0]
        e2es = [r.e2e_latency for r in finished if r.e2e_latency > 0]
        tpots = [r.tpot for r in finished if r.tpot > 0]
        queue_lats = [r.queue_latency for r in finished if r.queue_latency > 0]

        total_prompt_tokens = sum(r.num_prompt_tokens for r in finished)
        total_completion_tokens = sum(r.num_completion_tokens for r in finished)
        total_cached_prompt_tokens = sum(r.num_cached_tokens for r in finished)

        prefill_steps = sum(1 for s in self.step_history if s.is_prefill)
        decode_steps = sum(1 for s in self.step_history if not s.is_prefill)

        prefix_hit_rate = 0.0
        if self.total_prefix_cache_lookup_blocks > 0:
            prefix_hit_rate = self.total_prefix_cache_hit_blocks / self.total_prefix_cache_lookup_blocks

        kv_usage = 0.0
        if self.step_history:
            last_step = self.step_history[-1]
            total_blocks = last_step.free_blocks + last_step.used_blocks
            if total_blocks > 0:
                kv_usage = last_step.used_blocks / total_blocks

        return EngineStatsSummary(
            total_requests=num_finished + len(self.active_requests),
            finished_requests=num_finished,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_cached_prompt_tokens=total_cached_prompt_tokens,
            total_steps=self.step_counter,
            total_prefill_steps=prefill_steps,
            total_decode_steps=decode_steps,
            total_preemptions=self.total_preemptions,
            avg_ttft=sum(ttfts) / len(ttfts) if ttfts else 0.0,
            p50_ttft=_percentile(ttfts, 50),
            p95_ttft=_percentile(ttfts, 95),
            p99_ttft=_percentile(ttfts, 99),
            avg_e2e_latency=sum(e2es) / len(e2es) if e2es else 0.0,
            p50_e2e_latency=_percentile(e2es, 50),
            p95_e2e_latency=_percentile(e2es, 95),
            p99_e2e_latency=_percentile(e2es, 99),
            avg_tpot=sum(tpots) / len(tpots) if tpots else 0.0,
            avg_queue_latency=sum(queue_lats) / len(queue_lats) if queue_lats else 0.0,
            total_time=total_time,
            generation_throughput=total_completion_tokens / total_time,
            total_token_throughput=(total_prompt_tokens + total_completion_tokens) / total_time,
            request_throughput=num_finished / total_time,
            prefix_cache_hit_rate=prefix_hit_rate,
            kv_cache_usage_pct=kv_usage * 100.0,
        )

    def reset(self):
        self.step_counter = 0
        self.start_engine_time = time.perf_counter()
        self.active_requests.clear()
        self.finished_requests.clear()
        self.step_history.clear()
        self.total_preemptions = 0
        self.total_prefix_cache_hit_blocks = 0
        self.total_prefix_cache_lookup_blocks = 0
