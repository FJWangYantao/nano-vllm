import json
import os
import sys
from abc import ABC, abstractmethod
from typing import Any
from nanovllm.monitor.metrics import EngineStatsSummary, StepMetrics, RequestMetrics


class BaseExporter(ABC):
    @abstractmethod
    def export_summary(self, summary: EngineStatsSummary):
        pass

    @abstractmethod
    def export_step(self, step: StepMetrics):
        pass


class ConsoleExporter(BaseExporter):
    """Prints monitoring metrics to console in a clean format."""

    def __init__(self, stream=sys.stdout, verbose: bool = False):
        self.stream = stream
        self.verbose = verbose

    def export_summary(self, summary: EngineStatsSummary):
        lines = [
            "=" * 60,
            " Nano-vLLM Runtime Statistics Summary",
            "=" * 60,
            f" Total Requests          : {summary.finished_requests}/{summary.total_requests} finished",
            f" Total Steps             : {summary.total_steps} (Prefill: {summary.total_prefill_steps}, Decode: {summary.total_decode_steps})",
            f" Total Prompt Tokens     : {summary.total_prompt_tokens} (Cached: {summary.total_cached_prompt_tokens})",
            f" Total Generation Tokens : {summary.total_completion_tokens}",
            f" Total Elapsed Time      : {summary.total_time:.3f} s",
            "-" * 60,
            " Throughput:",
            f"   Generation Throughput : {summary.generation_throughput:.2f} tok/s",
            f"   Total Token Throughput: {summary.total_token_throughput:.2f} tok/s",
            f"   Request Throughput    : {summary.request_throughput:.2f} req/s",
            "-" * 60,
            " Latency (s):",
            f"   TTFT (Avg / P50 / P95 / P99)  : {summary.avg_ttft:.4f}s / {summary.p50_ttft:.4f}s / {summary.p95_ttft:.4f}s / {summary.p99_ttft:.4f}s",
            f"   E2E  (Avg / P50 / P95 / P99)  : {summary.avg_e2e_latency:.4f}s / {summary.p50_e2e_latency:.4f}s / {summary.p95_e2e_latency:.4f}s / {summary.p99_e2e_latency:.4f}s",
            f"   TPOT (Decode Token Latency)   : {summary.avg_tpot * 1000:.2f} ms/tok",
            f"   Queue Latency                 : {summary.avg_queue_latency * 1000:.2f} ms",
            "-" * 60,
            " Resource & Cache:",
            f"   Prefix Cache Hit Rate : {summary.prefix_cache_hit_rate * 100:.2f}%",
            f"   KV Cache Utilization  : {summary.kv_cache_usage_pct:.2f}%",
            f"   Total Preemptions     : {summary.total_preemptions}",
            "=" * 60,
        ]
        self.stream.write("\n".join(lines) + "\n")
        self.stream.flush()

    def export_step(self, step: StepMetrics):
        phase = "PREFILL" if step.is_prefill else "DECODE"
        line = (
            f"[Step {step.step_id:04d} | {phase:<7}] "
            f"lat={step.step_latency*1000:.1f}ms, "
            f"seqs={step.num_seqs}, tokens={step.num_tokens}, "
            f"waiting={step.waiting_seqs}, running={step.running_seqs}, "
            f"kv_used={step.used_blocks}/({step.free_blocks + step.used_blocks})"
        )
        self.stream.write(line + "\n")
        self.stream.flush()


class JsonExporter(BaseExporter):
    """Saves metrics summary or detailed traces to JSON files."""

    def __init__(self, output_path: str = "monitoring_report.json", dump_details: bool = False):
        self.output_path = output_path
        self.dump_details = dump_details

    def export_summary(self, summary: EngineStatsSummary):
        data = {
            "summary": summary.__dict__,
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def export_step(self, step: StepMetrics):
        pass
