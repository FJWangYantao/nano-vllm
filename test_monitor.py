import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

from nanovllm.monitor.collector import MetricsCollector
from nanovllm.monitor.exporter import ConsoleExporter, JsonExporter
from nanovllm.monitor.metrics import EngineStatsSummary, RequestMetrics, StepMetrics
from nanovllm.monitor.profiler import Profiler


def test_monitoring():
    collector = MetricsCollector()
    console_exporter = ConsoleExporter(verbose=True)
    json_path = "metrics_test.json"
    json_exporter = JsonExporter(json_path)

    # 1. Simulate requests arriving
    collector.on_request_arrival(seq_id=1, num_prompt_tokens=128)
    collector.on_request_arrival(seq_id=2, num_prompt_tokens=256)

    # 2. Simulate scheduler & prefill step
    collector.on_request_start(seq_id=1)
    collector.on_request_start(seq_id=2)
    collector.on_prefix_cache_lookup(hit_blocks=2, total_blocks=4)

    s1 = collector.record_step(
        is_prefill=True,
        step_latency=0.01,
        num_seqs=2,
        num_tokens=384,
        waiting_seqs=0,
        running_seqs=2,
        free_blocks=100,
        used_blocks=10,
        num_preemptions_in_step=0,
    )
    console_exporter.export_step(s1)

    # 3. Simulate decode steps & TTFT
    collector.on_first_token(seq_id=1)
    collector.on_first_token(seq_id=2)

    for _ in range(4):
        s = collector.record_step(
            is_prefill=False,
            step_latency=0.005,
            num_seqs=2,
            num_tokens=2,
            waiting_seqs=0,
            running_seqs=2,
            free_blocks=98,
            used_blocks=12,
            num_preemptions_in_step=0,
        )
        console_exporter.export_step(s)

    # 4. Finish requests
    collector.on_request_finish(seq_id=1, num_completion_tokens=4, num_cached_tokens=32)
    collector.on_request_finish(seq_id=2, num_completion_tokens=4, num_cached_tokens=32)

    # 5. Summary & Export
    summary = collector.get_summary()
    console_exporter.export_summary(summary)
    json_exporter.export_summary(summary)

    # 6. Profiler context manager check
    profiler = Profiler(name="test_runner")
    with profiler.profile("prefill"):
        time.sleep(0.002)
    with profiler.profile("decode"):
        time.sleep(0.001)
    print("Profiler summary:", profiler.summary())

    assert summary.total_requests == 2
    assert summary.finished_requests == 2
    assert summary.total_prefill_steps == 1
    assert summary.total_decode_steps == 4
    assert summary.prefix_cache_hit_rate == 0.5

    if os.path.exists(json_path):
        os.remove(json_path)

    print("\nALL MONITORING UNIT TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_monitoring()
