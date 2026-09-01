import contextlib
import time
from typing import Iterator


class Profiler:
    """
    Lightweight timing profiler for instrumentation without heavy overhead.
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.timings: dict[str, list[float]] = {}
        self.counts: dict[str, int] = {}

    @contextlib.contextmanager
    def profile(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if name not in self.timings:
                self.timings[name] = []
                self.counts[name] = 0
            self.timings[name].append(elapsed)
            self.counts[name] += 1

    record_scope = profile

    def get_stats(self) -> dict[str, dict[str, float]]:
        result = {}
        for name, vals in self.timings.items():
            if not vals:
                continue
            result[name] = {
                "count": float(self.counts[name]),
                "total_sec": sum(vals),
                "avg_ms": (sum(vals) / len(vals)) * 1000.0,
                "min_ms": min(vals) * 1000.0,
                "max_ms": max(vals) * 1000.0,
            }
        return result

    summary = get_stats

    def reset(self):
        self.timings.clear()
        self.counts.clear()
