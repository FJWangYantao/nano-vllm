# Nano-vLLM 推理引擎运行时监控系统计划

## 1. 项目目标

为 Nano-vLLM 增加一套轻量、低侵入、可扩展的运行时监控系统，用于观察推理引擎在 Prefill、Decode、Scheduler、KV Cache、Prefix Cache、Tensor Parallel 等关键路径上的实时行为。

该系统首先服务于源码学习和性能分析，不以替代 Prometheus/Grafana 等完整可观测性平台为目标。第一阶段优先保证：

- 指标定义清晰；
- 埋点位置准确；
- 对推理性能影响尽量小；
- 能输出结构化数据；
- 后续可扩展到 Web Dashboard、Prometheus Exporter 或实验报告。

---

## 2. 第一阶段核心指标

### 2.1 请求级指标

| 指标 | 含义 | 推荐埋点位置 |
| --- | --- | --- |
| `request_count` | 累计请求数 | `LLMEngine.add_request()` |
| `finished_request_count` | 已完成请求数 | `Scheduler.postprocess()` / `LLMEngine.step()` |
| `ttft_ms` | Time To First Token，请求进入系统到首个输出 token 的时间 | `Sequence` + `Scheduler.postprocess()` |
| `e2e_latency_ms` | 请求端到端生成耗时 | `Sequence` 生命周期 |
| `prompt_tokens` | Prompt token 数 | `LLMEngine.add_request()` |
| `completion_tokens` | 输出 token 数 | `Sequence` |
| `total_tokens` | 总处理 token 数 | 聚合统计 |

### 2.2 推理性能指标

| 指标 | 含义 | 推荐埋点位置 |
| --- | --- | --- |
| `prefill_tokens_per_sec` | Prefill 吞吐 | `LLMEngine.generate()` / `step()` |
| `decode_tokens_per_sec` | Decode 吞吐 | `LLMEngine.generate()` / `step()` |
| `step_latency_ms` | 单次调度-推理循环耗时 | `LLMEngine.step()` |
| `prefill_step_latency_ms` | Prefill step 耗时 | `LLMEngine.step()` |
| `decode_step_latency_ms` | Decode step 耗时 | `LLMEngine.step()` |
| `batch_size` | 当前 step 调度的序列数 | `Scheduler.schedule()` |
| `scheduled_tokens` | 当前 step 实际调度 token 数 | `Scheduler.schedule()` |

### 2.3 Scheduler 指标

| 指标 | 含义 | 推荐埋点位置 |
| --- | --- | --- |
| `waiting_sequences` | waiting 队列长度 | `Scheduler.schedule()` |
| `running_sequences` | running 队列长度 | `Scheduler.schedule()` |
| `prefill_sequences` | 当前 Prefill 序列数 | `Scheduler.schedule()` |
| `decode_sequences` | 当前 Decode 序列数 | `Scheduler.schedule()` |
| `preemption_count` | KV Cache 不足导致的累计抢占次数 | `Scheduler.preempt()` |
| `chunked_prefill_count` | 发生 Chunked Prefill 的累计次数 | `Scheduler.schedule()` |
| `scheduler_wait_ms` | 请求在 waiting 队列中的等待时间 | `Sequence` + Scheduler |

### 2.4 KV Cache 指标

| 指标 | 含义 | 推荐埋点位置 |
| --- | --- | --- |
| `kv_total_blocks` | KV Cache 总 block 数 | `BlockManager.__init__()` |
| `kv_used_blocks` | 当前已使用 block 数 | `BlockManager` |
| `kv_free_blocks` | 当前空闲 block 数 | `BlockManager` |
| `kv_utilization` | KV Cache 使用率 | `used / total` |
| `block_alloc_count` | 累计物理块分配次数 | `_allocate_block()` |
| `block_free_count` | 累计物理块释放次数 | `_deallocate_block()` |

### 2.5 Prefix Cache 指标

| 指标 | 含义 | 推荐埋点位置 |
| --- | --- | --- |
| `prefix_cache_lookup_blocks` | 尝试查询的前缀 block 总数 | `can_allocate()` |
| `prefix_cache_hit_blocks` | 命中的前缀 block 总数 | `can_allocate()` |
| `prefix_cache_miss_blocks` | 未命中的 block 总数 | `can_allocate()` |
| `prefix_cache_hit_rate` | 前缀缓存命中率 | 聚合计算 |
| `prefix_cache_saved_tokens` | 因命中缓存而跳过 Prefill 的 token 数 | `allocate()` / Scheduler |

### 2.6 GPU / Tensor Parallel 指标（二阶段）

- GPU allocated memory；
- GPU reserved memory；
- GPU peak memory；
- 每个 rank 的显存占用；
- Tensor Parallel world size；
- NCCL 同步/通信耗时（后续实验性加入）；
- CUDA Graph replay 次数与 eager 执行次数。

---

## 3. 监控架构设计

推荐采用“核心引擎只负责采集，展示层独立”的方式。

```text
LLMEngine / Scheduler / BlockManager / ModelRunner
                    |
                    v
             RuntimeMetrics
                    |
        +-----------+-----------+
        |                       |
        v                       v
 ConsoleReporter          JsonlReporter
        |                       |
        +-----------+-----------+
                    |
                    v
         Future: Web / Prometheus
```

核心原则：

1. 推理核心代码不直接依赖 Web 框架；
2. 指标采集与指标展示解耦；
3. 监控模块故障不能影响正常推理；
4. 高频路径只做 O(1) 的计数、时间戳或简单算术；
5. 默认关闭高开销监控能力。

---

## 4. 建议代码结构

```text
nanovllm/
├── monitoring/
│   ├── __init__.py
│   ├── metrics.py
│   ├── collector.py
│   ├── reporter.py
│   └── snapshot.py
│
├── engine/
│   ├── llm_engine.py
│   ├── scheduler.py
│   ├── block_manager.py
│   └── model_runner.py
```

### `metrics.py`
定义指标名称、计数器和直方统计对象。

### `collector.py`
统一维护 RuntimeMetrics，提供：

- `inc()`；
- `set()`；
- `observe()`；
- `snapshot()`；
- `reset()`。

### `snapshot.py`
使用 `dataclass` 表达某一时刻的系统状态，避免展示层直接访问引擎内部对象。

### `reporter.py`
第一版提供：

- ConsoleReporter；
- JsonlReporter。

后续可增加：

- PrometheusReporter；
- WebSocketReporter；
- DashboardReporter。

---

## 5. Sequence 生命周期扩展

建议在 `Sequence` 中加入监控时间戳：

```python
created_at
first_scheduled_at
first_token_at
finished_at
```

由此可以计算：

```text
Queue Time = first_scheduled_at - created_at
TTFT       = first_token_at - created_at
E2E        = finished_at - created_at
```

同时记录：

```text
prompt_token_count
completion_token_count
preempt_count
cached_token_count
```

注意：Sequence 只保存原始运行状态，统计逻辑尽量放在 monitoring 模块中。

---

## 6. 各模块埋点计划

### 6.1 `LLMEngine`

重点监控整个 step 的耗时及吞吐。

需要加入：

- 请求进入时间；
- step 开始/结束时间；
- Prefill / Decode step 类型；
- 当前 step token 数；
- 已完成请求数；
- 全局吞吐统计。

第一版应把当前 `generate()` 中只用于 tqdm 的 throughput 计算迁移为可持久保存的 RuntimeMetrics。

### 6.2 `Scheduler`

每次 `schedule()` 后记录：

```text
waiting_sequences
running_sequences
scheduled_sequences
scheduled_tokens
is_prefill
```

当发生：

```python
self.preempt(...)
```

增加：

```text
preemption_count += 1
```

当：

```text
remaining < num_tokens
```

且首个 Sequence 被切分时，记录一次 Chunked Prefill。

### 6.3 `BlockManager`

`BlockManager` 是 KV Cache 与 Prefix Cache 指标的核心数据源。

在 `_allocate_block()`：

```text
block_alloc_count += 1
```

在 `_deallocate_block()`：

```text
block_free_count += 1
```

实时状态直接由：

```python
len(self.used_block_ids)
len(self.free_block_ids)
```

计算。

在 `can_allocate()` 中统计：

```text
prefix_cache_lookup_blocks
prefix_cache_hit_blocks
prefix_cache_miss_blocks
```

### 6.4 `ModelRunner`

第一阶段不做过多高频埋点，只记录：

- CUDA Graph replay 次数；
- eager forward 次数；
- Prefill forward 次数；
- Decode forward 次数；
- 当前 GPU KV Cache 容量。

GPU 显存采样建议低频执行，而不是每个 token 查询一次。

---

## 7. 第一版输出形式

### 7.1 Console Snapshot

每隔一定 step 或一定时间输出：

```text
[Nano-vLLM Runtime]
requests:      32 finished / 64 total
queue:         waiting=8 running=24
throughput:    prefill=8240 tok/s decode=1320 tok/s
latency:       ttft_avg=86.3 ms step=7.2 ms
kv-cache:      314 / 512 blocks (61.3%)
prefix-cache:  hit_rate=42.7% saved=8192 tokens
preemption:    3
```

### 7.2 JSONL

每次 snapshot 追加一行：

```json
{
  "timestamp": 0,
  "waiting_sequences": 8,
  "running_sequences": 24,
  "prefill_tokens_per_sec": 8240,
  "decode_tokens_per_sec": 1320,
  "kv_utilization": 0.613,
  "prefix_cache_hit_rate": 0.427,
  "preemption_count": 3
}
```

这样后续可以直接用 Python/Pandas 生成实验图表。

---

## 8. 配置设计

在 `Config` 中增加：

```python
monitoring_enabled: bool = False
monitoring_interval: float = 1.0
monitoring_output: str = "console"
monitoring_jsonl_path: str | None = None
```

推荐默认：

```text
monitoring_enabled = False
```

确保原始 benchmark 不受监控逻辑影响。

---

## 9. 实施阶段

### Phase 1：指标基础设施

目标：建立统一监控模块。

任务：

- [ ] 新建 `nanovllm/monitoring/`；
- [ ] 实现 `RuntimeMetrics`；
- [ ] 实现 Counter / Gauge / Latency 基础统计；
- [ ] 实现 `snapshot()`；
- [ ] 加入 Config 开关；
- [ ] 保证关闭监控时几乎无额外开销。

验收：可以在不修改现有调度逻辑的情况下获取一个空的 Runtime Snapshot。

### Phase 2：Engine + Scheduler

任务：

- [ ] 请求生命周期时间戳；
- [ ] TTFT；
- [ ] E2E latency；
- [ ] step latency；
- [ ] Prefill throughput；
- [ ] Decode throughput；
- [ ] waiting / running queue size；
- [ ] preemption count；
- [ ] chunked prefill count。

验收：运行 `example.py` 时能输出调度与延迟数据。

### Phase 3：KV Cache + Prefix Cache

任务：

- [ ] KV block 使用率；
- [ ] block allocation/free 统计；
- [ ] prefix cache lookup/hit/miss；
- [ ] prefix cache hit rate；
- [ ] saved tokens。

验收：构造相同前缀 Prompt 后能观察到 Prefix Cache 命中率明显升高。

### Phase 4：Reporter

任务：

- [ ] ConsoleReporter；
- [ ] JsonlReporter；
- [ ] 运行结束后的 Summary Report；
- [ ] 输出平均/P50/P95 TTFT 和 E2E latency。

### Phase 5：GPU / CUDA Graph

任务：

- [ ] GPU memory gauge；
- [ ] eager forward count；
- [ ] CUDA Graph replay count；
- [ ] 各 rank 显存信息；
- [ ] 评估 NCCL 通信耗时监控方案。

### Phase 6：可视化（可选）

可选择一种：

1. Streamlit 本地 Dashboard；
2. FastAPI + WebSocket + 前端图表；
3. Prometheus + Grafana。

首选建议为 Streamlit，因为开发成本最低，最适合性能实验。

---

## 10. Benchmark / 实验设计

监控系统完成后应设计至少四组实验。

### 实验 A：监控系统自身开销

分别运行：

```text
monitoring_enabled=False
monitoring_enabled=True
```

比较：

- 总生成时间；
- Decode throughput；
- Prefill throughput。

目标：第一版核心监控开销尽量控制在 1%~3% 以内。

### 实验 B：Prefix Cache

构造：

```text
一组完全随机 Prompt
vs
一组拥有长公共前缀的 Prompt
```

观察：

```text
prefix_cache_hit_rate
saved_tokens
prefill_throughput
TTFT
```

### 实验 C：KV Cache 压力

通过提高：

```text
max_num_seqs
max_model_len
prompt length
output length
```

逐步增加 KV Cache 压力。

观察：

```text
kv_utilization
preemption_count
TTFT
throughput
```

### 实验 D：Tensor Parallel

比较：

```text
TP=1
TP=2
```

观察：

```text
throughput
latency
GPU memory per rank
```

---

## 11. 测试计划

### 单元测试

重点覆盖：

- Counter 增量正确；
- Gauge 状态正确；
- Prefix Cache hit/miss 统计正确；
- Block allocate/free 不出现负数；
- TTFT 只记录一次；
- monitoring disabled 时不改变原始逻辑。

### 集成测试

测试路径：

```text
prompt
→ add_request
→ scheduler
→ prefill
→ decode
→ finished
→ metrics snapshot
```

### 性能回归

修改监控模块后运行原有 `bench.py`，保存：

```text
baseline throughput
monitoring throughput
performance delta
```

---

## 12. 最终完成标准

第一版 Runtime Monitoring System 完成应满足：

- [ ] 能统计 TTFT、E2E、Prefill/Decode Throughput；
- [ ] 能观察 waiting/running 队列；
- [ ] 能观察 KV Cache 使用率；
- [ ] 能统计 Prefix Cache Hit Rate；
- [ ] 能统计 Preemption；
- [ ] 能导出 JSONL；
- [ ] 能输出运行结束 Summary；
- [ ] 关闭监控后不改变原始推理行为；
- [ ] 开启监控后性能损失可测量且可接受；
- [ ] README 中给出监控使用示例。

---

## 13. 推荐开发顺序

建议严格按照以下顺序推进：

```text
RuntimeMetrics
    ↓
LLMEngine step latency / throughput
    ↓
Sequence TTFT / E2E
    ↓
Scheduler queue / preemption
    ↓
BlockManager KV utilization
    ↓
Prefix Cache hit rate
    ↓
JSONL reporter
    ↓
Benchmark
    ↓
Dashboard
```

不要一开始直接做 Web Dashboard。先保证指标定义和采集逻辑正确，再做展示层。

---

## 14. 项目完成后的可展示成果

最终可以形成：

```text
Nano-vLLM Runtime Monitor

- Request-level latency tracing
- TTFT / E2E / TPOT metrics
- Prefill & Decode throughput monitoring
- Scheduler queue visualization
- KV Cache utilization tracking
- Prefix Cache hit-rate analysis
- Preemption statistics
- CUDA Graph / eager execution statistics
- JSONL benchmark export
- Runtime dashboard
```

该项目最终重点不是“做一个漂亮仪表盘”，而是建立一套能够解释 LLM 推理系统内部运行状态的可观测性机制，并利用监控数据研究调度、KV Cache、Prefix Cache 与吞吐/延迟之间的关系。
