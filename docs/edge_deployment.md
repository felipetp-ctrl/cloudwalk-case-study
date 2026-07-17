# 2.4 Edge Deployment Feasibility

## Model Summary

| Property | Full Model (deployed) | Pruned Model (not deployed) |
|---|---|---|
| Features | 57 (15 per-request + 42 session) | 32 |
| Trees | 286 | 286 |
| Max depth | 6 | 6 |
| Num leaves | 68 | 68 |
| ONNX file size | 38.0 KB | — |
| PR-AUC (test) | 1.0000 | 0.8790 |

The **full model** is the deployment candidate. Pruning to 32 features causes unacceptable degradation: PR-AUC drops from 1.0 to 0.879, credential_stuffing recall collapses to 0.0, and ddos_l7 recall drops to 0.03. The model distributes importance across many TLS-granularity session features; removing any subset breaks the redundancy needed for detection.

## Inference Benchmarks

All measurements are single-request (batch size 1), the worst-case scenario for edge inference where each HTTP request is classified independently.

### Latency

| Runtime | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|
| LightGBM Python (57f) | 0.557 | 0.635 | 0.766 |
| ONNX Runtime Python (57f) | 0.020 | 0.022 | 0.025 |
| Rust native (57f) | 0.511 | 0.527 | 0.560 |
| WASM via wasmtime (57f) | 0.650 | 0.670 | 0.698 |

All runtimes achieve p99 < 5ms, satisfying the latency constraint. The Rust/WASM numbers are the production-representative measurements — Python benchmarks include interpreter overhead that would not exist in production.

### Throughput

| Runtime | p99 (ms) | Throughput/thread | Threads for 50k req/s |
|---|---|---|---|
| ONNX Runtime Python (57f) | 0.025 | 40,814/s | 1.2 |
| Rust native (57f) | 0.560 | 1,958/s | 28.0 |
| WASM via wasmtime (57f) | 0.698 | 1,539/s | 34.9 |

ONNX Runtime Python achieves the highest single-thread throughput due to its optimized C++ backend. The Rust/WASM path trades per-thread throughput for sandboxed execution in a WASM runtime. At 50k req/s, the WASM path needs ~35 threads — achievable with 5-9 edge nodes each running 4-8 cores, which is standard for a CDN deployment.

### Memory

| Format | Full Model (57f) |
|---|---|
| Joblib PKL | 95.5 KB |
| ONNX | 38.0 KB |
| WASM binary (incl. runtime) | 12,450 KB |

The ONNX model is compact enough to load into a Cloudflare Worker's memory limit (128 MB). The WASM binary includes the full tract runtime; the model itself is only 38 KB.

## Simplification Strategy

Feature pruning was evaluated (57 → 32 features at 99% importance threshold) but **rejected**: PR-AUC drops from 1.0 to 0.879, and credential_stuffing recall collapses to 0.0. The model distributes importance across many TLS-granularity session features; removing any subset breaks detection.

If simplification were needed for a more constrained runtime, options include:
1. **Reduce `n_estimators`** from 286 to ~100-150 — reduces inference time proportionally. Can be evaluated by re-running the benchmark with `iteration_range=(0, N)` in LightGBM.
2. **Reduce `num_leaves`** from 68 to ~30 — smaller trees are faster to traverse. Requires retraining and re-evaluating.
3. **Model distillation** — train a simpler model (logistic regression or small decision tree) on the LightGBM's probability outputs. LR achieved 0.9589 PR-AUC with the same 57 features in section 2.3, making it a viable fallback if tree inference proves too heavy.
4. **Per-request-only features** — drop session features and use only 15 per-request features. Eliminates the need for per-source state at the edge, but detection quality depends on TLS fingerprints being non-spoofed.

None of these are needed given the measured latencies — the current model fits comfortably within the 5ms budget.

## Serving Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Edge Node                         │
│                                                      │
│  HTTP Request                                        │
│       │                                              │
│       ▼                                              │
│  ┌──────────────┐    ┌──────────────┐               │
│  │   Feature     │───▶│  ONNX Model  │               │
│  │  Extraction   │    │  (tract/WASM)│               │
│  │ (57 features) │    │              │               │
│  │ + session     │    │              │               │
│  │   state       │    │              │               │
│  └──────────────┘    └──────┬───────┘               │
│                             │                        │
│                       score (0-1)                     │
│                             │                        │
│                    ┌────────▼────────┐               │
│                    │  score ≥ 0.50?  │               │
│                    └────────┬────────┘               │
│                      yes /   \ no                    │
│                         /     \                      │
│              ┌─────────▼┐   ┌─▼──────────┐          │
│              │  Block /  │   │ Forward to │          │
│              │ Throttle  │   │   Origin   │          │
│              └─────┬─────┘   └─────┬──────┘          │
│                    │               │                 │
│                    └───────┬───────┘                 │
│                            ▼                         │
│                   ┌────────────────┐                 │
│                   │  Async: emit   │                 │
│                   │  score + meta  │                 │
│                   │  to log stream │                 │
│                   └────────────────┘                 │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
              ┌────────────────────┐
              │  Logging Pipeline  │
              │  (Kafka / Kinesis) │
              └────────────────────┘
```

### Feature Extraction at the Edge

The 57-feature model requires two types of features:

**Per-request features (15):** Computable from the HTTP request itself — no external lookups, O(1) string operations:

| Feature | Source | Computation |
|---|---|---|
| `header_count` | Request headers | Count of header key-value pairs |
| `ua_is_bot_library` | `User-Agent` header | Regex match against known bot libraries |
| `ua_is_browser` | `User-Agent` header | Regex match against browser signatures |
| `ua_length` / `ua_entropy` | `User-Agent` header | String length / Shannon entropy |
| `has_authorization` / `has_accept_language` / `has_referer` / `has_cookie` | Request headers | Header presence checks |
| `hour_of_day` | Request timestamp | Extract hour (0-23) |
| `path_depth` / `path_length` / `path_entropy` | URL path | Segments count / string length / Shannon entropy |
| `path_has_params` | URL path | Check for `?` or `=` |
| `is_sensitive_endpoint` | URL path | Regex match against auth/payment paths |

**Session features (42):** Require maintaining per-source state (per IP and per TLS fingerprint) with causal windowing (1m, 5m, 30m windows):

| Feature group | Windows | Metrics |
|---|---|---|
| IP-level (per source IP) | 1m, 5m, 30m | request_count, unique_paths, path_entropy, method_diversity, sensitive_endpoint_ratio, inter_request_time_mean, inter_request_time_std |
| TLS-level (per TLS fingerprint) | 1m, 5m, 30m | request_count, unique_paths, path_entropy, method_diversity, sensitive_endpoint_ratio, inter_request_time_mean, inter_request_time_std |

**Edge state requirement:** Each edge node must maintain sliding-window counters per source IP and per TLS fingerprint. This adds ~1-2 KB of state per active source, manageable within a Cloudflare Worker's memory limit. State can be stored in a local HashMap with TTL-based eviction (evict sources not seen in >30 minutes).

### Decision Logic

The cost-optimal threshold from section 2.3 is **0.022** (any threshold in the range [0.02, 0.76] achieves identical results due to the model's extremely bimodal probability distribution). At the default threshold (0.5):
- Precision: 1.0
- Recall: 1.0
- F1: 1.0
- FPR: 0.0 on test data

The extreme separation (benign mean=0.0001, malicious min=0.762) is driven by TLS fingerprints being perfect label proxies in the synthetic data. In production with TLS spoofing, the distribution would overlap more, making threshold tuning genuinely important.

The threshold is configured externally (not hardcoded in the model) so it can be adjusted without redeploying the model itself — e.g., lowered during an active attack campaign to increase recall at the cost of some precision.

## Model Update Flow

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Training   │───▶│    Model     │───▶│     CDN      │
│   Pipeline   │    │   Registry   │    │  (S3 + CF)   │
│  (central)   │    │  (S3 bucket) │    │              │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
                              ┌────────────────┼────────────────┐
                              ▼                ▼                ▼
                        ┌──────────┐    ┌──────────┐    ┌──────────┐
                        │  Edge 1  │    │  Edge 2  │    │  Edge N  │
                        │  poll /  │    │  poll /  │    │  poll /  │
                        │  5 min   │    │  5 min   │    │  5 min   │
                        └──────────┘    └──────────┘    └──────────┘
```

### Versioning

Each model version in the registry includes:
- The `.onnx` file
- Metadata: training date, dataset hash, PR-AUC, optimal threshold, feature list
- A monotonically increasing version number

Edge nodes store the current and previous model versions. On receiving a new version:
1. Load new model into memory alongside the current one
2. Run both in shadow mode for a configurable period (e.g., 10 minutes)
3. If shadow metrics are healthy, atomically switch traffic to the new model
4. Keep the previous model in memory for instant rollback

### Rollback Triggers

Automatic rollback if any of these are detected within the first hour of deployment:
- p99 inference latency > 4ms (80% of budget)
- Block rate increases by >2x compared to pre-deployment baseline
- Model returns NaN or errors on >0.01% of requests

### Canary Deployment

For high-risk model updates (e.g., new feature set, architecture change):
1. Route 1% of traffic to the new model
2. Compare precision/recall estimates against the incumbent
3. Ramp to 10% → 50% → 100% over 4 hours if metrics hold
4. Abort and rollback if any rollback trigger fires

## Production Monitoring

### Metrics to Track

| Metric | How | Alert Threshold |
|---|---|---|
| Inference latency (p99) | Histogram per edge node, 1-min buckets | > 3ms (2ms buffer for feature extraction) |
| Score distribution | Histogram of model output scores, 5-min buckets | KS-test p-value < 0.01 vs training distribution |
| Block rate | % of requests blocked per 5-min window | > 2x rolling 1-hour baseline |
| Error rate | Model errors / total inferences | > 0.01% |
| Feature drift | PSI per feature, hourly | PSI > 0.2 on any feature |

### Concept Drift Detection

The assessment states "attack patterns change weekly." Two complementary approaches:

1. **Feature-level drift (PSI):** Compare the distribution of each input feature between the training data and the last 24 hours of live traffic. Population Stability Index > 0.2 triggers a retraining alert. This catches distributional shifts in traffic patterns (e.g., a new bot framework with different user-agent lengths).

2. **Score-level drift (KS test):** Compare the distribution of model output scores between the training period and recent traffic. A significant shift (p < 0.01) suggests the model's internal representation no longer matches reality — even if individual features haven't shifted much.

Both fire alerts to the ML team, who can then:
- Inspect recent traffic samples
- Trigger a retraining pipeline with fresh labels
- Deploy a new model version through the update flow above

### Feedback Loop

```
Edge decision (block/allow)
        │
        ▼
Logging Pipeline (Kafka)
        │
        ├──▶ Blocked requests → security team review (weekly sample)
        │         └──▶ estimated FPR
        │
        └──▶ Allowed requests → origin WAF + fraud detection
                  └──▶ attacks that passed through → new training labels
```

This loop addresses the "labels are noisy and delayed" constraint: downstream defenses provide ground-truth labels 1-3 days later, which feed back into the next retraining cycle.
