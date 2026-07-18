# Technical Writeup

## A Note on the Synthetic Dataset

Before presenting results, it is important to flag a critical characteristic of the synthetic data that frames everything below: **each TLS fingerprint in the dataset is 100% malicious or 100% benign**. No fingerprint is shared between attack and legitimate traffic. This means TLS-granularity session features (e.g., `tls_30m_request_count`) act as perfect label proxies — the model identifies attack *tools* by their fingerprint, not attack *behavior* by its patterns.

The PR-AUC of 1.0 reported in Section 2.3 is technically correct (no data leakage), but it reflects this dataset property, not genuine behavioral detection. In production, where attackers use TLS fingerprint spoofing (JA3 randomization via tools like curl-impersonate), the model's performance would degrade to somewhere between:

- **Floor (~0.71 PR-AUC):** Attackers fully randomize TLS fingerprints — only IP + per-request features remain.
- **Ceiling (1.0 PR-AUC):** Attackers use default tool fingerprints — TLS identity provides trivial separation.

This caveat applies to all results below. The pipeline, feature engineering, and deployment architecture are designed to be robust regardless — behavioral features provide the fallback signal when TLS identity becomes unreliable.

---

## Part 2 — Practical Implementation

### 2.1 — Exploratory Analysis & Label Joining

The `incident_labels` table uses three identifier types (IP address, CIDR range, TLS fingerprint) with temporal bounds (`active_from` / `active_until`), while `http_requests` has individual request-level data. The joining pipeline ([`src/label_joining.py`](src/label_joining.py)) handles all three:

1. **IP exact match** — join on `source_ip = source_identifier` where `timestamp` falls within the incident's active window.
2. **CIDR range match** — for each `ip_range` label, check whether the request's `source_ip` belongs to the network using Python's `ipaddress` module, with the same temporal filter.
3. **TLS fingerprint match** — join on `tls_fingerprint = source_identifier` with temporal bounds.

A request is labeled malicious if it matches *any* incident through any of the three strategies. The result: **592 malicious (1.18%) vs 49,408 benign (98.82%)**.

**Labeling gaps:** Label absence does not mean benign — it means "not identified as part of a known incident." The labeling delay ranges from 0.1 to 2.9 days, meaning attacks in the last 1-3 days of the dataset may be unlabeled. The dataset's 1.18% attack rate also diverges from the assessment constraint of <0.1%, which affects how class imbalance strategies should be calibrated for production.

For full details, see [`notebooks/eda.ipynb`](notebooks/eda.ipynb).

### 2.2 — Feature Engineering

The feature pipeline ([`src/features.py`](src/features.py)) extracts two categories of features:

**Per-request features (15):** Computed from the HTTP request alone — path characteristics (`path_depth`, `path_length`, `path_entropy`, `path_has_params`), user-agent signals (`ua_length`, `ua_entropy`, `ua_is_browser`, `ua_is_bot_library`), header presence flags (`header_count`, `has_accept_language`, `has_referer`, `has_cookie`, `has_authorization`), temporal (`hour_of_day`), and endpoint sensitivity (`is_sensitive_endpoint`).

**Session features (42 active):** Aggregated at two granularities (per source IP and per TLS fingerprint) across three time windows (1min, 5min, 30min). Seven base metrics per variant: `request_count`, `unique_paths`, `path_entropy`, `method_diversity`, `sensitive_endpoint_ratio`, `inter_request_time_mean`, `inter_request_time_std`. Total: 7 × 3 × 2 = 42.

**Causal windowing:** Each request's session features aggregate only *prior* requests within the same time window — the current request is never included in its own aggregates. This matches the real-time edge computation model and prevents data leakage. Implemented via `cumcount()`, `expanding().mean().shift(1)`, and custom causal functions for nunique/entropy.

**Excluded features:** 36 server-response session features (aggregates of `status_code`, `response_time_ms`, `body_size_bytes`) were excluded because they encode past WAF decisions, creating a circular dependency with the labels. 4 frequency-encoded categoricals were excluded because they leak test distribution into training. 3 server-response per-request features (`status_code_group`, `response_time_ms`, `body_size_bytes`) were excluded because they are unavailable at inference time (the model classifies the request *before* the server responds).

For justification of each feature choice, see [`docs/2.2-feature-engineering-decisions.md`](docs/2.2-feature-engineering-decisions.md).

### 2.3 — Baseline Model

Four models were trained in a simple → complex progression ([`src/model.py`](src/model.py)):

| Model | PR-AUC | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|
| Logistic Regression | 0.9589 | 1.0000 | 0.9265 | 0.9619 | 0.0000 |
| Random Forest | 1.0000 | 1.0000 | 0.5592 | 0.7173 | 0.0000 |
| XGBoost | 1.0000 | 1.0000 | 0.9837 | 0.9918 | 0.0000 |
| **LightGBM** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.0000** |

**LightGBM was selected** based on (PR-AUC, F1) tiebreaker: all tree models achieve PR-AUC 1.0, but only LGBM reaches F1 1.0 with perfect recall across all three test attack types (zero_day_exploit, credential_stuffing, ddos_l7).

**Class imbalance** was handled via `scale_pos_weight` (proportional to the ~83:1 imbalance ratio) combined with confidence-based sample weights from `incident_labels.confidence` (`high=1.0`, `medium=0.6`, `low=0.3`). SMOTE was rejected because the 592 malicious samples span 5 heterogeneous attack types — interpolating between DDoS and credential stuffing samples produces synthetic examples that represent no real attack pattern.

**Temporal train/test split:** Train on days 6-9, test on days 10-12 (cutoff at January 10). This ensures no future data leaks into training. Notably, `zero_day_exploit` appears only in the test set — the model has never seen this attack type during training but detects it via its unique TLS fingerprint. Hyperparameters were tuned with Optuna using expanding-window temporal cross-validation (3 folds).

**Feature pruning** from 57 to 32 features (99% cumulative importance threshold) was evaluated but rejected: PR-AUC drops from 1.0 to 0.879, with credential_stuffing recall collapsing to 0.0. The model distributes importance across many TLS-granularity features; removing any subset breaks detection.

For full analysis including per-attack-type breakdown and threshold analysis, see [`docs/2.3-baseline-model-decisions.md`](docs/2.3-baseline-model-decisions.md).

### 2.4 — Edge Deployment Feasibility

**Inference latency** — all runtimes satisfy the <5ms constraint:

| Runtime | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|
| LightGBM Python | 0.557 | 0.635 | 0.766 |
| ONNX Runtime Python | 0.020 | 0.022 | 0.025 |
| Rust native (tract) | 0.511 | 0.527 | 0.560 |
| **WASM via wasmtime** | **0.650** | **0.670** | **0.698** |

The WASM numbers are the production-representative measurements. WASM adds ~27% overhead vs native Rust due to sandboxing, but remains well within budget.

**Memory footprint** — the ONNX model is 38 KB. The WASM binary (including the tract runtime) is ~12 MB. Both fit comfortably within edge runtime memory limits (e.g., Cloudflare Workers: 128 MB).

**Export and serving:** The model is exported to ONNX via `onnxmltools` ([`src/export.py`](src/export.py)), then loaded by a Rust binary using `tract-onnx` compiled to `wasm32-wasip1` ([`edge-inference/src/main.rs`](edge-inference/src/main.rs)). The `ZipMap` ONNX operator was disabled (`zipmap=False`) for tract compatibility, and numerical validation confirms max absolute error of 9.08e-08 between LightGBM and ONNX predictions.

**Serving architecture:**

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
```

**Model updates** use a pull-based flow: edge nodes poll the model registry every 5 minutes, run new and old models in shadow mode for 10 minutes, then atomically switch. Automatic rollback triggers fire if p99 latency exceeds 4ms, block rate doubles, or error rate exceeds 0.01%.

**Production monitoring** combines PSI per feature (hourly, threshold > 0.2) for feature-level drift, KS test on output scores (5-minute buckets, p < 0.01) for score-level drift, and a feedback loop where blocked requests are sampled for security team review and allowed-but-attacked requests provide new training labels.

For full details including canary deployment, rollback triggers, and monitoring thresholds, see [`docs/2.4-edge-deployment-decisions.md`](docs/2.4-edge-deployment-decisions.md) and [`docs/edge_deployment.md`](docs/edge_deployment.md).

---

## Part 3 — Tradeoff Deep-Dives

### 3.1 — Stateless Edge, Stateful Signals

The 42 session features require per-source sliding-window state, but edge nodes are stateless and requests from the same source may hit different nodes. The proposed architecture uses **consistent hashing on `source_ip` at the load balancer** to route all requests from the same IP to the same edge node. This is not a novel technique — CDN providers like Cloudflare and Fastly already use consistent hashing for cache affinity, and the same mechanism provides session affinity for free.

Each edge node maintains a **local `HashMap<SourceKey, SessionCounters>`** with TTL-based eviction (30 minutes, matching the longest feature window). Each entry stores 42 counters (7 metrics × 3 windows × 2 granularities), consuming ~1-2 KB per active source. With 10,000 concurrently active sources per node, total state is ~10-20 MB — negligible against a typical edge runtime's memory budget. Session lookups are O(1) hash table operations with zero network latency, keeping the total budget at ~1ms feature extraction + ~1ms inference = 2ms, with 3ms headroom.

**Graceful degradation when state is unavailable:** If an edge node restarts or hits memory pressure, the session state is lost. The model falls back to the 15 per-request features only, with PR-AUC dropping to ~0.71 — degraded but still functional. State repopulates organically as new requests arrive; full session capacity is restored within 30 minutes (the longest window). This is an acceptable tradeoff: brief degradation on a single node does not compromise the system globally, and per-request features still catch the most obvious attacks (bot user-agents, sensitive endpoint targeting).

**Cross-node gap:** IP rotation by attackers causes requests to land on different nodes, breaking session continuity. Two mitigations: (1) TLS fingerprints are more stable than IPs — if the load balancer also hashes on TLS fingerprint, session state persists even as IPs rotate; (2) the most damaging attack types (DDoS, credential stuffing) require high request volume from fixed IPs within short windows, which the 1-minute and 5-minute features capture before the attacker rotates.

### 3.2 — The Labeling Bottleneck

The core problem is a 1-3 day gap between a novel attack starting and having labeled data for it. The supervised model, by definition, cannot detect what it has never seen. The strategy is a **three-layer hybrid pipeline** where each layer covers a different speed/accuracy tradeoff:

1. **Supervised model (LGBM)** — the primary defense. Handles all known attack patterns with high precision. Retrained weekly with the latest labels from post-incident forensics and WAF rule triggers. This is the only layer that makes block/allow decisions.

2. **Statistical anomaly detector** — runs in parallel, flag-only (never blocks). Computes z-scores on session features (`request_count`, `inter_request_time_std`, `unique_paths`, `sensitive_endpoint_ratio`) against a rolling 24-hour benign baseline. Sources exceeding z > 3σ on multiple features simultaneously are flagged as anomaly candidates. No labels required — this layer detects distributional outliers regardless of whether the pattern has been seen before.

3. **Feedback loop** — flagged anomalies are routed to the security team's review queue with priority proportional to the anomaly severity. The team confirms or rejects each candidate. Confirmations become training labels for the next retraining cycle. Rejections calibrate the z-score thresholds to reduce future false flags.

**The handoff timeline for a novel attack:** The anomaly detector flags unusual traffic within hours (once enough requests accumulate to produce statistically significant z-scores). The security team investigates and confirms — producing the first labels within ~24 hours. The next scheduled retrain (or an emergency retrain triggered by the alert) incorporates these labels, and the supervised model covers the new pattern going forward. During the initial hours before the anomaly detector has enough data, downstream defenses (origin WAF rules, fraud detection) serve as the backstop — the assessment explicitly states that "false negatives are partially mitigated by downstream defenses."

### 3.3 — Threshold Economics

**Parameters:** 100M requests/day, 0.05% malicious (50,000 attacks, 99,950,000 benign). FP cost = $2.50, FN cost = $0.10.

**Expected daily cost at a given operating point:**

```
Daily cost = FPR × 99,950,000 × $2.50  +  (1 - Recall) × 50,000 × $0.10
           = FPR × $249,875,000  +  FNR × $5,000
```

The **asymmetry is extreme**: each 0.001% increase in FPR costs $2,499/day, while catching all remaining attacks saves at most $5,000/day total. The optimal operating point maximizes recall subject to a precision floor.

**Minimum viable precision** — the threshold below which blocking costs more than it saves:

```
precision_min = (π × cost_FN) / ((1-π) × cost_FP + π × cost_FN)
              = (0.0005 × 0.10) / (0.9995 × 2.50 + 0.0005 × 0.10)
              = 0.00005 / 2.49880
              ≈ 0.002%
```

At precision 0.002%, the cost of false positives exactly equals the savings from true positives. Any precision above this is economically justified. In practice, even at 1% precision (blocking 100 benign requests for every 1 attack caught), the net savings are positive: the system should **maximize recall aggressively**.

**With FN cost at $5.00** (active credential stuffing campaign):

```
Daily cost = FPR × $249,875,000  +  FNR × $250,000
precision_min = (0.0005 × 5.00) / (0.9995 × 2.50 + 0.0005 × 5.00)
              = 0.0025 / 2.50125
              ≈ 0.1%
```

The minimum precision threshold shifts from 0.002% to 0.1% — still very low, meaning the economics still heavily favor recall. The FN cost would need to exceed ~$5,000 per request before precision becomes the binding constraint.

**Practical caveat:** The $2.50 FP cost captures only direct lost revenue. Blocking a legitimate customer on a payment flow also creates support tickets, churn risk, and reputational damage — costs that are real but harder to quantify. A production system should set the precision floor higher than the pure economic optimum to account for these indirect costs.

### 3.4 — Adversarial Robustness

When the attacker randomizes TLS fingerprints and adds timing jitter, the model loses its two strongest signal families. Recall drops from 92% to 41%. The response operates on three timescales:

**Immediate (hours):** Lower the blocking threshold to recover partial recall at the cost of more false positives — the threshold economics from Section 3.3 show this is almost always net-positive. Simultaneously, activate the threat intelligence feed as a supplementary signal: it covers ~30% of attacks with high precision and requires no model changes. Enable aggressive rate limiting on sensitive endpoints (`/auth`, `/login`, `/payment`, `/tokenize`) as a WAF-level fallback. Alert the security team to investigate the evasion pattern and begin labeling the new attack traffic.

**Short-term (days):** Retrain the model *excluding* all TLS features, forcing it to learn behavioral patterns: timing regularity (`inter_request_time_std`), endpoint diversity (`unique_paths`), request volume (`request_count`), and sensitive endpoint targeting (`sensitive_endpoint_ratio`). The IP + per-request feature floor is ~0.71 PR-AUC on the current dataset, but focused retraining with behavioral emphasis — combined with the new labeled data from the security team's investigation — should improve this. Additionally, introduce spoofing-resistant features that are harder to fake than JA3: HTTP/2 frame ordering, header canonicalization order, and TCP/IP stack fingerprinting (window size, TTL, TCP options).

**Long-term (weeks):** Adopt an **ensemble architecture** where a fast per-request model (15 features, <0.1ms) runs as a first pass on all traffic, and the full session-based model runs only on requests that score above a low suspicion threshold. This reduces dependence on any single feature family and limits the blast radius of feature evasion. Incorporate **adversarial training** — inject TLS-randomized attack samples during training so the model learns to classify without relying on TLS identity. Finally, implement **feature rotation**: periodically vary which features the model weights most heavily, increasing the cost for attackers to reverse-engineer and evade the detection logic.

---

## Production Readiness

### CI/CD Pipeline

The project includes two GitHub Actions workflows ([`.github/workflows/`](.github/workflows/)):

**`ci.yml`** runs on every push and pull request:
1. **Lint** — `ruff check` + `ruff format --check` for consistent code style.
2. **Test** — full pytest suite (53 tests) covering label joining, features, model, export, monitoring.
3. **ONNX Validation** — trains a model from scratch, exports to ONNX, and validates numerical equivalence. Also asserts minimum metric thresholds (PR-AUC > 0.85, F1 > 0.80) to catch regressions.
4. **Rust Build** — compiles the edge-inference binary to verify Rust code integrity.

**`model-validation.yml`** runs on-demand (`workflow_dispatch`) for full model validation:
- Runs Optuna hyperparameter tuning (30 trials) with temporal cross-validation.
- Evaluates against configurable metric thresholds (PR-AUC, F1).
- Runs data drift detection and prediction stability analysis.
- Exports the validated ONNX model and a JSON report as GitHub artifacts.

This pipeline validates the entire chain — from raw data to deployable ONNX model — ensuring that code changes never silently degrade model quality.

### Model Monitoring

The monitoring module ([`src/monitoring.py`](src/monitoring.py)) implements three production monitoring patterns:

**Data drift detection** — Kolmogorov-Smirnov test per feature between reference (training) and current (production) distributions. Features where p < 0.05 are flagged as drifted. In a security context, drift in session features like `request_count` or `inter_request_time_std` can signal new attack patterns that the model hasn't been trained on.

**Prediction stability** — Population Stability Index (PSI) between reference and current score distributions, plus alert rate tracking. PSI < 0.1 indicates stable predictions; PSI > 0.25 signals significant shift requiring investigation. For an edge security model, a sudden spike in alert rate could mean either a real attack wave or model degradation — the PSI helps distinguish between the two.

**Performance tracking** — sliding-window computation of precision, recall, and FPR with degradation detection against baseline metrics. Windows where metrics drop below 90% of baseline are flagged. This catches gradual model decay that per-request metrics would miss.

The `generate_monitoring_report()` function consolidates all signals into a single report with a `needs_retraining` flag, enabling automated retraining triggers in production. All monitoring functions are covered by 14 unit tests simulating healthy, drifted, and degraded scenarios.
