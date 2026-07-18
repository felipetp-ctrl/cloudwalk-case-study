# Technical Writeup

## Note on TLS Feature Removal

TLS-granularity session features (e.g., `tls_30m_request_count`) were removed from the model because they create a **circular dependency with the labels**: incident labels are partially assigned by TLS fingerprint matching, which means TLS-based session features act as near-perfect label proxies. The model would learn to identify attack *tools* by their fingerprint rather than attack *behavior* by its patterns. After removing these features, the model uses 36 features (15 per-request + 21 IP-level session features) and reflects genuine behavioral detection capability.

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

**Session features (21):** Aggregated at **IP granularity only** across three time windows (1min, 5min, 30min). Seven base metrics per window: `request_count`, `unique_paths`, `path_entropy`, `method_diversity`, `sensitive_endpoint_ratio`, `inter_request_time_mean`, `inter_request_time_std`. Total: 7 × 3 × 1 = 21. TLS-granularity session features (an additional 21) were removed because they create a circular dependency with labels — since labels are partially assigned by TLS fingerprint matching, TLS-based session aggregates become near-perfect proxies for the label itself. The model now captures behavioral patterns (request volume, timing, endpoint targeting) rather than tool identity.

**Causal windowing:** Each request's session features aggregate only *prior* requests within the same time window — the current request is never included in its own aggregates. This matches the real-time edge computation model and prevents data leakage. Implemented via `cumcount()`, `expanding().mean().shift(1)`, and custom causal functions for nunique/entropy.

**Excluded features:** 36 server-response session features (aggregates of `status_code`, `response_time_ms`, `body_size_bytes`) were excluded because they encode past WAF decisions, creating a circular dependency with the labels. 3 server-response per-request features (`status_code_group`, `response_time_ms`, `body_size_bytes`) were excluded because they are unavailable at inference time (the model classifies the request *before* the server responds).

For justification of each feature choice, see [`docs/2.2-feature-engineering-decisions.md`](docs/2.2-feature-engineering-decisions.md).

### 2.3 — Baseline Model

Four models were trained in a simple → complex progression ([`src/model.py`](src/model.py)), evaluated with temporal split (train days 6-9, test days 10-12) and 36 features. Precision, Recall, F1, and FPR are reported at the optimal F1 threshold; PR-AUC and ROC-AUC are threshold-independent:

| Model | CV PR-AUC | Test PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.5432 | 0.0123 | 0.0689 | 0.1091 | 0.0245 | 0.0400 | 0.002345 |
| Random Forest | 0.7771 | 0.8086 | 0.9850 | 0.7044 | 0.7878 | 0.7437 | 0.003876 |
| XGBoost | 0.8894 | 0.6206 | 0.8713 | 0.9797 | 0.5918 | 0.7379 | 0.000144 |
| **LightGBM** | **0.9441** | **0.8138** | **0.9871** | **0.9658** | **0.5755** | **0.7212** | **0.000239** |

**LightGBM was selected** based on PR-AUC: it achieves the highest test PR-AUC (0.8138) — the primary metric for imbalanced classification — indicating substantially better ranking quality across all thresholds than XGBoost (0.6206) or Random Forest (0.8086). At the optimal F1 threshold, LGBM maintains high precision (0.97) with moderate recall (0.58).

**Per-attack recall at threshold 0.5:**

- **credential_stuffing: 87% recall** — the model genuinely learns behavioral patterns (many login attempts from the same IP, high sensitive endpoint ratio, regular timing). This is real behavioral detection.
- **zero_day_exploit: 0% recall** — only 6 samples, never seen in training, and without TLS fingerprint as a shortcut, there is no distinctive IP-level behavior to distinguish these from benign traffic.
- **ddos_l7: 0% recall** — distributed attack where each participating IP sends only a few requests. Per-IP session features cannot capture the coordinated multi-IP nature of the attack. Individual requests look normal; it is the aggregate across many IPs that constitutes the attack.

**Class imbalance** was handled via `scale_pos_weight=138.27` combined with confidence-based sample weights from `incident_labels.confidence` (`high=1.0`, `medium=0.6`, `low=0.3`). SMOTE was rejected because the 592 malicious samples span 5 heterogeneous attack types — interpolating between DDoS and credential stuffing samples produces synthetic examples that represent no real attack pattern. Hyperparameters were tuned with Optuna using expanding-window temporal cross-validation (3 folds). Full best parameters: `n_estimators=589, num_leaves=66, max_depth=8, learning_rate=0.0380, subsample=0.687, colsample_bytree=0.641, min_child_samples=44, scale_pos_weight=138.27, reg_alpha=0.036, reg_lambda=0.001`.

**Feature pruning** from 36 to 30 features (99% cumulative importance threshold): PR-AUC drops only 0.024 (from 0.8138 to 0.7894). This is much less lossy than before TLS feature removal, because importance is now distributed across genuine behavioral features rather than concentrated in a few TLS-identity proxies. Pruning is a viable option if latency constraints require it.

**Threshold analysis** shows that the default threshold of 0.5 yields perfect precision (1.0) but moderate recall (0.4898). Lowering the threshold to 0.144 (cost-optimal for $2.50 FP / $0.10 FN) recovers some recall (0.5102) while maintaining precision at 1.0. Even at an aggressive threshold of 0.01 (high FN cost scenario, $5.00 per missed attack), recall reaches only 0.5551 with precision at 0.9714. The fundamental ceiling is the model's inability to detect distributed attacks (DDoS) and novel attacks (zero-day) using per-IP features alone — no threshold adjustment can recover signal that is not in the features.

#### Source-Level Classifier (DDoS Detection)

Analysis of the DDoS L7 pattern revealed that the attack signal lives at the **source level**, not the request level. DDoS IPs exhibit a distinctive behavioral profile that is invisible on a per-request basis but obvious in aggregate:

| Characteristic | DDoS IPs | Benign IPs |
|---|---|---|
| Unique paths visited | **always 1** (`/api/v1/transactions`) | median 8 |
| Path concentration | **1.0** (100%) | 0.125 (12.5%) |
| Methods per path | **4.0** (PUT/HEAD/DELETE/GET) | 0.33 |
| HEAD usage | 20.6% | 1.6% (13x higher) |

A **source-level classifier** ([`src/source_model.py`](src/source_model.py)) aggregates per-request data into behavioral profiles and classifies sources after they accumulate at least 2 requests. The model is **trained on IP-day aggregates** because that is the granularity at which incident labels are available (`active_from`/`active_until` spans). In **production**, the same 13 features are computed over the same sliding windows used by Tier 1 (1min, 5min, 30min) — after an IP accumulates ≥2 requests in any window, the source model evaluates its behavioral profile and applies the verdict to subsequent requests. The features are proportions and ratios (`path_concentration`, `method_per_path`, `sensitive_ratio`) that are invariant to window size — the behavioral signature of a DDoS IP (single-endpoint focus, unusual method mix) appears whether computed over 1 minute or 1 day. The threshold of 2 (rather than 5) reflects that DDoS IPs in this dataset send a median of only 4 requests per IP-day — higher thresholds miss most of the attack surface.

**Two-tier ensemble results** (source scores propagated back to requests, combined with request-level model via `max(request_prob, source_prob)`):

| Approach | PR-AUC | Precision | Recall | F1 | DDoS Recall | CS Recall |
|---|---|---|---|---|---|---|
| Request-level only (Tier 1) | 0.8138 | 0.966 | 0.576 | 0.721 | 0% | 87% |
| **Two-tier ensemble** (min_requests=2) | **0.9892** | **0.983** | **0.951** | **0.967** | **94%** | **100%** |

The source-level model recovers DDoS detection from 0% to 94% recall by operating at the right granularity — classifying IPs by aggregate behavior rather than individual requests. The ensemble's 4 false positive requests across ~21,000 test requests (FPR = 0.02%) is an acceptable trade-off given the $2.50 FP cost against the DDoS recall gain. The remaining 6% missed DDoS comes from IPs with only 1 request in their active day, which cannot be profiled at the source level and fall through to downstream defenses.

**`min_requests` sensitivity analysis:**

| min_requests | PR-AUC | Precision | DDoS Recall | FP |
|---|---|---|---|---|
| **2** | **0.9892** | 0.983 | **94%** | 4 |
| 3 | 0.9443 | 0.974 | 86% | 6 |
| 4 | 0.9112 | 1.000 | 78% | 0 |
| 5 | 0.7821 | 1.000 | 47% | 0 |

**Production integration:** At the edge, the request-level model (Tier 1) scores every request immediately (<1ms). In parallel, per-IP counters accumulate across the same three sliding windows used by Tier 1's session features (1min, 5min, 30min). After an IP reaches ≥2 requests in any window, the source-level model (Tier 2) computes 13 behavioral features over that window's requests and classifies the IP. If flagged, the IP is added to an in-memory blocklist — all subsequent requests are blocked without per-request inference. Shorter windows (1min) catch fast attacks like aggressive credential stuffing; longer windows (30min) catch slower distributed patterns. This two-tier approach provides instant coverage for known patterns (Tier 1) and progressively stronger coverage for distributed attacks (Tier 2) as behavioral evidence accumulates across each window.

For full analysis including per-attack-type breakdown and threshold analysis, see [`docs/2.3-baseline-model-decisions.md`](docs/2.3-baseline-model-decisions.md).

### 2.4 — Edge Deployment Feasibility

**Inference latency** — both tiers satisfy the <5ms constraint:

| Model | Runtime | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|
| Tier 1 (36f) | LightGBM Python | 0.515 | 0.552 | 0.621 |
| Tier 1 (36f) | ONNX Runtime Python | 0.027 | 0.031 | 0.051 |
| Tier 2 (13f) | LightGBM Python | 0.461 | 0.596 | 0.694 |
| Tier 2 (13f) | ONNX Runtime Python | 0.018 | 0.020 | 0.036 |

Combined worst-case (both tiers sequential): 0.087ms — well within budget. In practice, Tier 2 runs asynchronously when an IP accumulates ≥2 requests within any sliding window (1min, 5min, 30min), so it does not add to the critical path. ONNX single-request p99 of 0.051ms (Tier 1) means 2.6 threads handle 50k req/s; Tier 2 needs only 1.8 threads.

**Memory footprint** — Tier 1 ONNX is 766.7 KB, Tier 2 ONNX is 384.3 KB, combined 1.15 MB. The WASM binary (including the tract runtime) is ~12 MB. Both fit comfortably within edge runtime memory limits (e.g., Cloudflare Workers: 128 MB). Runtime peak memory is 11.3 KB (Tier 1) + 1.8 KB (Tier 2) per inference.

**Export and serving:** Both models are exported to ONNX via `onnxmltools` ([`src/export.py`](src/export.py)), then loaded by a Rust binary using `tract-onnx` compiled to `wasm32-wasip1` ([`edge-inference/src/main.rs`](edge-inference/src/main.rs)). The `ZipMap` ONNX operator was disabled (`zipmap=False`) for tract compatibility, and numerical validation confirms max absolute error of 4.98e-07 (Tier 1) and 1.48e-07 (Tier 2) between LightGBM and ONNX predictions.

**Serving architecture (two-tier):**

```
┌───────────────────────────────────────────────────────────────┐
│                          Edge Node                            │
│                                                               │
│  HTTP Request                                                 │
│       │                                                       │
│       ▼                                                       │
│  ┌─────────────┐    ┌──────────────────┐                      │
│  │ IP Blocklist │───▶│ IP flagged?       │                     │
│  │  (in-memory) │    │ (source model)    │                     │
│  └─────────────┘    └────────┬─────────┘                      │
│                        yes /   \ no                            │
│                           /     \                              │
│               ┌──────────▼┐   ┌─▼────────────┐                │
│               │  Block    │   │  Request-Level│                │
│               │  (fast)   │   │  ONNX Model   │                │
│               └─────┬─────┘   │  (36 features)│                │
│                     │         └────────┬──────┘                │
│                     │            score (0-1)                   │
│                     │                 │                         │
│                     │        ┌────────▼────────┐               │
│                     │        │  score ≥ 0.50?  │               │
│                     │        └────────┬────────┘               │
│                     │          yes /   \ no                    │
│                     │             /     \                       │
│                     │  ┌─────────▼┐   ┌─▼──────────┐          │
│                     │  │  Block / │   │ Forward to  │          │
│                     │  │ Throttle │   │   Origin    │          │
│                     │  └────┬─────┘   └─────┬──────┘          │
│                     │       │               │                  │
│                     └───┬───┴───────────────┘                  │
│                         ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Update IP counters (shared with Tier 1 session state)  │   │
│  │  Windows: 1min, 5min, 30min                             │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  IP reached ≥ 2 requests in any window?                 │   │
│  │  → compute 13 source features over that window          │   │
│  │  → source model classifies IP                           │   │
│  │  → if malicious: add to blocklist                       │   │
│  │                                                         │   │
│  │  1min:  catches fast attacks (aggressive brute-force)   │   │
│  │  5min:  catches moderate-speed patterns                 │   │
│  │  30min: catches slow/distributed attacks (DDoS L7)      │   │
│  └─────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
```

**Model updates** use a pull-based flow: edge nodes poll the model registry every 5 minutes, run new and old models in shadow mode for 10 minutes, then atomically switch. Automatic rollback triggers fire if p99 latency exceeds 4ms, block rate doubles, or error rate exceeds 0.01%.

**Production monitoring** combines PSI per feature (hourly, threshold > 0.2) for feature-level drift, KS test on output scores (5-minute buckets, p < 0.01) for score-level drift, and a feedback loop where blocked requests are sampled for security team review and allowed-but-attacked requests provide new training labels.

For full details including canary deployment, rollback triggers, and monitoring thresholds, see [`docs/2.4-edge-deployment-decisions.md`](docs/2.4-edge-deployment-decisions.md).

---

## Part 3 — Tradeoff Deep-Dives

### 3.1 — Stateless Edge, Stateful Signals

The 21 session features require per-IP sliding-window state, but edge nodes are stateless and requests from the same source may hit different nodes. The proposed architecture uses **consistent hashing on `source_ip` at the load balancer** to route all requests from the same IP to the same edge node. This is not a novel technique — CDN providers like Cloudflare and Fastly already use consistent hashing for cache affinity, and the same mechanism provides session affinity for free.

Each edge node maintains a **local `HashMap<SourceIP, SessionCounters>`** with TTL-based eviction (30 minutes, matching the longest feature window). Each entry stores counters for both tiers: 21 Tier 1 session counters (7 metrics × 3 windows) plus the raw data needed to compute Tier 2's 13 source-level features across the same three windows — consuming ~0.5-1 KB per active source. Both tiers share this state: Tier 1 reads its session features on every request, and Tier 2 computes its behavioral profile when an IP accumulates ≥2 requests in any window. With 10,000 concurrently active sources per node, total state is ~5-10 MB — negligible against a typical edge runtime's memory budget. Session lookups are O(1) hash table operations with zero network latency, keeping the total budget at ~1ms feature extraction + ~1ms inference = 2ms, with 3ms headroom.

**Graceful degradation when state is unavailable:** If an edge node restarts or hits memory pressure, the session state is lost. Tier 1 falls back to the 15 per-request features only, and Tier 2 is temporarily disabled (no accumulated behavior to evaluate). State repopulates organically as new requests arrive; Tier 2 becomes active again within seconds (after ≥2 requests from the same IP), and full 30-minute session capacity is restored within 30 minutes. This is an acceptable tradeoff: brief degradation on a single node does not compromise the system globally, and per-request features still catch the most obvious attacks (bot user-agents, sensitive endpoint targeting).

**Cross-node gap:** IP rotation by attackers causes requests to land on different nodes, breaking session continuity. With TLS-granularity features removed, the primary mitigation is the time window structure itself: the most damaging attack types that the model can detect (credential stuffing) require high request volume from fixed IPs within short windows, which the 1-minute and 5-minute features capture before the attacker rotates. For distributed attacks like L7 DDoS — where IP rotation is inherent to the attack pattern — the model's per-IP features are fundamentally insufficient, and defense relies on origin-side rate limiting and WAF rules (see Section 3.2).

### 3.2 — The Labeling Bottleneck

The assessment mentions a threat intelligence feed covering ~30% of known attacks. The remaining 70% require detection without pre-existing labels. The request-level model's recall of 49% at threshold 0.5 reflects this gap: it detects credential stuffing well (87% recall) but misses distributed attacks (DDoS, 0% recall) and novel attack types (zero-day exploit, 0% recall). The source-level classifier (Section 2.3) recovers DDoS recall to 94% by classifying IPs after observing 2+ requests, but the unsupervised/hybrid strategy remains necessary for truly novel attacks. A single supervised model cannot close the 70% gap alone. The strategy is a **multi-layer hybrid pipeline** where each layer covers a different segment of the threat landscape:

1. **Supervised model (LGBM)** — the primary defense for known behavioral patterns. Handles credential stuffing and similar attacks where per-IP session features carry strong signal: high request counts, regular timing, concentrated endpoint targeting. Retrained weekly with the latest labels from post-incident forensics and WAF rule triggers. This layer makes block/allow decisions at the edge.

2. **Threat intelligence feed** — provides high-precision signals for ~30% of attacks using known-bad IPs, CIDR ranges, and attack signatures. Operates as a complementary layer: requests matching threat intel are blocked regardless of model score. This covers attacks the model might miss (especially distributed attacks coordinated from known botnets) without introducing false positives.

3. **Statistical anomaly detector** — runs in parallel, flag-only (never blocks). Computes z-scores on session features (`request_count`, `inter_request_time_std`, `unique_paths`, `sensitive_endpoint_ratio`) against a rolling 24-hour benign baseline. Sources exceeding z > 3σ on multiple features simultaneously are flagged as anomaly candidates. No labels required — this layer detects distributional outliers regardless of whether the pattern has been seen before.

4. **Feedback loop** — flagged anomalies and threat intel matches are routed to the security team's review queue with priority proportional to severity. Confirmations become training labels for the next retraining cycle. Rejections calibrate the z-score thresholds and threat intel quality scores. Over time, this loop closes the gap: anomalies confirmed as attacks become supervised training data, expanding the model's coverage beyond the initial 30% of known patterns.

**The DDoS gap (partially addressed):** The source-level classifier (Section 2.3) recovers DDoS L7 recall from 0% to 94% by classifying IPs after observing 2+ requests. This works because the DDoS IPs in this dataset exhibit distinctive aggregate behavior (single-endpoint concentration, unusual method mix). However, this approach has limits: truly distributed attacks where each IP sends only 1-2 requests would still evade detection. For those cases, rate limiting and WAF rules at the origin remain the primary defense.

**The handoff timeline for a novel attack:** The anomaly detector flags unusual traffic within hours (once enough requests accumulate to produce statistically significant z-scores). The security team investigates and confirms — producing the first labels within ~24 hours. The next scheduled retrain (or an emergency retrain triggered by the alert) incorporates these labels, and the supervised model covers the new pattern going forward. During the initial hours before the anomaly detector has enough data, the threat intelligence feed and downstream defenses (origin WAF rules, fraud detection) serve as the backstop — the assessment explicitly states that "false negatives are partially mitigated by downstream defenses."

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

**Relevance to the current model:** Unlike the previous model with near-perfect separation, the current model has a real precision-recall tradeoff. At threshold 0.5, precision is 1.0 and recall is 0.4898 — no false positives, but half of attacks are missed. Lowering the threshold to 0.144 (cost-optimal for normal costs) only marginally improves recall to 0.5102 while maintaining perfect precision. Even at threshold 0.01 (high FN cost scenario), recall reaches 0.5551 with precision dropping slightly to 0.9714. The threshold economics confirm that aggressive threshold lowering is justified, but the recall ceiling is bounded by the model's feature limitations, not by the threshold. The missing recall (DDoS, zero-day) requires architectural solutions (threat intel, anomaly detection, WAF rules), not threshold tuning.

**Practical caveat:** The $2.50 FP cost captures only direct lost revenue. Blocking a legitimate customer on a payment flow also creates support tickets, churn risk, and reputational damage — costs that are real but harder to quantify. A production system should set the precision floor higher than the pure economic optimum to account for these indirect costs.

### 3.4 — Adversarial Robustness

With TLS features removed, the model's attack surface has shifted. The primary evasion vectors are now **IP rotation** and **timing jitter**, which target the model's IP-level session features.

**IP rotation:** An attacker changes source IPs frequently, breaking session continuity. Each new IP starts with a clean slate — zero request count, no timing history, no path diversity signal. The model falls back to the 15 per-request features, which carry weaker signal. This is the most effective evasion strategy against the current model. The model's 0% recall on DDoS already demonstrates this limitation: DDoS inherently uses many IPs with few requests each, which is functionally equivalent to IP rotation from the model's perspective.

**Timing jitter:** An attacker adds random noise to request timing, degrading the `inter_request_time_mean` and `inter_request_time_std` features. However, volume-based features (`request_count`, `unique_paths`) and endpoint targeting features (`sensitive_endpoint_ratio`) remain robust — an attacker conducting credential stuffing must still hit authentication endpoints at high volume regardless of timing noise.

**What the model cannot do:** The source-level classifier addresses DDoS where each IP sends 5+ requests with distinctive aggregate behavior (94% recall). However, attacks where each IP sends only 1-2 requests — making aggregate profiling impossible — remain undetectable at the edge. This is a fundamental constraint: no amount of per-IP feature engineering can detect an attack whose signature exists only in the *aggregate across many IPs* each contributing minimal individual signal.

**Mitigation strategy across timescales:**

**Immediate (hours):** Lower the blocking threshold to recover partial recall — the threshold economics from Section 3.3 confirm this is net-positive. Activate the threat intelligence feed as a supplementary signal: it covers ~30% of attacks with high precision and requires no model changes. Enable aggressive rate limiting on sensitive endpoints (`/auth`, `/login`, `/payment`, `/tokenize`) as a WAF-level fallback.

**Short-term (days):** Introduce cross-IP aggregate features at the origin or a centralized aggregation layer — global request rate to specific endpoints, IP diversity per endpoint per time window, geographic distribution of requests. These features can detect distributed attacks but require centralized computation, so they supplement the edge model rather than replace it.

**Long-term (weeks):** Adopt an **ensemble architecture** where the fast per-request edge model (15 features, <0.1ms) runs as a first pass, and a centralized model with cross-IP features runs on flagged traffic. Incorporate **adversarial training** — inject IP-rotated attack samples during training so the model learns to classify without relying on session continuity. Implement **feature rotation**: periodically vary which features the model weights most heavily, increasing the cost for attackers to reverse-engineer and evade the detection logic.

---

## Production Readiness

### CI/CD Pipeline

The project includes two GitHub Actions workflows ([`.github/workflows/`](.github/workflows/)):

**`ci.yml`** runs on every push and pull request:
1. **Lint** — `ruff check` + `ruff format --check` for consistent code style.
2. **Test** — full pytest suite (62 tests) covering label joining, features, model, source model, export, monitoring.
3. **ONNX Validation** — trains both tier models from scratch, exports to ONNX, and validates numerical equivalence. Also asserts minimum metric thresholds (PR-AUC > 0.50, F1 > 0.40 for Tier 1; PR-AUC > 0.50 for Tier 2) to catch regressions.
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
