# Machine Learning Engineer – Edge Security Assessment

## Context

CloudWalk's edge infrastructure processes billions of HTTP requests daily across our CDN/WAF layer. Our security team has identified a growing class of sophisticated L7 attacks that bypass traditional rule-based WAF defenses — including low-and-slow credential stuffing, API abuse, and zero-day exploit attempts that don't match known signatures.

Currently, our WAF relies on static rules and rate-limiting thresholds maintained manually by security engineers. This approach suffers from high false-positive rates on legitimate traffic spikes (e.g., flash sales) and consistently misses attackers who rotate IPs, randomize request timing, and mimic normal user behavior.

We want to build an **ML-based anomaly detection system that runs at the edge**, classifying incoming HTTP request sequences as malicious or benign in real time, enabling automated blocking or throttling before requests ever reach origin servers.

Our red team regularly simulates attack campaigns against our infrastructure. These simulations, combined with production incident data, form the basis of our labeled dataset.

## Main Question

**How would you design, train, and deploy a machine learning model to detect malicious HTTP traffic at the edge?** Explain your full process — from data handling and feature engineering through model selection, deployment strategy, and production monitoring.

Where information is ambiguous or missing, state your assumptions explicitly and justify them.

## Constraints

- **Latency budget:** Inference must complete in under **5ms per request** at the edge node. You cannot call back to a centralized model server for every request.
- **Throughput:** Each edge node handles ~50k requests/second at peak.
- **Labels are noisy and delayed.** Ground-truth labels come from: (a) red team simulations, (b) post-incident forensics by the security team (often days later), and (c) WAF rule triggers (high precision but low recall).
- **Attackers adapt.** Assume significant concept drift — attack patterns change weekly as adversaries evolve their tooling.
- **Class imbalance:** Malicious traffic is <0.1% of total volume.
- **Context matters.** A single request in isolation may look benign; the attack signal often emerges across a *sequence* of requests from the same source (session/IP/fingerprint).
- **Edge environment:** Models must be deployable to a constrained runtime (think Cloudflare Workers / WASM). No GPU available at inference time. Model size should be kept minimal.
- **False positives are expensive.** Blocking a legitimate customer on a payment flow has direct revenue impact. False negatives are also costly but are partially mitigated by downstream defenses.
- **You have access to a transactional-level threat intelligence feed** that flags known-bad IPs, ASNs, and fingerprints — but it covers only ~30% of actual attacks.

## Available Data

### Table: `http_requests` (sample)

| request_id | timestamp | source_ip | method | path | status_code | response_time_ms | body_size_bytes | user_agent | tls_fingerprint | country | asn |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a8f3e1 | 2025-01-15T10:32:01Z | 203.0.113.42 | POST | /api/v1/auth/login | 200 | 12 | 340 | Mozilla/5.0 ... | ja3_abc123 | BR | AS16509 |
| b2c4d7 | 2025-01-15T10:32:01Z | 198.51.100.7 | GET | /api/v1/cards/tokenize | 403 | 3 | 0 | python-requests/2.28 | ja3_def456 | US | AS14061 |

### Table: `request_headers` (sample)

| request_id | header_name | header_value |
|---|---|---|
| a8f3e1 | Content-Type | application/json |
| a8f3e1 | Accept-Language | pt-BR,en;q=0.9 |
| b2c4d7 | Content-Type | application/x-www-form-urlencoded |

### Table: `incident_labels` (sample)

| incident_id | source_identifier | identifier_type | attack_class | confidence | labeled_at | active_from | active_until |
|---|---|---|---|---|---|---|---|
| inc_001 | 203.0.113.0/24 | ip_range | credential_stuffing | high | 2025-01-17 | 2025-01-15T08:00Z | 2025-01-15T14:00Z |
| inc_002 | ja3_xyz789 | tls_fingerprint | scanner | medium | 2025-01-16 | 2025-01-14T00:00Z | 2025-01-16T00:00Z |

### Table: `threat_intel_feed` (sample)

| indicator | indicator_type | threat_type | first_seen | last_seen | source |
|---|---|---|---|---|---|
| 198.51.100.7 | ip | botnet_c2 | 2025-01-10 | 2025-01-15 | feed_abc |
| ja3_def456 | tls_fingerprint | scanner | 2025-01-12 | 2025-01-14 | feed_xyz |

---

## Part 2 — Practical Implementation

You are provided with a synthetic dataset that simulates 7 days of HTTP traffic through our edge infrastructure. The dataset includes both normal traffic and several embedded attack campaigns.

**Files provided:**
- `http_requests.csv` — ~50,000 HTTP request logs
- `request_headers.csv` — Associated request headers
- `incident_labels.csv` — Post-incident labels from the security team (arrives with delay)

### Tasks

**2.1 — Exploratory Analysis & Label Joining**

The `incident_labels` table uses IP ranges, individual IPs, and TLS fingerprints to identify malicious sources — but `http_requests` has individual request-level data. Write code to join these labels back to individual requests, handling the temporal bounds (`active_from` / `active_until`) correctly. Report how many requests in the dataset you can label as malicious vs. benign, and discuss any labeling gaps or ambiguities you encounter.

**2.2 — Feature Engineering**

Design and implement a feature engineering pipeline that extracts meaningful signals from the raw request data. Consider both:
- **Per-request features** (e.g., path characteristics, header anomalies, body size)
- **Session/source-level aggregated features** (e.g., request rate, endpoint diversity, error rate, timing regularity)

Justify your feature choices. Which features do you expect to have the highest predictive power, and why?

**2.3 — Baseline Model**

Train a baseline model to classify requests (or sources) as malicious vs. benign. You are free to choose your approach, but you must:
- Handle the class imbalance appropriately
- Use a proper train/test split that respects temporal ordering (no future data leakage)
- Report precision, recall, F1, and false positive rate
- Discuss why you chose your model and what tradeoffs you made

**2.4 — Edge Deployment Feasibility**

Given that inference must run in <5ms on a WASM runtime with no GPU:
- Estimate your model's inference time and memory footprint
- If your chosen model is too heavy, propose a distillation or simplification strategy
- Describe how you would export and serve this model at the edge

### Deliverables

Submit your work as a self-contained repository or folder with:
- All code (Python preferred, notebooks are fine for exploration but include .py scripts for the pipeline)
- A `README.md` explaining how to run your code
- A short writeup (can be in the README or separate doc) covering your reasoning for each task

---

## Part 3 — Tradeoff Deep-Dives

Answer the following questions concisely. We value clarity of thought over length — aim for 1-2 paragraphs per question unless a diagram or calculation is needed.

**3.1 — Stateless Edge, Stateful Signals**

Session-level features (request rate, endpoint sequence patterns) require state across requests. But edge nodes are stateless — each request may hit a different node, and there is no shared memory between them. How do you maintain session context at the edge? Propose a concrete architecture, discuss its latency implications, and identify what breaks if the state store becomes unavailable.

**3.2 — The Labeling Bottleneck**

Your best labels arrive 1-3 days after an attack. New attack patterns may appear at any time. How do you handle the gap between a novel attack starting and having labeled data for it? Describe a concrete strategy that combines supervised and unsupervised approaches, and explain when each takes over.

**3.3 — Threshold Economics**

Assume the following (simplified):
- Blocking a legitimate payment request costs $2.50 in lost revenue on average
- A successful attack request that reaches origin costs $0.10 on average (amortized across the damage caused)
- Your model processes 100M requests/day, of which 0.05% are actually malicious

At what precision/recall operating point should you set your blocking threshold? Show your math. How does this change if the attack cost jumps to $5.00 per request during an active credential stuffing campaign?

**3.4 — Adversarial Robustness**

An attacker reverse-engineers that your model heavily weighs TLS fingerprint and request timing regularity. They begin randomizing their JA3 fingerprint per-request and adding jitter to their request intervals. Your model's recall drops from 92% to 41%. What is your response plan? Describe both the immediate mitigation and the longer-term model architecture changes you would make.
