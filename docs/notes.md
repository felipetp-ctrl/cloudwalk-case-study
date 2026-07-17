# What I need to do:
ML classification model for anomaly detection of "incoming HTTP request sequences as malicious or benign in real time, enabling automated blocking or throttling before requests ever reach origin servers." **(at the edge)**. Do the feature engineering needed, deployment strategy (considering other models they already have) and production monitoring.

To make it, I have some constraints that they showed on the original .md file. 
> ## Constraints
> - **Latency budget:** Inference must complete in under **5ms per request** at the edge node. You cannot call back to a centralized model server for every request.
>> Here, I think we can use the model on Rust, maybe using ONNX too. 
> - **Throughput:** Each edge node handles ~50k requests/second at peak.
>> This is important to not overwhelm the requests to each node, running out of memory. How to get 50k requests persecond?
> - **Labels are noisy and delayed.** Ground-truth labels come from: (a) red team simulations, (b) post-incident forensics by the security team (often days later), and (c) WAF rule triggers (high precision but low recall).
>> Unsupervised model? or self-supervised? too heavy? idk
> - **Attackers adapt.** Assume significant concept drift — attack patterns change weekly as adversaries evolve their tooling.
>> CI/CD, alerts of retraining. Online learning?
> - **Class imbalance:** Malicious traffic is <0.1% of total volume.
>> Read better the paper, infer about it later (maybe SMOTE + adjust threshold and metrics(?))
> - **Context matters.** A single request in isolation may look benign; the attack signal often emerges across a *sequence* of requests from the same source (session/IP/fingerprint).
>> Windows of requests (30s/5minutes?)
> - **Edge environment:** Models must be deployable to a constrained runtime (think Cloudflare Workers / WASM). No GPU available at inference time. Model size should be kept minimal.
>> Light models like LightGBM, study more
> - **False positives are expensive.** Blocking a legitimate customer on a payment flow has direct revenue impact. False negatives are also costly but are partially mitigated by downstream defenses.
>> As said earlier, adjust threshold and metrics
> - **You have access to a transactional-level threat intelligence feed** that flags known-bad IPs, ASNs, and fingerprints — but it covers only ~30% of actual attacks.
>> Look more the data, what to do with the rest of ~70%?

# Anotações
Sobre class imbalance (substituindo SMOTE):
Pesquise como o parâmetro scale_pos_weight do XGBoost e class_weight='balanced' do LightGBM funcionam internamente. A ideia central é: em vez de fabricar dados sintéticos, o modelo penaliza mais os erros na classe minoritária durante o treinamento. Entenda por que isso é preferível a SMOTE quando as amostras positivas são heterogêneas (ataques de tipos diferentes não devem ser interpolados entre si).

Pesquise também o conceito de focal loss — ele reduz a contribuição das amostras fáceis (tráfego benigno óbvio) e força o modelo a focar nos casos difíceis. Entenda em que situação focal loss seria melhor que class_weight, e em qual não faria diferença significativa.

O que colocar no documento
Adicione uma seção de "Decisões sobre class imbalance" com algo assim:

SMOTE descartado: com apenas ~592 amostras maliciosas de tipos heterogêneos (credential stuffing, scanner), a interpolação entre exemplos de ataques diferentes geraria amostras sintéticas que não representam nenhum padrão real de ataque
Estratégia escolhida: scale_pos_weight / class_weight com razão proporcional ao desbalanceamento (~83:1), porque penaliza erros na classe minoritária sem fabricar dados
Alternativa considerada: focal loss — útil se o modelo tiver muitas predições "fáceis" com alta confiança no tráfego benigno, forçando foco nos casos de fronteira
Métrica primária: não usar accuracy (inútil com 98.8% de negativos). Usar precision-recall AUC e F1, com atenção especial ao false positive rate dado o custo de $2.50 por bloqueio incorreto

# Tasks
- [x] 2.1 Exploratory Analysis & Label Joining
    - [x] Label join code (IP exato, CIDR range, TLS fingerprint — todos com filtro temporal)
    - [x] Resultado: 592 malicious (1.18%) vs 49,408 benign (98.82%)
    - [x] Discussão de gaps/ambiguidades (label absence ≠ benign, CIDR/TLS false positives, labeling delay, multi-incident overlap, dataset imbalance diverge do constraint)

- [ ] 2.2 Feature engineering
    - [ ] Per-request features
    - [ ] Session/source-level aggregated features
    - [ ] Use confidence levels as weights?
    - [ ] Aggregate everything on a pipeline
    - [ ] Justify feature choices, predictive power, etc

- [ ] 2.3 Baseline Model
    - [ ] Handle the class imbalance appropriately (SMOTE?) -> (No, if we use LightGBM or XGBoost, focus on class_weight and scale_pos_weight)
    - [ ] Use a proper train/test split that respects temporal ordering (no future data leakage)
    - [ ] Report precision, recall, F1, and false positive rate
    - [ ] Discuss why you chose your model and what tradeoffs you made

- [ ] 2.4 Edge Deployment Feasibility
    - [ ] Estimate your model's inference time and memory footprint (timeit - python)
    - [ ] If your chosen model is too heavy (don't believe so), propose a distillation or simplification strategy
    - [ ] Describe how you would export and serve this model at the edge (search more about how edge infrastructure works)