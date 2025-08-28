# Stage 14 — Deployment & Monitoring: Reflection

**Risks.** In production the model can fail from: (1) schema drift (columns added/renamed, types change); (2) rising nulls or silent imputations; (3) feature distribution shift due to seasonality, promos, or new cohorts; (4) label delay/label bias that hides true performance; (5) traffic spikes or dependency failures that increase latency/errors.

**Monitoring plan (four layers).**
- **Data:** schema hash check; per-feature null rate; freshness (minutes since last batch); drift via Population Stability Index (PSI) on top features (warn ≥0.1, act ≥0.2).
- **Model:** rolling 14-day AUC/MAE by segment; calibration (ECE); score distribution guardrails (p5/p95). Trigger if AUC < 0.62 for 7 days or MAE +15% vs baseline.
- **System:** p95 latency and error rate from the serving layer; throughput/capacity (CPU, memory, queue); batch job success. Page if p95 > 400 ms for 15 minutes or error rate > 0.5%.
- **Business:** weekly approval/convert rate (±5 pts), revenue per 1k requests (±10%), override/complaint rate (>2%) to detect harmful drift.

**Ownership & handoffs.** Data Engineering owns **Data** monitors; DS/ML owns **Model**; Platform/SRE owns **System** SLOs; Analytics/PM owns **Business** KPIs. Alerts go to Slack `#ml-alerts` and the on-call pager for Sev-1. **First runbook step:** freeze deploys, capture snapshot (model/data versions, traffic, recent releases), and triage which layer triggered. **Retraining cadence:** monthly or when PSI ≥ 0.2 on two key features or rolling AUC < 0.60 for two weeks. Promote via shadow evaluation and a blue/green rollout with a 10% canary before full traffic.
