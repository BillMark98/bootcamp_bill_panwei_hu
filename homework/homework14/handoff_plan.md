# Handoff Plan (Optional)

- Package model as a versioned artifact (registry tag + Git SHA); record feature schema and training data window.
- Environment pinning via requirements/lockfile; provide a minimal Dockerfile (python:3.11-slim).
- Serve a REST `/predict` with JSON schema examples, `/healthz`, and `/metrics` (Prometheus).
- Structured logs: request_id, model_version, feature_hash, latency_ms, status; ship to log store with retention (7d staging, 30d prod).
- Dashboards: Data (schema, nulls, freshness, PSI), Model (AUC/MAE, ECE, score p5/p95), System (latency/error/SLO), Business (approval/conv, revenue/1k, overrides).
- Alert policy: Slack `#ml-alerts`; PagerDuty for Sev-1; each alert links to a runbook.
- Rollout: blue/green with 10% canary and shadow evaluation; auto-rollback on SLO breach.
- Retraining triggers: PSI ≥ 0.2 on ≥2 key features or 2-week AUC < 0.60; change review before promote.
- Weekly ops review (DS, DE, SRE, PM) and monthly post-release report with incidents/KPIs.
