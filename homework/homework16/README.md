# Final Project (Clean Repo) — Lifecycle Mapping

## Structure

## Stakeholder Summary
See `deliverables/final_report.md` for executive summary, charts with interpretation, assumptions & risks, sensitivity, and decision implications.

## Lifecycle Mapping
This repo follows the 16-stage lifecycle and is summarized in `stage16/framework_guide_filled.md`. Key points:
- **Evaluation & Uncertainty:** RMSE + bootstrap 95% CI; scenario comparisons with CI error bars.
- **Productization:** Flask API with schema examples; pinned requirements.
- **Deployment & Monitoring:** Data/Model/System/Business metrics (PSI, AUC, p95 latency, error).
- **Orchestration:** Deterministic tasks with checkpoints and retries; DAG documented.

## How to Reproduce
1) Create env and `pip install -r requirements.txt`  
2) Run notebooks in `/notebooks/` top-to-bottom  
3) Generate deliverables via the provided scripts (e.g., Stage 12/15/16 makers)
