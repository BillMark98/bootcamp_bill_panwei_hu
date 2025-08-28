# Lifecycle Framework Guide — Final (Stage 16)

This guide maps each lifecycle stage to your project decisions and outcomes.

| Lifecycle Stage | What You Did | Challenges | Solutions / Decisions | Future Improvements |
|-----------------|--------------|------------|-----------------------|---------------------|
| **1. Problem Framing & Scoping** | Clarified objective: predict/quantify target and communicate risk to stakeholders. Defined success as lower RMSE and stable errors across segments. | Ambiguity on constraints and acceptable error. | Wrote crisp problem statement and acceptance criteria; defined target, features, and scope boundaries. | Include explicit guardrails (latency, cost) up front and align on “must-have vs nice-to-have.” |
| **2. Tooling Setup** | Python 3, scikit-learn, matplotlib; notebooks for exploration; simple Make/venv. | Env bumps and version drift. | Pinned requirements and recorded commands in README. | Add Dockerfile/lockfile; CI sanity checks for env reproducibility. |
| **3. Python Fundamentals** | Data wrangling, plotting, functions; modularized helpers. | Occasional performance and readability issues. | Refactored into `src/` utilities, added docstrings and type hints. | Add unit tests, linting, pre-commit hooks. |
| **4. Data Acquisition / Ingestion** | Loaded CSV/synthetic fallback; documented schema. | Missing values and occasional schema changes. | Centralized ingestion; schema checks and basic validations. | Automate fetch/partitioning; add schema registry. |
| **5. Data Storage** | Organized under `/data/raw`, `/data/processed`, `/models`, `/deliverables`. | Versioning and lineage tracking. | Timestamped folders; recorded hashes in reports. | Introduce DVC or lightweight data registry. |
| **6. Data Preprocessing** | Imputation (mean/median), scaling as needed. | Skew/outliers affected imputers. | Compared mean vs median; chose robust defaults per feature. | Add robust scalers and feature-specific strategies. |
| **7. Outlier Analysis** | Residual inspection; boxplots by segment. | Distinguishing noise vs true events. | Flagged outliers but kept when informative; documented rationale. | Add isolation forest / robust z-score to automate flags. |
| **8. EDA** | Trends, distributions, correlations; consistent axes. | Mixed signals on subgroup variance. | Standardized plot templates; subgroup overlays. | Add interactive EDA (facets, tooltips). |
| **9. Feature Engineering** | Basic ratios/lags; polynomial (d=2) trial. | Risk of overfitting, collinearity. | Validated via held-out RMSE and stability; trimmed noisy terms. | Explore domain-driven features and regularized models. |
| **10. Modeling (Reg/TS/Class)** | Linear baseline; compared polynomial; used TimeSeriesSplit where applicable. | Tradeoff between simplicity and flexibility. | Selected simplest model that met acceptance criteria. | Consider regularization and lightweight tree models. |
| **11. Evaluation & Risk Communication** | RMSE with **bootstrap 95% CI**; scenario comparison with error bars; subgroup residuals. | Communicating uncertainty clearly. | Standardized CI visuals and stakeholder summary bullets. | Add parametric-vs-bootstrap comparison, calibration checks. |
| **12. Results Reporting & Stakeholder Communication** | Markdown report + exported PNGs; executive summary; assumptions & risks. | Translating tech to decisions. | One-pagers with headline takeaways and decision implications. | Add a short deck; align visuals to branding. |
| **13. Productization** | Packaged model (`model.pkl`); Flask API (`POST /predict`, `GET /plot`); repo structure with README. | Maintainability and API contract clarity. | Modular `src/`, schema examples, requirements pinned. | Add auth, tests, and containerization; versioned model registry. |
| **14. Deployment & Monitoring** | Defined 4-layer monitoring: **Data/Model/System/Business**; thresholds for PSI, AUC, p95 latency, error rate; runbook. | Threshold setting and alert routing. | Concrete alerts to Slack/on-call; first-step freeze and snapshot. | Add dashboards and drills; automate retrain triggers. |
| **15. Orchestration & System Design** | Decomposed pipeline into tasks (Ingest→Validate→Clean→Train→Evaluate→Package→Report) with checkpoints and retries; DAG sketch. | Dependency handling and idempotency. | `.ok` checkpoints, content-hashed artifacts, backoff retries. | Migrate to an orchestrator (e.g., Airflow/Prefect) with calendars/backfills. |
| **16. Lifecycle Review & Reflection** | Consolidated decisions; aligned repo to lifecycle and rubric for final submission. | Ensuring consistency across documents and code. | Final pass: README, report, figures, framework guide, and checklist. | Add CI/CD for checks, periodic audits, and post-mortems. |

---

### Reflection Prompts (brief answers)
- **Most difficult:** Balancing model simplicity vs flexibility while keeping comms clear.  
- **Most rewarding:** Seeing uncertainty and scenario analysis shape stakeholder decisions.  
- **Stage connections:** Early scoping and schema choices constrained features and evaluation later; monitoring thresholds informed deployment gates.  
- **Do differently next time:** Lock environment earlier; add tests/CI before modeling; decide retrain triggers with business owners upfront.  
- **Skills to strengthen:** Drift detection, calibration, and production-grade observability.
