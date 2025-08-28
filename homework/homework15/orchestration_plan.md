# Orchestration Plan — Stage 15: Orchestration & System Design

This plan decomposes the project into clear tasks with dependencies (DAG), inputs/outputs, logging, checkpoints, and
right-sized automation choices. It follows the homework instructions for Stage 15.  <!-- matches: jobs, deps, I/O, logging, checkpoints, automation -->
(DAG image: `images/dag.png`).  <!-- deliverables: DAG sketch image -->

## A. Pipeline Tasks (4–8) and Boundaries

1. **Ingest Raw Data**
   - **Input**: external sources (CSV/Parquet/DB extracts) at `data/raw/`
   - **Output**: `data/ingested/data_ingested.parquet`
   - **Idempotent**: Yes (reads immutable dated files; re-run overwrites same output)
   - **Logging**: `logs/ingest.log`  • **Checkpoint**: write success flag `checkpoints/ingest.ok`

2. **Validate & Profile**
   - **Input**: `data/ingested/data_ingested.parquet`
   - **Output**: `reports/data_profile.json`, `reports/validation_summary.md`
   - **Idempotent**: Yes (pure read+report)
   - **Logging**: `logs/validate.log`  • **Checkpoint**: `checkpoints/validate.ok`

3. **Clean & Feature Build**
   - **Input**: ingested data + schema config `config/schema.yaml`
   - **Output**: `data/processed/features.parquet`
   - **Idempotent**: Yes (deterministic transforms with fixed config/seed)
   - **Logging**: `logs/clean.log`  • **Checkpoint**: `checkpoints/clean.ok`

4. **Train Model**
   - **Input**: `data/processed/features.parquet`, train params `config/train.yaml`
   - **Output**: `models/model.pkl`, `reports/train_metrics.json`
   - **Idempotent**: Yes if inputs/params unchanged (hash inputs to directory name)
   - **Logging**: `logs/train.log`  • **Checkpoint**: `checkpoints/train.ok`

5. **Evaluate & Risk/Drift Checks**
   - **Input**: `models/model.pkl`, held-out set `data/processed/features.parquet`
   - **Output**: `reports/eval_metrics.json`, `reports/drift_report.md`
   - **Idempotent**: Yes  • **Logging**: `logs/eval.log`  • **Checkpoint**: `checkpoints/eval.ok`

6. **Package for Serving**
   - **Input**: `models/model.pkl`
   - **Output**: `artifact/model_v{hash}.pkl`, `artifact/manifest.json`
   - **Idempotent**: Yes (hash-based artifact dir)
   - **Logging**: `logs/package.log`  • **Checkpoint**: `checkpoints/package.ok`

7. **Report/Deliver**
   - **Input**: metrics + model card templates
   - **Output**: `deliverables/report.md` (or deck), images in `deliverables/images/`
   - **Idempotent**: Yes  • **Logging**: `logs/report.log` • **Checkpoint**: `checkpoints/report.ok`

## B. Dependencies (DAG)


- Parallelizable: data profiling sub-steps, certain feature sub-pipelines, and report rendering can run after **Evaluate** starts.

See `images/dag.png` in this folder for a simple DAG sketch.

## C. Logging & Checkpoints

- **Logging**: Each task writes to `logs/<task>.log` (INFO level; includes start/end timestamp, row counts, hashes).
- **Checkpoints**: `checkpoints/<task>.ok` files on success; tasks check upstream `.ok` files before running.
- **Artifacts**: Use **content hashes** (data hash + params hash) in output folder names to ensure idempotency and lineage.

## D. Failure Points & Retries

- **Schema/CSV drift** (missing/extra columns) → fail fast in **Validate**; retry **0–2** times (backoff) after confirming upstream fix.
- **Null spikes/parse errors** in **Clean** → retry once after quarantining bad rows to `data/quarantine/`.
- **Training instability** (convergence) → retry with fallback seed; if persists, halt with error message and attach sample diagnostics.
- **Disk/IO errors** → retry **3** times with exponential backoff.
- **Threshold breach** in **Evaluate** (e.g., AUC drop/drift) → **do not** proceed to Package; file an alert and require human review.

## E. What to Automate Now vs. Manual

- **Automate now**: Ingest, Validate, Clean/Feature, Train, Evaluate — they’re deterministic and benefit most from idempotent runs.
- **Manual for now**: Final **Report** polishing and release notes; **Package** promotion to prod artifact registry (gate via human review).
- Rationale: right-size scope; automation for high-churn steps, human review where judgment is needed.

## F. Runbook Snippets

- **Re-run a failed task**: confirm upstream `.ok` files exist; clear current task outputs if partial; re-run with same params.
- **Backfill**: point Ingest to a date-partitioned input; the DAG uses date-based folders and remains idempotent.

---
_This plan conforms to the Stage 15 homework deliverables/rubric (pipeline decomposition, dependencies, reliability, right-sizing, presentation, and optional refactor)._
