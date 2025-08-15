
# Does Turtle Trading Still Work in Today’s Markets?
**Stage:** Problem Framing & Scoping (Stage 01)

---

## Problem Statement
We aim to determine whether the classic **Turtle Trading** (breakout trend-following) rules generate **robust, net-of-cost** excess returns in current markets. The core question: *Does a Donchian-channel breakout system with risk-based position sizing still deliver attractive risk-adjusted performance after realistic costs and modern competition, or has the edge been arbitraged away?*

The study will replicate and modernize the Turtle approach across a diversified, liquid multi-asset universe (e.g., major futures/FX or ETF proxies), evaluate performance from the early 2000s through 2025, and stress-test stability across market regimes (pre/post-2008, QE/ZIRP, COVID, 2022–2023 inflationary regime, 2024–2025). Outcomes will inform a **go / modify / no-go** decision for deploying a trend-following sleeve.

---

## Stakeholder & User
- **Primary stakeholder / decision-maker:** Portfolio Manager (PM) or personal trading lead deciding whether to allocate capital to a rules-based trend-following sleeve.
- **End users:** 
  - **Quant researcher** (builds data pipeline/backtests, reports results).
  - **Risk manager** (assesses drawdowns, correlation to existing book, tail risk).
- **Workflow context & timing:** Initial research sprint (1–2 weeks of prototyping), followed by validation and a small paper-trade phase before any live deployment.

---

## Useful Answer & Decision
- **Type of answer:** Primarily **descriptive** (historical performance) with **predictive lean** via walk-forward and regime analysis; no causal identification required.
- **Decision framing:** 
  - **GO** if the strategy shows **out-of-sample** and **recent-period** net Sharpe above threshold, controlled drawdowns, and low correlation to existing exposures.  
  - **MODIFY** if edge exists but is parameter-sensitive, capacity-limited, or overly cyclical.  
  - **NO-GO** if results are not robust net of costs or degrade materially in recent years.
- **Key metrics (net of costs):**
  - Annualized return/volatility, **Sharpe**, **Sortino**, **Calmar**, **max drawdown**, **hit rate**, **profit factor**.
  - **Turnover**, average holding period, slippage sensitivity, **capacity proxy** (ADV %, notional scaling).
  - **Rolling** (36–60m) Sharpe, **t-stat** of excess returns, **regime** breakdown, correlation to equities, rates, commodities, FX carry/value/momo.
  - **Stability**: parameter sweep heatmaps, walk-forward curves, SPA/White’s Reality Check style sanity tests (bootstrap resampling).

---

## Assumptions & Constraints
- **Data availability:** Daily (or higher) prices for a diversified, liquid set of futures (or ETF proxies if futures not available). Continuous futures with transparent **roll methodology** (back-adjusted or panama-adjusted). FX spot/rolling futures where applicable.
- **Trading frictions:** Commission + fees + **slippage**; model as (a) % of price for ETFs or (b) **ticks** for futures. Include spreads and conservative impact for fast markets.
- **Execution model:** End-of-day signals; next-day open or close execution; no leverage beyond policy (e.g., risk parity scaling within margin limits).
- **Risk constraints:** Unit risk via **ATR-based** position sizing; per-instrument and portfolio-level risk caps; diversification across asset classes.
- **Compliance:** Research-only; no live trading implied; adhere to data licenses.

---

## Known Unknowns / Risks
- **Regime non-stationarity:** Trend premia may wax/wane; 2010s vs 2020s behavior can differ.
- **Overfitting & multiple testing:** Parameter sweeps inflate false positives; control with **walk-forward**, **nested CV**, and **reality checks**.
- **Cost mis-estimation:** Understated slippage/spreads can turn gross edge into net loss.
- **Roll & contract selection:** Different continuous futures methods materially change PnL.
- **Capacity & crowding:** Diminishing returns at scale; correlation spikes in stress.
- **Proxy risk:** ETF substitutes may introduce tracking and borrow constraints.

---

## Success Criteria (Net of Costs)
- **Primary:** Rolling 3–5y **Sharpe ≥ 0.6** recently (2020–2025), **max DD ≤ 25%**, and **low/moderate correlation** to equities (< 0.3).
- **Robustness:** Consistent performance across parameter grids (e.g., 20/55 and neighbors), instruments, and regimes; limited degradation under higher slippage.
- **Stability:** Out-of-sample (holdout / walk-forward) maintains **SR ≥ 0.4–0.5** with similar drawdown/risk profile.
- **Actionability:** Clear risk budget, capital deployment sizing, and monitoring plan.

---

## Approach & Methods (High-Level)
- **Rules to test:**
  - **Entries:** N-day **Donchian channel** breakouts (e.g., 20/55), optional filter (long-term trend or volatility regime).
  - **Exits:** Opposite breakout (N/2), or trailing stop; optional time-based stop.
  - **Sizing:** **ATR(20)** or similar risk targeting to equalize unit risk across assets.
  - **Portfolio:** Equal risk contribution with per-asset and per-sector caps.
- **Validation workflow:**
  1. Clean & stitch data; define continuous contracts/ETF proxies and calendars.
  2. Implement baseline Turtle rules; verify against known references on a small subset.
  3. **Parameter sweeps** with proper train/validation splits; **walk-forward** evaluation.
  4. **Cost sensitivity** analysis; stress tests (gaps, volatility clusters, thin liquidity days).
  5. **Attribution:** by asset, asset-class, regime; correlation to major risk premia.
  6. **Robustness checks:** bootstrapping, White’s SPA/Reality Check style tests.

---

## Lifecycle Mapping (Goal → Stage → Deliverable)
- Confirm viability of Turtle trend-following today → **Stage 01: Framing** → This README + plan.
- Acquire & validate data; define roll rules → **Stage 02: Data & EDA** → Data dictionary, QC report.
- Implement baseline & sweeps → **Stage 03: Modeling/Backtest** → Backtest notebook, results CSVs.
- Validate & stress test; costs/capacity → **Stage 04: Validation** → Robustness report, plots.
- Summarize & recommend → **Stage 05: Reporting** → Slide/memo with **GO/MODIFY/NO-GO**.

---

## Repo Plan

## Repo Plan
/data/, /src/, /notebooks/, /docs/ ; cadence for updates
