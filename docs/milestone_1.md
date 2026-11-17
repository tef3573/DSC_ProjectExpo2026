# Milestone 1: Point-in-Time Data Pipeline & Survivorship-Bias-Free Universe

## 1. Objective & Quantitative Mandate
Milestone 1 establishes the foundational data ingestion and point-in-time universe construction pipeline for the Black CEO Alpha Backtest (`2000-01-03` through `2026-11-16`, 6,759 NYSE trading sessions). To eliminate survivorship bias and look-ahead bias, the pipeline integrates delisted equity histories, adjusts for corporate restructurings, and enforces a strict $+1$ trading-day execution lag on all executive tenure transitions.

---

## 2. Core Modules & Data Artifacts

| Artifact / Module | Role in Pipeline | Key Dimensions & Schema |
| :--- | :--- | :--- |
| `data/black_ceo_roster.csv` | Curated executive tenure metadata | 32 unique tickers with verified CEO start/end timestamps |
| `data/delisted/JCP.csv` | Local historical archive for delisted J.C. Penney | Adjusted daily close series covering Marvin Ellison's tenure (`2015-08-01` to `2018-06-01`) |
| `src/data_loader.py` | Multi-source equity price ingestion & calendar alignment | Fetches 35 symbols (`32` constituents + `SPY`, `RSP`, `QVCG`), exports `data/prices_daily.csv` (`6759 x 34`) |
| `src/universe.py` | Point-in-time binary constituent mask generator | Enforces $t + 1$ execution lag, exports `data/universe_mask.csv` (`6759 x 32`) |
| `tests/test_data_schema.py` | Schema & date integrity verification | Validates roster date ordering (`start_date < end_date`) and delisted CSV positivity |
| `tests/test_universe.py` | Tenure boundary & calendar parity unit tests | Confirms $0 \le N_t \le 32$, SPY index alignment, and delisted JCP tenure accuracy |
| `tests/test_integration_milestone1.py` | End-to-end Milestone 1 continuity test | Verifies dimensional and index continuity across roster, prices, and universe mask |

---

## 3. Data Engineering & Bias Mitigation Controls

### 3.1 Survivorship Bias Prevention
* **Delisted Constituent Integration (`JCP`):** Because J.C. Penney filed for bankruptcy and delisted from the NYSE in 2020, live market feeds omit its historical series. `src/data_loader.py` automatically falls back to `data/delisted/JCP.csv` so Marvin Ellison's tenure (`2015-08-01` to `2018-06-01`) is captured without survivorship distortion.
* **Corporate Action Continuity (`QVCG` $\rightarrow$ `QRTEA`):** Corporate ticker reorganizations and tracking-stock mergers (`QVCG` into `QRTEA`) are stitched seamlessly within `src/data_loader.py` prior to calendar alignment.
* **Benchmark Calendar Synchronization:** All price series are inner-aligned onto the `SPY` exchange trading calendar (`6,759` sessions) to guarantee zero phantom weekend/holiday rows and uniform cross-sectional indexing.

### 3.2 Look-Ahead Bias Prevention ($t + 1$ Execution Lag)
* Let $I_{i, t}^{\text{raw}} \in \{0, 1\}$ denote whether ticker $i$ is led by a Black CEO on calendar session $t$ according to `data/black_ceo_roster.csv`.
* To ensure portfolio weights at session $t$ never condition on same-day corporate announcements, `src/universe.py` applies a strict $+1$ trading-day shift:

$$
I_{i, t} = I_{i, t - 1}^{\text{raw}}, \qquad I_{i, 0} = 0
$$

* Across the `6,759`-session backtest horizon, the active cross-sectional universe size $N_t = \sum_{i=1}^{32} I_{i, t}$ ranges from $\min(N_t) = 0$ to $\max(N_t) = 17$ (mean $N_t = 8.00$).

---

## 4. Verification Summary
* **Data Schema & Tenure Checks:** Confirmed all 32 roster entries have valid `start_date < end_date` intervals and positive adjusted closing prices.
* **Integration Continuity (`tests/test_integration_milestone1.py`):** Verified exact `DatetimeIndex` equality (`6,759` sessions) and column parity across `data/prices_daily.csv` and `data/universe_mask.csv`.
