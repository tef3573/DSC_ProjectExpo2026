# Black CEO Equity Strategy: Quantitative Alpha & Factor Attribution

#### An institutional-grade, survivorship-bias-free, point-in-time quantitative backtesting engine modeling excess return and risk attribution of a dynamic equity portfolio composed of Black Fortune 500 & S&P 500 CEOs using risk-adjusted excess returns ($\alpha > 0$) relative to market cap-weighted (`SPY`) and equal-weighted (`RSP`) S&P 500 benchmarks from 2000 through October 2026.
---

## Research Hypotheses
* **Null Hypothesis ($H_0$):** $\alpha \le 0$  
  A portfolio of Black CEO-led equities produces returns explained entirely by systematic factor loadings (Market, Size, Value, Profitability, Investment).
* **Alternative Hypothesis ($H_1$):** $\alpha > 0,\; p < 0.05$  
  A dynamic, quality-tilted portfolio of Black CEO-led companies produces persistent, statistically significant risk-adjusted alpha with superior downside preservation.

---

## Methodological Rigor
1. **Zero Survivorship Bias:** Integrates historical price series for delisted constituents (e.g., J.C. Penney `JCP` during Marvin Ellison's 2015–2018 tenure) rather than restricting the universe to surviving entities.
2. **Zero Look-Ahead Bias:** Implements a point-in-time dynamic boolean eligibility matrix $M_{t,i} \in \{0, 1\}$. Assets enter the investable set on the official start date and exit on the resignation date, with a strict $t+1$ execution lag.
3. **Cash-Drag Defense:** When the active constituent count $N_t < 3$, equity weight is scaled down to $N_t / 3.0$ and the remainder is swept to cash earning the risk-free rate ($R_f$) to prevent idiosyncratic concentration blowups.
4. **Transaction Friction:** Incorporates a 10 bps ($0.0010$) trading friction deducted on daily portfolio turnover.
5. **Decoupled Dual Benchmarking:** Compares performance against cap-weighted S&P 500 (`SPY`) across the entire horizon (2000–2026) and against equal-weighted S&P 500 (`RSP`) post-May 2003 (RSP inception date) to prevent flat-line distortions.

---
## Milestone 1 Status: Verified & Frozen (v0.1.0)

| Requirement | Implementation Details | Status |
| :--- | :--- | :--- |
| **Audited Universe** | 21 unique executive tenures across large/mid-cap equities | Verified |
| **Market Data Ingestion** | Automated Yahoo Finance batch engine (`2000-01-01` to current) | Verified |
| **Corporate Action Defense** | `QVCG` continuity fallback merged into `QRTEA` | Verified |
| **Survivorship-Bias Defense** | Merged delisted archive (`data/delisted/JCP.csv`) | Verified |
| **Trading Calendar** | Reindexed to `SPY` (6,748 sessions, 5-day bounded forward-fill) | Verified |
| **Execution Lag** | Strict $t+1$ shift applied to eliminate look-ahead bias | Verified |

---


## Test Verification
```text
bash
python3 -m unittest discover -s tests -v

## Architecture Overview

.
├── data/
│   ├── black_ceo_roster.csv     # Audited executive tenure roster (21 entries)
│   ├── delisted/
│   │   └── JCP.csv              # Survivorship-free historical NYSE prices (740 sessions)
│   └── prices_daily.csv         # Daily adjusted close matrix (2000–2026)
│   ├── prices_daily.csv         # 6,748 trading sessions x 23 columns (SPY-aligned)
│   └── universe_mask.csv        # Boolean eligibility matrix (t+1 execution lag)
├── docs/
│   └── universe.md              # Mathematical rules and diversification defense
├── src/
│   ├── __init__.py              # Package initializer
│   ├── data_loader.py           # Ingestion engine & delisted merger
│   ├── universe.py              # Point-in-time mask builder
│   └── backtest.py              # Vectorized backtesting & metrics engine
├── tests/
│   └── test_data_schema.py      # Automated schema & date consistency tests
│   └── test_universe.py         # Lag & active constituent count bounds
│   └── test_integration_milestone1.py # End-to-end pipeline integration check
├── requirements.txt             # Pinned environment dependencies
├── web/                        
└── README.md
