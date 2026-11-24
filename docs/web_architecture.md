# 1-BLOOMBERG Quantitative Workstation Architecture

## 1. 6-Screen Viewport-Locked Layout (`100vh x 100vw`)
- **Outer Frame:** Persistent top command bar (`1<GO>`–`6<GO>`), left navigation rail (`Analysis`, `Data Library`, `Monitor`), right monitor rail (`Key Indicators` + `Active Constituents`), and bottom live telemetry status bar.
- **Screens:**
  - `01 Overview`: BI Playbook executive summary, animated KPI strip, and clickable empirical findings feed.
  - `02 Equity & Drawdown`: Interactive log/linear cumulative wealth curve (`$10,000` initial) and underwater drawdown profile.
  - `03 Universe Breadth`: Point-in-time active Black CEO constituent count ($N_t$, `0` to `17` issuers).
  - `04 Macro Regimes`: 6-regime event attribution table and full-horizon institutional scorecard.
  - `05 FF5 & Bootstrap`: Fama-French 5-Factor Newey-West HAC loadings and 10,000-trial Monte Carlo bootstrap comparison.
  - `06 40-CEO Roster`: Searchable and status-filterable dossier across all 40 CEO tenures / 36 unique issuers with tenure Peak Stock Prices and SVG sparklines.

## 2. Keyboard & UI Interaction Controls
- **Function Commands:** Type `1<GO>` through `6<GO>` in the top orange command bar (or press keys `1`–`6` outside input fields) to switch screens.
- **Sidebar Toggle:** Click `94<Go> to Hide Sidebar` in the top command line to collapse or restore the left navigation rail.
- **Portfolio Weighting & Regime Filters:** Top sub-bar dropdowns dynamically recalculate charts and regime attribution between `Equal-Weight (Net)` and `Convex-Quality QP (Net)`.
