# Interactive Web Dashboard & Visualization Architecture (Milestone 5)

## 1. Overview & Design Principles

The **Black CEO Portfolio Backtest** visualization pipeline decouples quantitative compute (`src/backtest.py`, `src/attribution.py`, `src/bootstrap.py`) from interactive presentation (`src/charts.py`, `notebook/01_interactive_visualizations.ipynb`, and `web/`). Rather than executing live optimizations or 10,000-trial bootstraps on page load, the pipeline pre-computes deterministic Plotly JSON figure specifications stored in `data/charts/`.

## 2. Artifact & Data Flow

```text
data/backtest_results.csv ──┐
data/summary_metrics.json ──┼──► src/charts.py ──► data/charts/*.json ──► Web Dashboard (Plotly.js)
data/factor_loadings.csv ───┤         │
data/mc_results.json ───────┘         └──► notebook/01_interactive_visualizations.ipynb
```
