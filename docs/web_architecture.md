# Interactive Web Dashboard & Visualization Architecture (Milestone 5)

## 1. Overview & Design Principles

The **Black CEO Portfolio Backtest** visualization pipeline decouples quantitative compute (`src/backtest.py`, `src/attribution.py`, `src/bootstrap.py`) from interactive presentation (`src/charts.py`, `notebook/01_interactive_visualizations.ipynb`, and `web/`). Rather than executing live optimizations or 10,000-trial bootstraps on page load, the pipeline pre-computes deterministic Plotly JSON figure specifications stored in `data/charts/`.

## 2. Artifact & Data Flow

```text
data/backtest_results.csv ──┐
data/summary_metrics.json ──┼──► src/charts.py ──► data/charts/*.json ──► Web Dashboard (Plotly.js)
data/factor_loadings.csv ───┤         │
data/mc_results.json ───────┘         └──► notebook/01_interactive_visualizations.ipynb
./venv/bin/python - << 'EOF'
from pathlib import Path

Path("docs").mkdir(parents=True, exist_ok=True)
doc_text = """# Interactive Web Dashboard & Visualization Architecture (Milestone 5)

## 1. Overview & Design Principles

The **Black CEO Portfolio Backtest** visualization pipeline decouples quantitative compute (`src/backtest.py`, `src/attribution.py`, `src/bootstrap.py`) from interactive presentation (`src/charts.py`, `notebook/01_interactive_visualizations.ipynb`, and `web/`). Rather than executing live optimizations or 10,000-trial bootstraps on page load, the pipeline pre-computes deterministic Plotly JSON figure specifications stored in `data/charts/`.

## 2. Artifact & Data Flow

    data/backtest_results.csv ──┐
    data/summary_metrics.json ──┼──► src/charts.py ──► data/charts/*.json ──► Web Dashboard (Plotly.js)
    data/factor_loadings.csv ───┤         │
    data/mc_results.json ───────┘         └──► notebook/01_interactive_visualizations.ipynb

### Exported Chart Artifacts (`data/charts/`)

| Artifact Path | Builder Function (`src/charts.py`) | Sub-Issue | Interactive Controls |
| :--- | :--- | :--- | :--- |
| `data/charts/cumulative_equity_curve.json` | `build_cumulative_equity_curve_figure()` | `#243` | Linear / Log y-axis toggle, `1Y / 5Y / 10Y / MAX` range selector, unified hover |
| `data/charts/underwater_drawdown.json` | `build_underwater_drawdown_figure()` | `#244` | Peak-to-trough percentage area fills, `1Y / 5Y / 10Y / MAX` range selector |
| `data/charts/active_constituent_count.json` | `build_active_constituent_count_figure()` | `#245` | Stepped $N_t$ area curve, $N_t = 3$ full-deployment threshold line |

## 3. Programmatic & Notebook Execution

* **CLI Regenerator:** `./venv/bin/python -m src.charts`
* **Interactive Exploration Notebook:** Open `notebook/01_interactive_visualizations.ipynb` using the `./venv/bin/python` kernel to inspect and render each figure interactively via `fig.show()`.

## 4. Schema & Serialization Contract

Every figure returned by `src/charts.py` is serialized via `export_figure_json(fig, output_path)` using Plotly's `fig.to_json(validate=True)` encoder (`plotly >= 7.1.0`) and validated by `validate_plotly_json_schema()`, guaranteeing strict RFC 8259 UTF-8 JSON (`allow_nan=False`), finite numeric arrays (`data`), and layout configuration (`layout`) consumable directly by frontend `Plotly.newPlot()` calls.
"""
Path("docs/web_architecture.md").write_text(doc_text, encoding="utf-8")
print("Wrote docs/web_architecture.md")
