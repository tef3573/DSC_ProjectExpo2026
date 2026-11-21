"""
src/charts.py
Interactive Plotly Visualization Builders for Milestone 5: Web Dashboard & Presentation.

Sub-issue #243: Cumulative wealth / equity curve comparison across allocation schemes
and S&P 500 benchmarks with linear/log scale toggles and range selectors.
"""
from pathlib import Path
from typing import Dict, Optional
import json

import pandas as pd
import plotly.graph_objects as go

BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
BACKTEST_RESULTS_PATH: Path = DATA_DIR / "backtest_results.csv"
CHARTS_OUTPUT_DIR: Path = DATA_DIR / "charts"

SERIES_STYLES: Dict[str, Dict[str, object]] = {
    "Black_CEO_Cumulative": {
        "label": "Black CEO Equal-Weight (Net)",
        "color": "#D4AF37",
        "width": 2.6,
        "dash": "solid",
    },
    "Convex_Quality_Cumulative": {
        "label": "Black CEO Convex-Quality (Net)",
        "color": "#1B9E77",
        "width": 2.4,
        "dash": "solid",
    },
    "Inverse_Vol_Cumulative": {
        "label": "Black CEO Inverse-Volatility (Net)",
        "color": "#7570B3",
        "width": 2.0,
        "dash": "dot",
    },
    "SPY_Cumulative": {
        "label": "SPY (S&P 500 Cap-Weighted)",
        "color": "#2B5C8F",
        "width": 1.9,
        "dash": "solid",
    },
    "RSP_Cumulative": {
        "label": "RSP (S&P 500 Equal-Weighted, post-2003)",
        "color": "#D95F02",
        "width": 1.9,
        "dash": "dash",
    },
}


def build_cumulative_equity_curve_figure(
    results_df: Optional[pd.DataFrame] = None,
    results_path: Path = BACKTEST_RESULTS_PATH,
) -> go.Figure:
    """
    Build an interactive Plotly cumulative equity curve comparison figure (#243).

    Args:
        results_df: Optional pre-loaded backtest results DataFrame indexed by date.
        results_path: CSV path fallback when results_df is None.

    Returns:
        Configured plotly.graph_objects.Figure with linear/log toggle and range selector.
    """
    df = (
        results_df.copy()
        if results_df is not None
        else pd.read_csv(results_path, index_col=0, parse_dates=True)
    )
    df.index = pd.to_datetime(df.index)

    fig = go.Figure()
    for col_name, style in SERIES_STYLES.items():
        if col_name not in df.columns:
            continue
        series = df[col_name].dropna()
        if series.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=str(style["label"]),
                line={
                    "color": str(style["color"]),
                    "width": float(style["width"]),
                    "dash": str(style["dash"]),
                },
                hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: $%{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title={
            "text": "Cumulative Wealth Growth ($1.00 Initial Investment, Net of 10 bps Friction)",
            "x": 0.02,
            "xanchor": "left",
        },
        template="plotly_white",
        hovermode="x unified",
        height=540,
        margin={"l": 60, "r": 30, "t": 90, "b": 50},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
        },
        xaxis={
            "title": "Trading Date",
            "rangeselector": {
                "buttons": [
                    {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                    {"count": 5, "label": "5Y", "step": "year", "stepmode": "backward"},
                    {"count": 10, "label": "10Y", "step": "year", "stepmode": "backward"},
                    {"step": "all", "label": "MAX"},
                ]
            },
            "rangeslider": {"visible": False},
        },
        yaxis={
            "title": "Cumulative Wealth ($)",
            "tickprefix": "$",
            "type": "linear",
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 1.0,
                "xanchor": "right",
                "y": 1.13,
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Linear Scale",
                        "method": "relayout",
                        "args": [{"yaxis.type": "linear"}],
                    },
                    {
                        "label": "Log Scale",
                        "method": "relayout",
                        "args": [{"yaxis.type": "log"}],
                    },
                ],
            }
        ],
    )
    return fig


def export_figure_json(fig: go.Figure, output_path: Path) -> Path:
    """Serialize a Plotly Figure cleanly to deterministic JSON on disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(fig.to_json())
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    equity_fig = build_cumulative_equity_curve_figure()
    out_file = export_figure_json(equity_fig, CHARTS_OUTPUT_DIR / "cumulative_equity_curve.json")
    print(f"Exported cumulative equity curve figure JSON to {out_file}")
