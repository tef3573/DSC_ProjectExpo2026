"""
src/charts.py
Interactive Plotly Visualization Builders for Milestone 5: Web Dashboard & Presentation.

Sub-issue #243: Cumulative wealth / equity curve comparison across allocation schemes
and S&P 500 benchmarks with linear/log scale toggles and range selectors.
Sub-issue #244: Underwater peak-to-trough maximum drawdown (MDD) visualization.
Sub-issue #245: Active Black CEO constituent breadth (N_t) stepped area visualization.
"""
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import plotly.graph_objects as go

from src.timeseries import fit_garch_11, run_walk_forward_garch_overlay

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

DRAWDOWN_STYLES: Dict[str, Dict[str, object]] = {
    "Strategy_Drawdown": {
        "label": "Black CEO Equal-Weight (Net)",
        "color": "#D4AF37",
        "fill": "rgba(212, 175, 55, 0.16)",
        "width": 2.2,
    },
    "Convex_Quality_Drawdown": {
        "label": "Black CEO Convex-Quality (Net)",
        "color": "#1B9E77",
        "fill": "rgba(27, 158, 119, 0.14)",
        "width": 2.0,
    },
    "SPY_Drawdown": {
        "label": "SPY (S&P 500 Cap-Weighted)",
        "color": "#2B5C8F",
        "fill": None,
        "width": 1.8,
    },
    "RSP_Drawdown": {
        "label": "RSP (S&P 500 Equal-Weighted, post-2003)",
        "color": "#D95F02",
        "fill": None,
        "width": 1.8,
    },
}


def _load_backtest_df(
    results_df: Optional[pd.DataFrame],
    results_path: Path,
) -> pd.DataFrame:
    """Load or copy the backtest time series DataFrame with a normalized DatetimeIndex."""
    df = (
        results_df.copy()
        if results_df is not None
        else pd.read_csv(results_path, index_col=0, parse_dates=True)
    )
    df.index = pd.to_datetime(df.index)
    return df


def _build_date_rangeselector() -> Dict[str, object]:
    """Return standardized 1Y / 5Y / 10Y / MAX Plotly x-axis range selector buttons."""
    return {
        "y": 1.02,
        "yanchor": "bottom",
        "x": 0.0,
        "xanchor": "left",
        "buttons": [
            {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
            {"count": 5, "label": "5Y", "step": "year", "stepmode": "backward"},
            {"count": 10, "label": "10Y", "step": "year", "stepmode": "backward"},
            {"step": "all", "label": "MAX"},
        ],
    }


def _assert_finite_numbers(node: Any, json_path: str = "$") -> None:
    """Recursively verify that no NaN or Inf floats exist in a JSON-compatible tree."""
    if isinstance(node, float):
        if not math.isfinite(node):
            raise ValueError(f"Non-finite float ({node}) detected at {json_path}")
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            _assert_finite_numbers(item, f"{json_path}[{idx}]")
    elif isinstance(node, dict):
        for key, val in node.items():
            _assert_finite_numbers(val, f"{json_path}.{key}")


def validate_plotly_json_schema(payload: Dict[str, Any]) -> None:
    """Validate that a parsed Plotly figure dictionary satisfies the frontend contract."""
    if not isinstance(payload, dict):
        raise TypeError("Plotly figure payload must be a JSON object (dict).")
    if "data" not in payload or not isinstance(payload["data"], list) or len(payload["data"]) == 0:
        raise ValueError("Plotly figure JSON must contain a non-empty 'data' trace list.")
    if "layout" not in payload or not isinstance(payload["layout"], dict):
        raise ValueError("Plotly figure JSON must contain a 'layout' configuration object.")
    _assert_finite_numbers(payload)


def build_cumulative_equity_curve_figure(
    results_df: Optional[pd.DataFrame] = None,
    results_path: Path = BACKTEST_RESULTS_PATH,
) -> go.Figure:
    """Build an interactive Plotly cumulative equity curve comparison figure (#243)."""
    df = _load_backtest_df(results_df, results_path)

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
            "y": 0.96,
        },
        template="plotly_white",
        hovermode="x unified",
        height=560,
        margin={"l": 60, "r": 30, "t": 115, "b": 50},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.10,
            "xanchor": "left",
            "x": 0.0,
        },
        xaxis={
            "title": "Trading Date",
            "rangeselector": _build_date_rangeselector(),
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
                "y": 1.08,
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


def build_underwater_drawdown_figure(
    results_df: Optional[pd.DataFrame] = None,
    results_path: Path = BACKTEST_RESULTS_PATH,
) -> go.Figure:
    """Build an interactive Plotly underwater peak-to-trough drawdown chart (#244)."""
    df = _load_backtest_df(results_df, results_path)

    fig = go.Figure()
    for col_name, style in DRAWDOWN_STYLES.items():
        if col_name not in df.columns:
            continue
        series = df[col_name].dropna()
        if series.empty:
            continue
        dd_pct = -series.abs() * 100.0
        fill_color = style["fill"]
        fig.add_trace(
            go.Scatter(
                x=dd_pct.index,
                y=dd_pct.values,
                mode="lines",
                name=str(style["label"]),
                line={"color": str(style["color"]), "width": float(style["width"])},
                fill="tozeroy" if fill_color else "none",
                fillcolor=str(fill_color) if fill_color else None,
                hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:.2f}%<extra></extra>",
            )
        )

    fig.add_hline(y=0.0, line_width=1, line_color="#666666")
    fig.update_layout(
        title={
            "text": "Underwater Peak-to-Trough Drawdown Profile (2000–2026)",
            "x": 0.02,
            "xanchor": "left",
            "y": 0.96,
        },
        template="plotly_white",
        hovermode="x unified",
        height=480,
        margin={"l": 60, "r": 30, "t": 110, "b": 50},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.10,
            "xanchor": "left",
            "x": 0.0,
        },
        xaxis={
            "title": "Trading Date",
            "rangeselector": _build_date_rangeselector(),
            "rangeslider": {"visible": False},
        },
        yaxis={
            "title": "Drawdown from Running Peak (%)",
            "ticksuffix": "%",
            "range": [-85, 2],
        },
    )
    return fig


def build_active_constituent_count_figure(
    results_df: Optional[pd.DataFrame] = None,
    results_path: Path = BACKTEST_RESULTS_PATH,
) -> go.Figure:
    """Build an interactive Plotly stepped area chart of active Black CEO count N_t (#245)."""
    df = _load_backtest_df(results_df, results_path)
    if "Active_Constituents_Nt" not in df.columns:
        raise KeyError("Missing required column 'Active_Constituents_Nt' in backtest DataFrame.")

    counts = df["Active_Constituents_Nt"].dropna()
    max_count = int(counts.max()) if not counts.empty else 17

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=counts.index,
            y=counts.values,
            mode="lines",
            name="Active Black CEO Constituents (N_t)",
            line={"color": "#2B5C8F", "width": 2.2, "shape": "hv"},
            fill="tozeroy",
            fillcolor="rgba(43, 92, 143, 0.20)",
            hovertemplate="%{x|%Y-%m-%d}<br>Active Constituents (N_t): %{y:d}<extra></extra>",
        )
    )

    fig.add_hline(
        y=3,
        line_width=1.5,
        line_dash="dash",
        line_color="#D95F02",
        annotation_text="Full-Deployment Threshold (N_t ≥ 3; 1/3 cap below)",
        annotation_position="top left",
    )

    fig.update_layout(
        title={
            "text": "Point-in-Time Active Black CEO Portfolio Breadth (N_t, t+1 Lagged)",
            "x": 0.02,
            "xanchor": "left",
            "y": 0.96,
        },
        template="plotly_white",
        hovermode="x unified",
        height=440,
        margin={"l": 60, "r": 30, "t": 100, "b": 50},
        showlegend=False,
        xaxis={
            "title": "Trading Date",
            "rangeselector": _build_date_rangeselector(),
            "rangeslider": {"visible": False},
        },
        yaxis={
            "title": "Active Constituents (N_t)",
            "dtick": 2,
            "range": [0, max_count + 2],
        },
    )
    return fig


def build_conditional_volatility_figure(
    results_path: Path = BACKTEST_RESULTS_PATH,
    target_ann_vol: float = 0.20,
) -> go.Figure:
    """
    Build interactive GARCH(1,1) annualized conditional volatility comparison across
    Black_CEO_Net, Convex_Quality_Net, SPY_Daily, and the causal 1-step-ahead OOS forecast.
    """
    df = pd.read_csv(results_path, index_col=0, parse_dates=True)
    fig = go.Figure()

    garch_specs = [
        ("Black_CEO_Net", "Black CEO Equal-Weight GARCH(1,1)", "#D4AF37", 2.2, "solid"),
        ("Convex_Quality_Net", "Black CEO Convex-Quality GARCH(1,1)", "#1B9E77", 2.0, "solid"),
        ("SPY_Daily", "SPY Benchmark GARCH(1,1)", "#2B5C8F", 1.8, "solid"),
    ]
    for col, label, color, width, dash in garch_specs:
        if col not in df.columns:
            continue
        fit_res = fit_garch_11(df[col].dropna())
        vol_pct = fit_res.conditional_vol_annualized * 100.0
        fig.add_trace(
            go.Scatter(
                x=vol_pct.index,
                y=vol_pct.values,
                mode="lines",
                name=label,
                line={"color": color, "width": width, "dash": dash},
                hovertemplate="%{x|%Y-%m-%d}<br>" + label + ": %{y:.2f}%<extra></extra>",
            )
        )

    if "Black_CEO_Net" in df.columns:
        wf_res = run_walk_forward_garch_overlay(
            df["Black_CEO_Net"].dropna(), target_ann_vol=target_ann_vol
        )
        oos_vol_pct = wf_res.forecast_vol_ann * 100.0
        fig.add_trace(
            go.Scatter(
                x=oos_vol_pct.index,
                y=oos_vol_pct.values,
                mode="lines",
                name="Causal 1-Step-Ahead OOS Forecast",
                line={"color": "#A855F7", "width": 1.6, "dash": "dot"},
                hovertemplate="%{x|%Y-%m-%d}<br>OOS 1-Step Forecast: %{y:.2f}%<extra></extra>",
            )
        )

    fig.add_hline(
        y=target_ann_vol * 100.0,
        line_width=1.4,
        line_dash="dash",
        line_color="#D95F02",
        annotation_text=f"Vol-Target Ceiling ({target_ann_vol:.0%} Ann. Vol)",
        annotation_position="top left",
    )

    fig.update_layout(
        title={
            "text": (
                "GARCH(1,1) Annualized Conditional Volatility & Causal OOS Forecast (2000–2026)"
            ),
            "x": 0.02,
            "xanchor": "left",
            "y": 0.96,
        },
        template="plotly_white",
        hovermode="x unified",
        height=460,
        margin={"l": 60, "r": 30, "t": 100, "b": 50},
        xaxis={
            "title": "Trading Date",
            "rangeselector": _build_date_rangeselector(),
            "rangeslider": {"visible": False},
        },
        yaxis={
            "title": "Annualized Conditional Volatility (%)",
            "rangemode": "tozero",
        },
    )
    return fig


def export_figure_json(fig: go.Figure, output_path: Path) -> Path:
    """Serialize a Plotly Figure to strict UTF-8 JSON and validate its schema contract."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_json: str = fig.to_json(validate=True)
    payload: Dict[str, Any] = json.loads(raw_json)
    validate_plotly_json_schema(payload)
    strict_json: str = json.dumps(payload, indent=2, allow_nan=False)
    output_path.write_text(strict_json + "\n", encoding="utf-8")
    return output_path


if __name__ == "__main__":
    equity_fig = build_cumulative_equity_curve_figure()
    equity_out = export_figure_json(equity_fig, CHARTS_OUTPUT_DIR / "cumulative_equity_curve.json")
    print(f"Exported cumulative equity curve figure JSON to {equity_out}")

    dd_fig = build_underwater_drawdown_figure()
    dd_out = export_figure_json(dd_fig, CHARTS_OUTPUT_DIR / "underwater_drawdown.json")
    print(f"Exported underwater drawdown figure JSON to {dd_out}")

    breadth_fig = build_active_constituent_count_figure()
    breadth_out = export_figure_json(
        breadth_fig, CHARTS_OUTPUT_DIR / "active_constituent_count.json"
    )
    print(f"Exported active constituent count figure JSON to {breadth_out}")

    cond_vol_fig = build_conditional_volatility_figure()
    cond_vol_out = export_figure_json(
        cond_vol_fig, CHARTS_OUTPUT_DIR / "conditional_volatility.json"
    )
    print(f"Exported conditional volatility figure JSON to {cond_vol_out}")
