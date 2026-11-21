"""
src/macro_regimes.py
Macro Event Regime Attribution (2000–2026) for Black CEO Portfolio Backtest.

Evaluates portfolio performance, peak-to-trough drawdown & recovery duration,
downside capture vs. SPY, and active Black CEO breadth (N_min to N_max) across
six distinct macroeconomic and rate regimes.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR: int = 252


@dataclass(frozen=True)
class MacroRegimeSpec:
    """Definition of a historical macro/rate event window."""

    regime_id: str
    label: str
    start_date: str
    end_date: str


MACRO_REGIMES: Tuple[MacroRegimeSpec, ...] = (
    MacroRegimeSpec(
        regime_id="dot_com_bust",
        label="Dot-Com Bust & 2001 Recession",
        start_date="2000-03-10",
        end_date="2002-10-09",
    ),
    MacroRegimeSpec(
        regime_id="gfc_2008",
        label="Global Financial Crisis",
        start_date="2007-10-09",
        end_date="2009-03-09",
    ),
    MacroRegimeSpec(
        regime_id="zirp_qe_era",
        label="ZIRP & Quantitative Easing Era",
        start_date="2009-03-10",
        end_date="2020-02-19",
    ),
    MacroRegimeSpec(
        regime_id="covid_shock_stimulus",
        label="COVID-19 Shock & Stimulus Surge",
        start_date="2020-02-20",
        end_date="2021-12-31",
    ),
    MacroRegimeSpec(
        regime_id="inflation_rate_hikes",
        label="2022–2023 Inflation & Rate-Hike Cycle",
        start_date="2022-01-03",
        end_date="2023-10-27",
    ),
    MacroRegimeSpec(
        regime_id="ai_high_rate_regime",
        label="AI & High-Rate Regime",
        start_date="2023-10-30",
        end_date="2026-11-20",
    ),
)


def _compute_drawdown_and_recovery(cum_curve: pd.Series) -> Tuple[float, Optional[int]]:
    """Compute peak-to-trough max drawdown and trading days from trough back to prior peak."""
    running_peak = cum_curve.cummax()
    drawdowns = (cum_curve - running_peak) / running_peak
    trough_pos = int(np.argmin(drawdowns.to_numpy(dtype=np.float64)))
    max_dd = float(abs(drawdowns.iloc[trough_pos]))
    if max_dd <= 0.0:
        return 0.0, 0

    peak_level = float(running_peak.iloc[trough_pos])
    post_trough = cum_curve.iloc[trough_pos:]
    recovered = post_trough[post_trough >= peak_level]
    if recovered.empty:
        return max_dd, None
    recovery_pos = int(cum_curve.index.get_loc(recovered.index[0]))
    return max_dd, max(0, recovery_pos - trough_pos)


def _compute_downside_capture(port_ret: pd.Series, bench_ret: pd.Series) -> float:
    """Compute downside capture ratio (%) during sessions when benchmark return < 0."""
    down_mask = bench_ret < 0.0
    if int(down_mask.sum()) == 0:
        return 0.0
    port_down = port_ret.loc[down_mask]
    bench_down = bench_ret.loc[down_mask]
    n_down = len(bench_down)
    port_geo = float((1.0 + port_down).prod() ** (1.0 / n_down) - 1.0)
    bench_geo = float((1.0 + bench_down).prod() ** (1.0 / n_down) - 1.0)
    if abs(bench_geo) < 1e-12:
        return 0.0
    return float(port_geo / bench_geo)


def _evaluate_series_in_window(
    series: pd.Series,
    spy_series: pd.Series,
) -> Dict[str, Any]:
    """Compute cumulative/annualized return, excess spread, MDD, recovery, and downside capture."""
    clean = series.dropna().astype(np.float64)
    n_obs = len(clean)
    if n_obs == 0:
        return {
            "sessions": 0,
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "excess_cumulative_vs_spy": 0.0,
            "annualized_volatility": 0.0,
            "max_drawdown": 0.0,
            "recovery_days": None,
            "downside_capture_vs_spy": 0.0,
        }

    cum_curve = (1.0 + clean).cumprod()
    cum_ret = float(cum_curve.iloc[-1] - 1.0)
    years = n_obs / float(TRADING_DAYS_PER_YEAR)
    ann_ret = float(float(cum_curve.iloc[-1]) ** (1.0 / years) - 1.0) if years > 0.0 else 0.0
    ann_vol = float(clean.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)) if n_obs > 1 else 0.0

    aligned_spy = spy_series.reindex(clean.index).fillna(0.0).astype(np.float64)
    spy_cum_ret = float((1.0 + aligned_spy).prod() - 1.0)
    max_dd, recovery_days = _compute_drawdown_and_recovery(cum_curve)
    downside_cap = _compute_downside_capture(clean, aligned_spy)

    return {
        "sessions": int(n_obs),
        "cumulative_return": cum_ret,
        "annualized_return": ann_ret,
        "excess_cumulative_vs_spy": float(cum_ret - spy_cum_ret),
        "annualized_volatility": ann_vol,
        "max_drawdown": max_dd,
        "recovery_days": recovery_days,
        "downside_capture_vs_spy": downside_cap,
    }


def evaluate_macro_regimes(
    results_path: Path = Path(__file__).resolve().parent.parent / "data" / "backtest_results.csv",
    regimes: Tuple[MacroRegimeSpec, ...] = MACRO_REGIMES,
) -> List[Dict[str, Any]]:
    """Compute regime attribution metrics across all configured macro event windows."""
    df = pd.read_csv(results_path, index_col=0, parse_dates=True)
    target_cols = ("Black_CEO_Net", "Convex_Quality_Net", "SPY_Daily", "RSP_Daily")

    report: List[Dict[str, Any]] = []
    for spec in regimes:
        window = df.loc[spec.start_date : spec.end_date]
        if window.empty:
            continue

        spy_slice = window["SPY_Daily"] if "SPY_Daily" in window.columns else pd.Series(dtype=float)
        if "Active_Constituents_Nt" in window.columns:
            nt_series = window["Active_Constituents_Nt"].dropna()
            n_min = int(nt_series.min()) if not nt_series.empty else 0
            n_max = int(nt_series.max()) if not nt_series.empty else 0
        else:
            n_min, n_max = 0, 0

        series_metrics: Dict[str, Dict[str, Any]] = {}
        for col in target_cols:
            if col not in window.columns:
                continue
            sub = window[col]
            if col == "RSP_Daily":
                sub = sub.loc[sub.index >= pd.Timestamp("2003-05-01")]
            series_metrics[col] = _evaluate_series_in_window(sub, spy_slice)

        report.append(
            {
                "regime_id": spec.regime_id,
                "label": spec.label,
                "start_date": spec.start_date,
                "end_date": spec.end_date,
                "sessions": int(len(window)),
                "active_constituents_min": n_min,
                "active_constituents_max": n_max,
                "series": series_metrics,
            }
        )
    return report


def export_macro_regimes_to_summary(
    macro_report: List[Dict[str, Any]],
    summary_path: Path = Path(__file__).resolve().parent.parent / "data" / "summary_metrics.json",
) -> Path:
    """Persist macro event regime attribution into data/summary_metrics.json."""
    payload = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    payload["macro_regime_attribution"] = macro_report
    summary_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary_path


def _cli_main() -> None:
    """Run macro regime attribution, persist to summary_metrics.json, and print table."""
    report = evaluate_macro_regimes()
    export_macro_regimes_to_summary(report)
    print("--- Macro Event Regime Attribution (2000–2026) ---")
    for row in report:
        ew = row["series"].get("Black_CEO_Net", {})
        spy = row["series"].get("SPY_Daily", {})
        rec = ew.get("recovery_days")
        rec_str = f"{rec}d" if rec is not None else "Unrec."
        print(
            f"{row['label']:38s} | "
            f"N={row['active_constituents_min']:2d}-{row['active_constituents_max']:2d} | "
            f"EW Cum={ew.get('cumulative_return', 0.0):+7.2%} "
            f"(SPY {spy.get('cumulative_return', 0.0):+7.2%}) | "
            f"MDD={ew.get('max_drawdown', 0.0):6.2%} (Rec {rec_str:>6s}) | "
            f"DownCap={ew.get('downside_capture_vs_spy', 0.0):5.2f}x"
        )


if __name__ == "__main__":
    _cli_main()
