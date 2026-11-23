"""
src/export_web_payload.py
Aggregates backtest, roster tenure analytics, factor attribution, Monte Carlo bootstrap,
and GARCH(1,1) diagnostics into web/data/dashboard_payload.json for the interactive UI.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Dict, List

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
WEB_DATA_DIR = BASE_DIR / "web" / "data"

DELISTED_TICKERS = {"AET", "GBT", "JCP", "QRTEA", "PBPB", "WBA"}
PORTRAIT_PALETTE = (
    ("#00d4ff", "#0a2540"),
    ("#00e676", "#0b2e1e"),
    ("#ffb300", "#332200"),
    ("#b388ff", "#21123d"),
    ("#ff6e40", "#38150a"),
    ("#40c4ff", "#0c2738"),
)


def _clean_float(val: Any, digits: int = 6) -> float | None:
    """Convert numeric value to finite rounded float or None."""
    if val is None or pd.isna(val):
        return None
    fval = float(val)
    if not np.isfinite(fval):
        return None
    return round(fval, digits)


def _initials(name: str) -> str:
    """Extract clean two-letter monogram initials for executive avatar badges."""
    skip = {"DR", "JR", "SR", "III", "II"}
    parts = [p for p in name.replace(".", "").split() if p.upper() not in skip]
    if not parts:
        return "CE"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return f"{parts[0][0]}{parts[-1][0]}".upper()


def _sample_sparkline(window: pd.Series) -> List[float]:
    """Normalize a price window to base-100 and downsample to <=36 points."""
    norm = (window / window.iloc[0]) * 100.0
    sampled = norm.iloc[:: max(1, len(norm) // 36)]
    if sampled.index[-1] != norm.index[-1]:
        sampled = pd.concat([sampled, norm.iloc[[-1]]])
    return [round(float(v), 2) for v in sampled.tolist()]


def _compute_tenure_stats(
    prices_df: pd.DataFrame,
    ticker: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> Dict[str, Any]:
    """Compute start, peak, end prices, tenure return, drawdown, and normalized sparkline."""
    empty_stats: Dict[str, Any] = {
        "start_price": None,
        "peak_price": None,
        "peak_date": None,
        "end_price": None,
        "tenure_return_pct": None,
        "tenure_mdd_pct": None,
        "sparkline": [],
    }
    if ticker not in prices_df.columns:
        return empty_stats

    series = prices_df[ticker].dropna()
    window = series[(series.index >= start_ts) & (series.index <= end_ts)]
    if window.empty:
        return empty_stats

    start_price = _clean_float(window.iloc[0], 2)
    end_price = _clean_float(window.iloc[-1], 2)
    peak_idx = window.idxmax()
    tenure_return_pct = (
        round(((end_price / start_price) - 1.0) * 100.0, 2)
        if start_price and start_price > 0 and end_price is not None
        else None
    )

    return {
        "start_price": start_price,
        "peak_price": _clean_float(window.loc[peak_idx], 2),
        "peak_date": peak_idx.strftime("%Y-%m-%d"),
        "end_price": end_price,
        "tenure_return_pct": tenure_return_pct,
        "tenure_mdd_pct": round(abs(float(((window / window.cummax()) - 1.0).min())) * 100.0, 2),
        "sparkline": _sample_sparkline(window),
    }


def _format_dossier_row(
    idx: int,
    row: pd.Series,
    active_final: pd.Series,
    equal_weight_final: float,
    prices_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Build a single CEO dossier dictionary from a roster row."""
    ticker = str(row["ticker"]).strip().upper()
    ceo = str(row["ceo_name"]).strip()
    start_ts = pd.to_datetime(row["start_date"])
    end_ts = pd.to_datetime(row["end_date"])
    is_active = bool(
        ticker in active_final.index
        and bool(active_final[ticker])
        and end_ts >= pd.Timestamp("2026-10-31")
    )
    palette = PORTRAIT_PALETTE[idx % len(PORTRAIT_PALETTE)]

    return {
        "id": f"{ticker}-{idx + 1}",
        "ticker": ticker,
        "company_name": str(row["company_name"]).strip(),
        "ceo_name": ceo,
        "sector": str(row["sector"]).strip(),
        "start_date": start_ts.strftime("%Y-%m-%d"),
        "end_date": end_ts.strftime("%Y-%m-%d"),
        "tenure_days": int((end_ts - start_ts).days),
        "status": (
            "ACTIVE"
            if is_active
            else ("DELISTED" if ticker in DELISTED_TICKERS else "EXITED / HISTORICAL")
        ),
        "current_weight_pct": equal_weight_final if is_active else 0.0,
        **_compute_tenure_stats(prices_df, ticker, start_ts, end_ts),
        "avatar": {"initials": _initials(ceo), "fg": palette[0], "bg": palette[1]},
    }


def build_ceo_dossiers(
    roster_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    mask_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Compute tenure analytics, peak stock prices, sparklines, and status per CEO."""
    active_final = mask_df.loc[mask_df.index[-1]]
    active_count = int(active_final.sum())
    equal_weight_final = round(100.0 / active_count, 2) if active_count > 0 else 0.0

    return [
        _format_dossier_row(idx, row, active_final, equal_weight_final, prices_df)
        for idx, row in roster_df.iterrows()
    ]


def _build_time_series_bundle(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Extract aligned daily curves and conditional volatility arrays for Plotly rendering."""
    dates = results_df.index.strftime("%Y-%m-%d").tolist()

    def _wealth_from_returns(ret_col: str) -> List[float]:
        rets = results_df[ret_col].fillna(0.0).to_numpy(dtype=float)
        wealth = 10000.0 * np.cumprod(1.0 + rets)
        return [round(float(v), 2) for v in wealth]

    def _drawdown_from_wealth(wealth_vals: List[float]) -> List[float]:
        arr = np.asarray(wealth_vals, dtype=float)
        dd = ((arr / np.maximum.accumulate(arr)) - 1.0) * 100.0
        return [round(float(v), 3) for v in dd]

    eq_wealth = _wealth_from_returns("Black_CEO_Net")
    spy_wealth = _wealth_from_returns("SPY_Daily")
    cq_wealth = (
        _wealth_from_returns("Convex_Quality_Net")
        if "Convex_Quality_Net" in results_df.columns
        else eq_wealth
    )

    rsp_wealth: List[float | None] = []
    if "RSP_Daily" in results_df.columns:
        running = 10000.0
        for dt_val, r_val in results_df["RSP_Daily"].items():
            if dt_val < pd.Timestamp("2003-05-01") or pd.isna(r_val):
                rsp_wealth.append(None)
            else:
                running *= 1.0 + float(r_val)
                rsp_wealth.append(round(running, 2))
    else:
        rsp_wealth = [None] * len(dates)

    active_counts = (
        results_df["Active_Count"].fillna(0).astype(int).tolist()
        if "Active_Count" in results_df.columns
        else [0] * len(dates)
    )

    return {
        "dates": dates,
        "wealth_10k": {
            "Black_CEO_Net": eq_wealth,
            "Convex_Quality_Net": cq_wealth,
            "SPY": spy_wealth,
            "RSP": rsp_wealth,
        },
        "drawdown_pct": {
            "Black_CEO_Net": _drawdown_from_wealth(eq_wealth),
            "Convex_Quality_Net": _drawdown_from_wealth(cq_wealth),
            "SPY": _drawdown_from_wealth(spy_wealth),
        },
        "active_count": active_counts,
    }


def export_web_payload(output_path: Path = WEB_DATA_DIR / "dashboard_payload.json") -> Path:
    """Build and write unified dashboard_payload.json and sync summary_metrics.json."""
    roster_df = pd.read_csv(DATA_DIR / "black_ceo_roster.csv")
    prices_df = pd.read_csv(DATA_DIR / "prices_daily.csv", index_col=0, parse_dates=True)
    mask_df = pd.read_csv(DATA_DIR / "universe_mask.csv", index_col=0, parse_dates=True)
    results_df = pd.read_csv(DATA_DIR / "backtest_results.csv", index_col=0, parse_dates=True)

    summary_metrics = json.loads((DATA_DIR / "summary_metrics.json").read_text(encoding="utf-8"))
    mc_results = json.loads((DATA_DIR / "mc_results.json").read_text(encoding="utf-8"))
    ts_diag = json.loads((DATA_DIR / "timeseries_diagnostics.json").read_text(encoding="utf-8"))
    if "timeseries_garch" not in summary_metrics:
        summary_metrics["timeseries_garch"] = ts_diag.get("garch_symmetric", {})
        (DATA_DIR / "summary_metrics.json").write_text(
            json.dumps(summary_metrics, indent=2) + "\n",
            encoding="utf-8",
        )
    ff5_df = pd.read_csv(DATA_DIR / "factor_loadings.csv")

    dossiers = build_ceo_dossiers(roster_df, prices_df, mask_df)
    ts_bundle = _build_time_series_bundle(results_df)

    payload: Dict[str, Any] = {
        "metadata": {
            "title": "Black CEO Leadership & Risk-Adjusted Equity Performance (2000–2026)",
            "horizon_start": ts_bundle["dates"][0],
            "horizon_end": ts_bundle["dates"][-1],
            "total_sessions": len(ts_bundle["dates"]),
            "total_ceo_tenures": len(dossiers),
            "unique_issuers": int(roster_df["ticker"].nunique()),
            "active_issuers_end": sum(1 for d in dossiers if d["status"] == "ACTIVE"),
            "transaction_cost_bps": 10,
            "execution_lag_days": 1,
        },
        "summary_metrics": summary_metrics,
        "factor_loadings": json.loads(ff5_df.to_json(orient="records")),
        "monte_carlo": mc_results,
        "timeseries_diagnostics": ts_diag,
        "ceo_dossiers": dossiers,
        "series": ts_bundle,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(DATA_DIR / "summary_metrics.json", WEB_DATA_DIR / "summary_metrics.json")
    return output_path


if __name__ == "__main__":
    exported = export_web_payload()
    print(f"Exported unified web dashboard payload to {exported}")
