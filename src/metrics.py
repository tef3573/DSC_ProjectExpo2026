"""
src/metrics.py
Institutional Portfolio Performance Analytics Engine.
- Sub-issues #114-#116: Performance, downside deviation, drawdown, and risk ratios.
- Sub-issues #120-#122: SPY and RSP benchmark alignment and window decoupling.
- Sub-issue #128: Build and export summary metrics dictionary to data/summary_metrics.json.
"""
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import json
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_JSON_PATH = DATA_DIR / "summary_metrics.json"
RSP_INCEPTION_DATE = "2003-05-01"


def compute_sharpe_ratio(
    returns: pd.Series,
    rf_annual: float = 0.015,
    periods_per_year: int = 252
) -> float:
    """Calculates annualized Sharpe Ratio with correct scaling."""
    clean_returns = returns.dropna()
    if len(clean_returns) < 2:
        return 0.0

    rf_daily = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    excess_returns = clean_returns - rf_daily

    daily_mean_excess = excess_returns.mean()
    daily_vol = clean_returns.std(ddof=1)

    if daily_vol == 0.0 or np.isnan(daily_vol):
        return 0.0

    annualized_mean_excess = daily_mean_excess * periods_per_year
    annualized_vol = daily_vol * np.sqrt(periods_per_year)
    return float(annualized_mean_excess / annualized_vol)


def compute_downside_deviation(
    returns: pd.Series,
    rf_annual: float = 0.015,
    periods_per_year: int = 252
) -> float:
    """Computes daily downside semi-deviation divided by total sample length T."""
    clean_returns = returns.dropna()
    t_total = len(clean_returns)
    if t_total < 1:
        return 0.0

    rf_daily = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    excess_returns = clean_returns - rf_daily

    downside = np.minimum(excess_returns.values, 0.0)
    variance_downside = np.sum(downside ** 2) / t_total
    return float(np.sqrt(variance_downside))


def compute_sortino_ratio(
    returns: pd.Series,
    rf_annual: float = 0.015,
    periods_per_year: int = 252
) -> float:
    """Annualized Sortino Ratio using total sample length T downside deviation."""
    clean_returns = returns.dropna()
    if len(clean_returns) < 2:
        return 0.0

    rf_daily = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    excess_returns = clean_returns - rf_daily

    annualized_mean_excess = excess_returns.mean() * periods_per_year
    daily_downside = compute_downside_deviation(clean_returns, rf_annual=rf_annual, periods_per_year=periods_per_year)

    if daily_downside == 0.0 or np.isnan(daily_downside):
        return 0.0

    annualized_downside = daily_downside * np.sqrt(periods_per_year)
    return float(annualized_mean_excess / annualized_downside)


def compute_drawdown_series(returns: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Computes cumulative wealth curve and running drawdown percentage series."""
    clean_returns = returns.dropna()
    if len(clean_returns) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    wealth = (1.0 + clean_returns).cumprod()
    running_peak = wealth.cummax()
    drawdown = (wealth - running_peak) / running_peak
    return wealth, drawdown


def compute_max_drawdown(returns: pd.Series) -> float:
    """Computes Maximum Drawdown (MDD) as an absolute positive decimal."""
    _, drawdown = compute_drawdown_series(returns)
    if len(drawdown) == 0:
        return 0.0
    return abs(float(drawdown.min()))


def compute_cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Computes Compound Annual Growth Rate (CAGR)."""
    clean_returns = returns.dropna()
    t_total = len(clean_returns)
    if t_total == 0:
        return 0.0

    final_wealth = float((1.0 + clean_returns).prod())
    if final_wealth <= 0.0:
        return -1.0

    years = t_total / float(periods_per_year)
    return float((final_wealth ** (1.0 / years)) - 1.0)


def compute_calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Computes Calmar Ratio (CAGR / |MDD|)."""
    mdd = compute_max_drawdown(returns)
    if mdd == 0.0:
        return 0.0
    cagr = compute_cagr(returns, periods_per_year=periods_per_year)
    return float(cagr / mdd)


def align_benchmark_series(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Tuple[pd.Series, pd.Series]:
    """Aligns strategy and benchmark series on an identical calendar window."""
    df = pd.DataFrame({"strategy": strategy_returns, "benchmark": benchmark_returns}).dropna()
    if start_date is not None:
        df = df.loc[df.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df.loc[df.index <= pd.to_datetime(end_date)]
    return df["strategy"], df["benchmark"]


def isolate_rsp_benchmark_series(
    strategy_returns: pd.Series,
    rsp_returns: pd.Series,
    inception_date: str = RSP_INCEPTION_DATE
) -> Tuple[pd.Series, pd.Series]:
    """Isolates strategy and RSP starting at inception, filtering pre-inception zeros."""
    strat, rsp = align_benchmark_series(strategy_returns, rsp_returns, start_date=inception_date)
    non_zero_indices = rsp[rsp != 0.0].index
    if len(non_zero_indices) > 0:
        first_active_date = non_zero_indices[0]
        if first_active_date > rsp.index[0] and (rsp.loc[:first_active_date].iloc[:-1] == 0.0).all():
            strat = strat.loc[strat.index >= first_active_date]
            rsp = rsp.loc[rsp.index >= first_active_date]
    return strat, rsp


def compute_benchmark_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf_annual: float = 0.015,
    periods_per_year: int = 252
) -> Dict[str, Any]:
    """Computes comparative institutional metrics against a reference benchmark."""
    strat, bench = align_benchmark_series(strategy_returns, benchmark_returns)
    if len(strat) < 2:
        return {}

    strat_cagr = compute_cagr(strat, periods_per_year=periods_per_year)
    bench_cagr = compute_cagr(bench, periods_per_year=periods_per_year)
    strat_vol = float(strat.std(ddof=1) * np.sqrt(periods_per_year))
    bench_vol = float(bench.std(ddof=1) * np.sqrt(periods_per_year))

    strat_sharpe = compute_sharpe_ratio(strat, rf_annual=rf_annual, periods_per_year=periods_per_year)
    bench_sharpe = compute_sharpe_ratio(bench, rf_annual=rf_annual, periods_per_year=periods_per_year)

    strat_mdd = compute_max_drawdown(strat)
    bench_mdd = compute_max_drawdown(bench)

    covariance = np.cov(strat.values, bench.values)[0, 1]
    bench_variance = np.var(bench.values, ddof=1)
    beta = float(covariance / bench_variance) if bench_variance > 0 else 1.0

    rf_daily = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    alpha_annual = float(((strat.mean() - rf_daily) - beta * (bench.mean() - rf_daily)) * periods_per_year)

    diff = strat - bench
    tracking_error = float(diff.std(ddof=1) * np.sqrt(periods_per_year))

    return {
        "observations": len(strat),
        "start_date": str(strat.index[0].date()) if hasattr(strat.index[0], "date") else str(strat.index[0]),
        "end_date": str(strat.index[-1].date()) if hasattr(strat.index[-1], "date") else str(strat.index[-1]),
        "strategy_cagr": strat_cagr,
        "benchmark_cagr": bench_cagr,
        "strategy_volatility": strat_vol,
        "benchmark_volatility": bench_vol,
        "strategy_sharpe": strat_sharpe,
        "benchmark_sharpe": bench_sharpe,
        "strategy_mdd": strat_mdd,
        "benchmark_mdd": bench_mdd,
        "beta": beta,
        "alpha_annual": alpha_annual,
        "tracking_error": tracking_error
    }


def build_summary_metrics_dict(
    results_df: pd.DataFrame,
    rf_annual: float = 0.015,
    periods_per_year: int = 252
) -> Dict[str, Any]:
    """
    Sub-issue #128: Builds hierarchical summary metrics dictionary for institutional reporting.
    """
    net_ret = results_df["Black_CEO_Net"]
    gross_ret = results_df["Black_CEO_Gross"]
    spy_ret = results_df["SPY_Daily"]
    rsp_ret = results_df["RSP_Daily"]

    cagr_net = compute_cagr(net_ret, periods_per_year=periods_per_year)
    cagr_gross = compute_cagr(gross_ret, periods_per_year=periods_per_year)
    ann_vol = float(net_ret.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe = compute_sharpe_ratio(net_ret, rf_annual=rf_annual, periods_per_year=periods_per_year)
    sortino = compute_sortino_ratio(net_ret, rf_annual=rf_annual, periods_per_year=periods_per_year)
    mdd = compute_max_drawdown(net_ret)
    calmar = compute_calmar_ratio(net_ret, periods_per_year=periods_per_year)

    cum_mult_net = float(results_df["Black_CEO_Cumulative"].iloc[-1])
    cum_mult_gross = float((1.0 + gross_ret).cumprod().iloc[-1])

    mean_turnover = float(results_df["One_Way_Turnover"].mean())
    ann_turnover = float(mean_turnover * periods_per_year)
    total_costs = float(results_df["Transaction_Costs"].sum())

    # Full horizon SPY comparison
    spy_comparison = compute_benchmark_metrics(net_ret, spy_ret, rf_annual=rf_annual, periods_per_year=periods_per_year)

    # Decoupled post-2003 RSP comparison
    strat_rsp, rsp_clean = isolate_rsp_benchmark_series(net_ret, rsp_ret)
    rsp_comparison = compute_benchmark_metrics(strat_rsp, rsp_clean, rf_annual=rf_annual, periods_per_year=periods_per_year)

    summary = {
        "metadata": {
            "strategy_name": "Black CEO Equal-Weight Portfolio",
            "backtest_start": str(results_df.index[0].date()) if hasattr(results_df.index[0], "date") else str(results_df.index[0]),
            "backtest_end": str(results_df.index[-1].date()) if hasattr(results_df.index[-1], "date") else str(results_df.index[-1]),
            "trading_sessions": len(results_df),
            "rf_annual": rf_annual,
            "min_diversification": 3,
            "transaction_fee_bps": 10.0
        },
        "strategy_performance": {
            "cagr_net": cagr_net,
            "cagr_gross": cagr_gross,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": mdd,
            "cumulative_multiple_net": cum_mult_net,
            "cumulative_multiple_gross": cum_mult_gross,
            "mean_daily_turnover": mean_turnover,
            "annualized_turnover": ann_turnover,
            "total_transaction_friction_cost": total_costs
        },
        "benchmark_spy_full_horizon": spy_comparison,
        "benchmark_rsp_isolated": rsp_comparison
    }
    return summary


def export_summary_metrics(
    summary_dict: Dict[str, Any],
    export_path: Path = DEFAULT_JSON_PATH
) -> Path:
    """
    Sub-issue #128: Serializes summary metrics dictionary to clean formatted JSON.
    """
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=4)
    logging.info(f"Sub-issue #128: Exported summary metrics JSON to {export_path}")
    return export_path
