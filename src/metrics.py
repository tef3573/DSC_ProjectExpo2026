"""
src/metrics.py
Institutional Portfolio Performance Analytics Engine.
- Sub-issue #114: Correct annualization factor for Sharpe ratio numerator (252 vs sqrt(252)).
- Sub-issue #115: Compute downside semi-deviation over total sample length T for Sortino ratio.
- Sub-issue #116: Calmar ratio and cumulative drawdown tracking functions.
- Sub-issue #120: Align SPY cap-weighted comparison across full 2000-2026 horizon.
- Sub-issue #121: Isolate RSP equal-weighted comparison starting May 1, 2003.
- Sub-issue #122: Filter pre-inception zero-return entries from RSP metrics.
"""
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

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
    min_dd = float(drawdown.min())
    return abs(min_dd)


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
    cagr = (final_wealth ** (1.0 / years)) - 1.0
    return float(cagr)


def compute_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
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
    """
    Sub-issue #120: Aligns strategy and benchmark series on an identical calendar window.
    """
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
    """
    Sub-issues #121 & #122: Isolates strategy and RSP starting at inception,
    filtering out any flat zero-return artifacts from pre-inception periods.
    """
    strat, rsp = align_benchmark_series(strategy_returns, rsp_returns, start_date=inception_date)

    # Sub-issue #122: Find first non-zero return session if any leading zeros exist at boundary
    non_zero_indices = rsp[rsp != 0.0].index
    if len(non_zero_indices) > 0:
        first_active_date = non_zero_indices[0]
        # Only trim if the leading session was a placeholder 0.0 before first active trade
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
    """
    Sub-issues #120, #121, & #122: Computes comparative institutional metrics against a reference benchmark.
    Returns: CAGR, Volatility, Sharpe, Max Drawdown, Beta, Alpha, and Tracking Error.
    """
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
