"""
src/backtest.py
Vectorized Portfolio Backtesting Engine with Institutional Friction and Artifact Export.

- Sub-issues #104, #106, #107, #108, #110: Core simulation and friction mechanics.
- Sub-issue #127: Export comprehensive daily time series to data/backtest_results.csv.
- Sub-issue #129: Format console scorecard table for clean terminal output.
- Sub-issue #135: Clean code refactor and pylint compliance.
- Sub-issue #179: Integrate src/optimizer.py allocation schemes with src/backtest.py engine.
- Sub-issue #180: Generate comparative performance curves for all 3 schemes.
- Sub-issue #181: Sync monthly rebalancing calendar across optimizers.
- Sub-issue #189: Enforce PEP 8 line-length and pylint standards.
"""
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging
import numpy as np
import pandas as pd

from src.metrics import (
    build_summary_metrics_dict,
    export_summary_metrics,
)
from src.optimizer import (
    CovarianceEstimator,
    ConvexPortfolioOptimizer,
    compute_inverse_volatility_weights,
    compute_fundamental_quality_scores,
    DEFAULT_LOOKBACK_DAYS,
    MIN_OBSERVATIONS_REQUIRED,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PRICES_PATH = DATA_DIR / "prices_daily.csv"
MASK_PATH = DATA_DIR / "universe_mask.csv"
DEFAULT_EXPORT_PATH = DATA_DIR / "backtest_results.csv"
COMPARATIVE_EXPORT_PATH = DATA_DIR / "comparative_schemes.csv"

VALID_ALLOCATION_SCHEMES = ("equal_weight", "inverse_vol", "convex_quality")


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Computes arithmetic daily percentage returns without deprecation warnings."""
    return prices.pct_change(fill_method=None).fillna(0.0)


def compute_equal_weights(
    universe_mask: pd.DataFrame,
    min_diversification: int = 3
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Computes dynamic equal target weights with cash-drag concentration defense."""
    n_t = universe_mask.sum(axis=1)
    stock_weights = pd.DataFrame(0.0, index=universe_mask.index, columns=universe_mask.columns)

    normal_mask = n_t >= min_diversification
    for col in universe_mask.columns:
        stock_weights.loc[normal_mask, col] = np.where(
            universe_mask.loc[normal_mask, col],
            1.0 / n_t.loc[normal_mask],
            0.0
        )

    defensive_mask = (n_t > 0) & (n_t < min_diversification)
    for col in universe_mask.columns:
        stock_weights.loc[defensive_mask, col] = np.where(
            universe_mask.loc[defensive_mask, col],
            1.0 / min_diversification,
            0.0
        )

    equity_total = stock_weights.sum(axis=1)
    cash_weights = 1.0 - equity_total
    return stock_weights, cash_weights, n_t


def get_rebalance_mask(
    universe_mask: pd.DataFrame,
    rebalance_monthly: bool = True
) -> np.ndarray:
    """
    Generates a synchronized boolean rebalancing schedule across all optimizers.

    Triggers a rebalance on:
      1. The initial session (t = 0)
      2. The first trading session of each calendar month (if rebalance_monthly=True)
      3. Any session where active universe membership changes (entry/exit event)
    """
    mask_arr = universe_mask.values.astype(bool)
    t_len = mask_arr.shape[0]
    dates = universe_mask.index

    if not rebalance_monthly:
        return np.ones(t_len, dtype=bool)

    rebal_flags = np.zeros(t_len, dtype=bool)
    for idx in range(t_len):
        if idx == 0:
            rebal_flags[idx] = True
            continue
        is_month_start = (
            dates[idx].month != dates[idx - 1].month
            or dates[idx].year != dates[idx - 1].year
        )
        membership_changed = not np.array_equal(mask_arr[idx], mask_arr[idx - 1])
        rebal_flags[idx] = is_month_start or membership_changed

    return rebal_flags


def _apply_synchronized_rebalance_schedule(
    raw_target_weights: np.ndarray,
    prior_returns: np.ndarray,
    active_mask: np.ndarray,
    rebalance_flags: np.ndarray,
    min_diversification: int = 3
) -> np.ndarray:
    """Holds target weights on rebalance dates and drifts intra-month allocations with returns."""
    t_len, n_assets = raw_target_weights.shape
    scheduled_weights = np.zeros((t_len, n_assets), dtype=np.float64)
    current_weights = np.zeros(n_assets, dtype=np.float64)

    for idx in range(t_len):
        n_active = int(np.sum(active_mask[idx]))
        if n_active == 0:
            current_weights = np.zeros(n_assets, dtype=np.float64)
        elif rebalance_flags[idx]:
            current_weights = raw_target_weights[idx].copy()
        else:
            grown = current_weights * (1.0 + prior_returns[idx])
            total_eq = float(np.sum(grown))
            target_budget = (
                (n_active / min_diversification) if n_active < min_diversification else 1.0
            )
            if total_eq > 0.0:
                current_weights = (grown / total_eq) * target_budget
            else:
                current_weights = raw_target_weights[idx].copy()

        scheduled_weights[idx] = current_weights

    return scheduled_weights


def _compute_convex_quality_schedule(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    prior_returns: pd.DataFrame,
    mask_arr: np.ndarray,
    rebal_flags: np.ndarray,
    min_diversification: int,
    lookback_window: int,
    lambda_tilt: float
) -> np.ndarray:
    """Computes rolling convex quality QP target weights with intra-month price drift."""
    t_len, n_assets = mask_arr.shape
    weights_arr = np.zeros((t_len, n_assets), dtype=np.float64)

    cov_estimator = CovarianceEstimator(lookback_window=lookback_window)
    qp_optimizer = ConvexPortfolioOptimizer()
    quality_matrix = compute_fundamental_quality_scores(
        returns=prior_returns,
        active_mask=mask_arr,
        lookback_window=lookback_window
    )

    ret_values = prior_returns.values
    current_weights = np.zeros(n_assets, dtype=np.float64)

    for idx in range(t_len):
        active_indices = np.where(mask_arr[idx])[0]
        n_active = len(active_indices)

        if n_active == 0:
            current_weights = np.zeros(n_assets, dtype=np.float64)
            weights_arr[idx] = current_weights
            continue

        if rebal_flags[idx]:
            equity_budget = (1.0 / min_diversification) if n_active < min_diversification else 1.0
            start_idx = max(0, idx - lookback_window + 1)
            window_slice = prior_returns.iloc[start_idx:idx + 1, active_indices]

            if len(window_slice) >= 2 and n_active >= 2:
                cov_sub, _ = cov_estimator.estimate(window_slice)
                w_sub, _ = qp_optimizer.solve(
                    cov_matrix=cov_sub,
                    quality_scores=quality_matrix[idx, active_indices],
                    lambda_tilt=lambda_tilt,
                    equity_budget=equity_budget
                )
            else:
                w_sub = np.full(n_active, equity_budget / n_active, dtype=np.float64)

            current_weights = np.zeros(n_assets, dtype=np.float64)
            current_weights[active_indices] = w_sub
        else:
            grown = current_weights * (1.0 + ret_values[idx])
            total_eq = float(np.sum(grown))
            target_budget = (
                (n_active / min_diversification) if n_active < min_diversification else 1.0
            )
            if total_eq > 0.0:
                current_weights = (grown / total_eq) * target_budget

        weights_arr[idx] = current_weights

    return weights_arr


def compute_target_weights(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    universe_mask: pd.DataFrame,
    asset_returns: Optional[pd.DataFrame] = None,
    allocation_scheme: str = "equal_weight",
    min_diversification: int = 3,
    lookback_window: int = DEFAULT_LOOKBACK_DAYS,
    lambda_tilt: float = 0.5,
    rebalance_monthly: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Dispatches target weight generation across supported optimization schemes."""
    if allocation_scheme not in VALID_ALLOCATION_SCHEMES:
        raise ValueError(
            f"Unsupported allocation_scheme '{allocation_scheme}'. "
            f"Expected one of {VALID_ALLOCATION_SCHEMES}."
        )

    tickers = list(universe_mask.columns)
    mask_arr = universe_mask.values.astype(bool)
    n_t = universe_mask.sum(axis=1)

    if allocation_scheme == "equal_weight":
        ew_weights, ew_cash, _ = compute_equal_weights(
            universe_mask, min_diversification=min_diversification
        )
        if asset_returns is not None and rebalance_monthly:
            prior_ret = asset_returns[tickers].loc[universe_mask.index].shift(1).fillna(0.0).values
            synced_arr = _apply_synchronized_rebalance_schedule(
                raw_target_weights=ew_weights.values,
                prior_returns=prior_ret,
                active_mask=mask_arr,
                rebalance_flags=get_rebalance_mask(universe_mask, rebalance_monthly=True),
                min_diversification=min_diversification
            )
            ew_weights = pd.DataFrame(synced_arr, index=universe_mask.index, columns=tickers)
            ew_cash = 1.0 - ew_weights.sum(axis=1)
        return ew_weights, ew_cash, n_t

    if asset_returns is None:
        raise ValueError(f"asset_returns is required for allocation_scheme='{allocation_scheme}'.")

    prior_returns = asset_returns[tickers].loc[universe_mask.index].shift(1).fillna(0.0)
    rebal_flags = get_rebalance_mask(universe_mask, rebalance_monthly=rebalance_monthly)

    if allocation_scheme == "inverse_vol":
        raw_iv_arr = compute_inverse_volatility_weights(
            returns=prior_returns,
            active_mask=mask_arr,
            lookback_window=MIN_OBSERVATIONS_REQUIRED,
            concentration_defense=True
        )
        weights_arr = _apply_synchronized_rebalance_schedule(
            raw_target_weights=raw_iv_arr,
            prior_returns=prior_returns.values,
            active_mask=mask_arr,
            rebalance_flags=rebal_flags,
            min_diversification=min_diversification
        )
    else:
        weights_arr = _compute_convex_quality_schedule(
            prior_returns=prior_returns,
            mask_arr=mask_arr,
            rebal_flags=rebal_flags,
            min_diversification=min_diversification,
            lookback_window=lookback_window,
            lambda_tilt=lambda_tilt
        )

    stock_weights = pd.DataFrame(weights_arr, index=universe_mask.index, columns=tickers)
    cash_weights = 1.0 - stock_weights.sum(axis=1)
    return stock_weights, cash_weights, n_t


def compute_weight_drift(
    target_weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    portfolio_gross_returns: pd.Series
) -> pd.DataFrame:
    """Computes post-return drifted weights at the close of trading session t."""
    aligned_returns = asset_returns[target_weights.columns]
    num = target_weights * (1.0 + aligned_returns)
    denom = 1.0 + portfolio_gross_returns.values[:, np.newaxis]
    denom = np.where(denom == 0.0, 1.0, denom)
    return num / denom


def compute_turnover_and_costs(
    target_weights: pd.DataFrame,
    drifted_weights: pd.DataFrame,
    fee_bps: float = 10.0
) -> Tuple[pd.Series, pd.Series]:
    """Computes one-way portfolio rebalance turnover and fee deductions."""
    cost_rate = fee_bps / 10000.0
    drifted_prior = drifted_weights.shift(1).fillna(0.0)

    equity_diff = (target_weights - drifted_prior).abs().sum(axis=1)
    target_cash = 1.0 - target_weights.sum(axis=1)
    drifted_cash = 1.0 - drifted_prior.sum(axis=1)
    cash_diff = (target_cash - drifted_cash).abs()

    one_way_turnover = 0.5 * (equity_diff + cash_diff)
    one_way_turnover.iloc[0] = 0.0

    costs = one_way_turnover * cost_rate
    costs.iloc[0] = 0.0
    return one_way_turnover, costs


def export_backtest_results(
    results_df: pd.DataFrame,
    export_path: Path = DEFAULT_EXPORT_PATH
) -> Path:
    """Formats and exports complete daily backtest time series including optimized curves."""
    export_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = results_df.copy()
    if COMPARATIVE_EXPORT_PATH.exists():
        comp_df = pd.read_csv(COMPARATIVE_EXPORT_PATH, index_col=0, parse_dates=True)
        for col in ("Convex_Quality_Net", "Convex_Quality_Cumulative", "Convex_Quality_Drawdown"):
            if col in comp_df.columns and col not in out_df.columns:
                out_df[col] = comp_df[col].reindex(out_df.index)
    out_df.to_csv(export_path, index=True)
    logging.info("Sub-issue #127 & #192: Exported daily time series to %s", export_path)
    return export_path


def format_scorecard_table(summary: Dict[str, Any]) -> str:
    """Formats summary metrics into an aligned institutional ASCII table."""
    meta = summary.get("metadata", {})
    strat = summary.get("strategy_performance", {})
    spy = summary.get("benchmark_spy_full_horizon", {})
    rsp = summary.get("benchmark_rsp_isolated", {})
    garch = summary.get("timeseries_garch", {})
    garch_strat = garch.get("Black_CEO_Net", {})
    garch_spy = garch.get("SPY_Daily", {})

    width = 86
    horizon_str = (
        f"Horizon: {meta.get('backtest_start', '')} to {meta.get('backtest_end', '')} "
        f"({meta.get('trading_sessions', '')} Sessions)"
    )
    lines = [
        "=" * width,
        f"{'BLACK CEO PORTFOLIO: INSTITUTIONAL PERFORMANCE SCORECARD':^{width}}",
        f"{horizon_str:^{width}}",
        "=" * width,
        f"{'Performance Metric':<32} {'Strategy (Net)':>16} {'SPY (Cap-Wtd)':>16} {'RSP (Eq-Wtd)*':>18}",
        "-" * width,
        (
            f"{'CAGR':<32} {strat.get('cagr_net', 0.0):>15.2%} "
            f"{spy.get('benchmark_cagr', 0.0):>15.2%} {rsp.get('benchmark_cagr', 0.0):>17.2%}"
        ),
        (
            f"{'Annualized Volatility':<32} {strat.get('annualized_volatility', 0.0):>15.2%} "
            f"{spy.get('benchmark_volatility', 0.0):>15.2%} "
            f"{rsp.get('benchmark_volatility', 0.0):>17.2%}"
        ),
        (
            f"{'Sharpe Ratio (Rf=1.5%)':<32} {strat.get('sharpe_ratio', 0.0):>16.2f} "
            f"{spy.get('benchmark_sharpe', 0.0):>16.2f} {rsp.get('benchmark_sharpe', 0.0):>18.2f}"
        ),
        f"{'Sortino Ratio':<32} {strat.get('sortino_ratio', 0.0):>16.2f} {'N/A':>16} {'N/A':>18}",
        f"{'Calmar Ratio':<32} {strat.get('calmar_ratio', 0.0):>16.2f} {'N/A':>16} {'N/A':>18}",
        (
            f"{'Maximum Drawdown (MDD)':<32} {strat.get('max_drawdown', 0.0):>15.2%} "
            f"{spy.get('benchmark_mdd', 0.0):>15.2%}{rsp.get('benchmark_mdd', 0.0):>17.2%}"
        ),
        "-" * width,
        f"{'Beta vs Benchmark':<32} {'N/A':>16} {spy.get('beta', 1.0):>16.2f} {rsp.get('beta', 1.0):>18.2f}",
        (
            f"{'Jensen Alpha (Annualized)':<32} {'N/A':>16} "
            f"{spy.get('alpha_annual', 0.0):>15.2%} {rsp.get('alpha_annual', 0.0):>17.2%}"
        ),
        (
            f"{'Tracking Error':<32} {'N/A':>16} "
            f"{spy.get('tracking_error', 0.0):>15.2%} {rsp.get('tracking_error', 0.0):>17.2%}"
        ),
        "-" * width,
        (
            f"{'Cumulative Wealth Multiple':<32} "
            f"{strat.get('cumulative_multiple_net', 1.0):>15.2f}x {'N/A':>16} {'N/A':>18}"
        ),
        (
            f"{'Annualized Turnover':<32} "
            f"{strat.get('annualized_turnover', 0.0):>15.2%} {'N/A':>16} {'N/A':>18}"
        ),
        (
            f"{'Total Incurred Friction Costs':<32} "
            f"{strat.get('total_transaction_friction_cost', 0.0):>15.2%} {'N/A':>16} {'N/A':>18}"
        ),
        "-" * width,
        f"{'GARCH(1,1) Persistence (a+b)':<32} "
        f"{garch_strat.get('persistence', 0.0):>16.4f} "
        f"{garch_spy.get('persistence', 0.0):>16.4f} {'N/A':>18}",
        f"{'GARCH(1,1) Half-Life (Days)':<32} "
        f"{garch_strat.get('half_life_days', 0.0):>15.1f}d "
        f"{garch_spy.get('half_life_days', 0.0):>15.1f}d {'N/A':>18}",
        f"{'GARCH(1,1) Unconditional Vol':<32} "
        f"{garch_strat.get('unconditional_ann_vol', 0.0):>15.2%} "
        f"{garch_spy.get('unconditional_ann_vol', 0.0):>15.2%} {'N/A':>18}",
        "=" * width,
        "* Note: RSP equal-weighted comparison is isolated starting May 1, 2003.",
        "=" * width,
    ]
    return "\n".join(lines)


def _compute_benchmark_curves(
    returns: pd.DataFrame,
    common_idx: pd.Index,
    ticker: str
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Computes daily returns, cumulative wealth, and drawdown curves for a benchmark ticker."""
    bench_ret = returns[ticker] if ticker in returns.columns else pd.Series(0.0, index=common_idx)
    bench_cum = (1.0 + bench_ret).cumprod()
    bench_peak = bench_cum.cummax()
    bench_dd = (bench_cum - bench_peak) / bench_peak
    return bench_ret, bench_cum, bench_dd


def run_backtest(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    prices: pd.DataFrame,
    mask: pd.DataFrame,
    fee_bps: float = 10.0,
    rf_annual: float = 0.015,
    min_diversification: int = 3,
    export_results: bool = True,
    allocation_scheme: str = "equal_weight",
    lambda_tilt: float = 0.5,
    rebalance_monthly: bool = True
) -> pd.DataFrame:
    """Executes full vectorized point-in-time backtest and generates daily performance curves."""
    returns = compute_returns(prices)
    common_idx = prices.index.intersection(mask.index)
    prices = prices.loc[common_idx]
    mask = mask.loc[common_idx]
    returns = returns.loc[common_idx]

    rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0
    stock_weights, cash_weights, n_t = compute_target_weights(
        universe_mask=mask,
        asset_returns=returns,
        allocation_scheme=allocation_scheme,
        min_diversification=min_diversification,
        lambda_tilt=lambda_tilt,
        rebalance_monthly=rebalance_monthly
    )

    aligned_r = returns[stock_weights.columns]
    portfolio_gross = (stock_weights * aligned_r).sum(axis=1) + cash_weights * rf_daily
    drifted_weights = compute_weight_drift(stock_weights, aligned_r, portfolio_gross)

    turnover, costs = compute_turnover_and_costs(stock_weights, drifted_weights, fee_bps=fee_bps)
    portfolio_net = portfolio_gross - costs

    net_cum = (1.0 + portfolio_net).cumprod()
    strat_peak = net_cum.cummax()
    strat_dd = (net_cum - strat_peak) / strat_peak

    spy_ret, spy_cum, spy_dd = _compute_benchmark_curves(returns, common_idx, "SPY")
    rsp_ret, rsp_cum, rsp_dd = _compute_benchmark_curves(returns, common_idx, "RSP")

    df_out = pd.DataFrame({
        "Black_CEO_Gross": portfolio_gross,
        "Black_CEO_Net": portfolio_net,
        "Black_CEO_Cumulative": net_cum,
        "Cash_Weight": cash_weights,
        "Active_Constituents_Nt": n_t,
        "One_Way_Turnover": turnover,
        "Transaction_Costs": costs,
        "Strategy_Drawdown": strat_dd,
        "SPY_Daily": spy_ret,
        "SPY_Cumulative": spy_cum,
        "SPY_Drawdown": spy_dd,
        "RSP_Daily": rsp_ret,
        "RSP_Cumulative": rsp_cum,
        "RSP_Drawdown": rsp_dd,
    }, index=common_idx)

    if export_results:
        export_backtest_results(df_out)
        summary = build_summary_metrics_dict(df_out, rf_annual=rf_annual)
        export_summary_metrics(summary)

    return df_out


def run_comparative_backtests(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    prices: pd.DataFrame,
    mask: pd.DataFrame,
    fee_bps: float = 10.0,
    rf_annual: float = 0.015,
    min_diversification: int = 3,
    lambda_tilt: float = 0.5,
    rebalance_monthly: bool = True,
    export_comparative: bool = True,
    export_path: Path = COMPARATIVE_EXPORT_PATH
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Executes backtests across all 3 allocation schemes and generates comparative curves."""
    scheme_results: Dict[str, pd.DataFrame] = {}
    scheme_summaries: Dict[str, Dict[str, Any]] = {}

    for scheme in VALID_ALLOCATION_SCHEMES:
        res_df = run_backtest(
            prices=prices,
            mask=mask,
            fee_bps=fee_bps,
            rf_annual=rf_annual,
            min_diversification=min_diversification,
            export_results=(scheme == "equal_weight"),
            allocation_scheme=scheme,
            lambda_tilt=lambda_tilt,
            rebalance_monthly=rebalance_monthly
        )
        scheme_results[scheme] = res_df
        scheme_summaries[scheme] = build_summary_metrics_dict(res_df, rf_annual=rf_annual)

    base_df = scheme_results["equal_weight"]
    comp_df = pd.DataFrame({
        "Equal_Weight_Net": scheme_results["equal_weight"]["Black_CEO_Net"],
        "Equal_Weight_Cumulative": scheme_results["equal_weight"]["Black_CEO_Cumulative"],
        "Equal_Weight_Drawdown": scheme_results["equal_weight"]["Strategy_Drawdown"],
        "Equal_Weight_Turnover": scheme_results["equal_weight"]["One_Way_Turnover"],
        "Inverse_Vol_Net": scheme_results["inverse_vol"]["Black_CEO_Net"],
        "Inverse_Vol_Cumulative": scheme_results["inverse_vol"]["Black_CEO_Cumulative"],
        "Inverse_Vol_Drawdown": scheme_results["inverse_vol"]["Strategy_Drawdown"],
        "Inverse_Vol_Turnover": scheme_results["inverse_vol"]["One_Way_Turnover"],
        "Convex_Quality_Net": scheme_results["convex_quality"]["Black_CEO_Net"],
        "Convex_Quality_Cumulative": scheme_results["convex_quality"]["Black_CEO_Cumulative"],
        "Convex_Quality_Drawdown": scheme_results["convex_quality"]["Strategy_Drawdown"],
        "Convex_Quality_Turnover": scheme_results["convex_quality"]["One_Way_Turnover"],
        "SPY_Cumulative": base_df["SPY_Cumulative"],
        "RSP_Cumulative": base_df["RSP_Cumulative"],
        "Active_Constituents_Nt": base_df["Active_Constituents_Nt"],
    }, index=base_df.index)

    if export_comparative:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        comp_df.to_csv(export_path, index=True)
        logging.info("Sub-issue #180: Exported 3-scheme comparative curves to %s", export_path)

    return comp_df, scheme_summaries


def format_comparative_scorecard(summaries: Dict[str, Dict[str, Any]]) -> str:
    """Formats side-by-side institutional metrics across all 3 schemes."""
    ew = summaries["equal_weight"]["strategy_performance"]
    iv = summaries["inverse_vol"]["strategy_performance"]
    cq = summaries["convex_quality"]["strategy_performance"]
    spy = summaries["equal_weight"]["benchmark_spy_full_horizon"]

    width = 92
    header_cols = (
        f"{'Metric':<28} {'Equal Weight':>14} {'Inverse-Vol':>14} "
        f"{'Convex Quality':>16} {'SPY':>14}"
    )
    lines = [
        "=" * width,
        f"{'BLACK CEO PORTFOLIO: COMPARATIVE ALLOCATION SCHEME SCORECARD':^{width}}",
        "=" * width,
        header_cols,
        "-" * width,
        (
            f"{'CAGR (Net)':<28} {ew['cagr_net']:>13.2%} {iv['cagr_net']:>13.2%} "
            f"{cq['cagr_net']:>15.2%} {spy['benchmark_cagr']:>13.2%}"
        ),
        (
            f"{'Annualized Volatility':<28} {ew['annualized_volatility']:>13.2%} "
            f"{iv['annualized_volatility']:>13.2%} {cq['annualized_volatility']:>15.2%} "
            f"{spy['benchmark_volatility']:>13.2%}"
        ),
        (
            f"{'Sharpe Ratio (Rf=1.5%)':<28} {ew['sharpe_ratio']:>14.2f} "
            f"{iv['sharpe_ratio']:>14.2f} {cq['sharpe_ratio']:>16.2f} "
            f"{spy['benchmark_sharpe']:>14.2f}"
        ),
        (
            f"{'Sortino Ratio':<28} {ew['sortino_ratio']:>14.2f} "
            f"{iv['sortino_ratio']:>14.2f} {cq['sortino_ratio']:>16.2f} {'N/A':>14}"
        ),
        (
            f"{'Maximum Drawdown (MDD)':<28} {ew['max_drawdown']:>13.2%} "
            f"{iv['max_drawdown']:>13.2%} {cq['max_drawdown']:>15.2%} "
            f"{spy['benchmark_mdd']:>13.2%}"
        ),
        (
            f"{'Cumulative Wealth Multiple':<28} {ew['cumulative_multiple_net']:>13.2f}x "
            f"{iv['cumulative_multiple_net']:>13.2f}x {cq['cumulative_multiple_net']:>15.2f}x "
            f"{'N/A':>14}"
        ),
        (
            f"{'Annualized Turnover':<28} {ew['annualized_turnover']:>13.2%} "
            f"{iv['annualized_turnover']:>13.2%} {cq['annualized_turnover']:>15.2%} "
            f"{'N/A':>14}"
        ),
        "=" * width,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    if not PRICES_PATH.exists() or not MASK_PATH.exists():
        logging.error("Missing input artifacts: %s or %s", PRICES_PATH, MASK_PATH)
    else:
        prices_df = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
        mask_df = pd.read_csv(MASK_PATH, index_col=0, parse_dates=True).astype(bool)
        results = run_backtest(prices_df, mask_df, export_results=True)
        summary_dict = build_summary_metrics_dict(results)
        print("\n" + format_scorecard_table(summary_dict) + "\n")
