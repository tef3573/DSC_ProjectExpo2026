"""
src/backtest.py
Vectorized Portfolio Backtesting Engine with Institutional Friction and Artifact Export.
- Sub-issues #104, #106, #107, #108, #110: Core simulation and friction mechanics.
- Sub-issue #127: Export comprehensive daily time series to data/backtest_results.csv.
- Sub-issue #129: Format console scorecard table for clean terminal output.
"""
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging
import numpy as np
import pandas as pd

from src.metrics import (
    build_summary_metrics_dict,
    export_summary_metrics,
    DEFAULT_JSON_PATH
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PRICES_PATH = DATA_DIR / "prices_daily.csv"
MASK_PATH = DATA_DIR / "universe_mask.csv"
DEFAULT_EXPORT_PATH = DATA_DIR / "backtest_results.csv"


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Computes arithmetic daily percentage returns."""
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
    """Sub-issue #127: Formats and exports complete daily backtest time series."""
    export_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(export_path, index=True)
    logging.info(f"Sub-issue #127: Exported daily time series to {export_path}")
    return export_path


def format_scorecard_table(summary: Dict[str, Any]) -> str:
    """Sub-issue #129: Formats summary metrics into an aligned institutional ASCII table."""
    meta = summary.get("metadata", {})
    strat = summary.get("strategy_performance", {})
    spy = summary.get("benchmark_spy_full_horizon", {})
    rsp = summary.get("benchmark_rsp_isolated", {})

    lines = []
    lines.append("=" * 86)
    lines.append(f"{'BLACK CEO PORTFOLIO: INSTITUTIONAL PERFORMANCE SCORECARD':^86}")
    lines.append(f"{'Horizon: ' + meta.get('backtest_start', '') + ' to ' + meta.get('backtest_end', '') + ' (' + str(meta.get('trading_sessions', '')) + ' Sessions)':^86}")
    lines.append("=" * 86)
    lines.append(f"{'Performance Metric':<32} {'Strategy (Net)':>16} {'SPY (Cap-Wtd)':>16} {'RSP (Eq-Wtd)*':>18}")
    lines.append("-" * 86)

    lines.append(f"{'CAGR':<32} {strat.get('cagr_net', 0.0):>15.2%} {spy.get('benchmark_cagr', 0.0):>15.2%} {rsp.get('benchmark_cagr', 0.0):>17.2%}")
    lines.append(f"{'Annualized Volatility':<32} {strat.get('annualized_volatility', 0.0):>15.2%} {spy.get('benchmark_volatility', 0.0):>15.2%} {rsp.get('benchmark_volatility', 0.0):>17.2%}")
    lines.append(f"{'Sharpe Ratio (Rf=1.5%)':<32} {strat.get('sharpe_ratio', 0.0):>16.2f} {spy.get('benchmark_sharpe', 0.0):>16.2f} {rsp.get('benchmark_sharpe', 0.0):>18.2f}")
    lines.append(f"{'Sortino Ratio':<32} {strat.get('sortino_ratio', 0.0):>16.2f} {'N/A':>16} {'N/A':>18}")
    lines.append(f"{'Calmar Ratio':<32} {strat.get('calmar_ratio', 0.0):>16.2f} {'N/A':>16} {'N/A':>18}")
    lines.append(f"{'Maximum Drawdown (MDD)':<32} {strat.get('max_drawdown', 0.0):>15.2%} {spy.get('benchmark_mdd', 0.0):>15.2%} {rsp.get('benchmark_mdd', 0.0):>17.2%}")
    lines.append("-" * 86)

    lines.append(f"{'Beta vs Benchmark':<32} {'N/A':>16} {spy.get('beta', 1.0):>16.2f} {rsp.get('beta', 1.0):>18.2f}")
    lines.append(f"{'Jensen Alpha (Annualized)':<32} {'N/A':>16} {spy.get('alpha_annual', 0.0):>15.2%} {rsp.get('alpha_annual', 0.0):>17.2%}")
    lines.append(f"{'Tracking Error':<32} {'N/A':>16} {spy.get('tracking_error', 0.0):>15.2%} {rsp.get('tracking_error', 0.0):>17.2%}")
    lines.append("-" * 86)

    lines.append(f"{'Cumulative Wealth Multiple':<32} {strat.get('cumulative_multiple_net', 1.0):>15.2f}x {'N/A':>16} {'N/A':>18}")
    lines.append(f"{'Annualized Turnover':<32} {strat.get('annualized_turnover', 0.0):>15.2%} {'N/A':>16} {'N/A':>18}")
    lines.append(f"{'Total Incurred Friction Costs':<32} {strat.get('total_transaction_friction_cost', 0.0):>15.2%} {'N/A':>16} {'N/A':>18}")
    lines.append("=" * 86)
    lines.append("* Note: RSP equal-weighted comparison is isolated starting May 1, 2003.")
    lines.append("=" * 86)

    return "\n".join(lines)


def run_backtest(
    prices: pd.DataFrame,
    mask: pd.DataFrame,
    fee_bps: float = 10.0,
    rf_annual: float = 0.015,
    min_diversification: int = 3,
    export_results: bool = True
) -> pd.DataFrame:
    """Executes full vectorized point-in-time backtest and generates daily performance curves."""
    returns = compute_returns(prices)
    common_idx = prices.index.intersection(mask.index)
    prices = prices.loc[common_idx]
    mask = mask.loc[common_idx]
    returns = returns.loc[common_idx]

    rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0
    stock_weights, cash_weights, n_t = compute_equal_weights(mask, min_diversification=min_diversification)

    portfolio_gross = pd.Series(0.0, index=common_idx)
    drifted_weights = pd.DataFrame(0.0, index=common_idx, columns=stock_weights.columns)

    for i, date in enumerate(common_idx):
        day_w = stock_weights.loc[date]
        day_r = returns.loc[date, stock_weights.columns]
        cash_w = cash_weights.loc[date]
        gross_ret = float(np.sum(day_w * day_r) + cash_w * rf_daily)
        portfolio_gross.iloc[i] = gross_ret

        denom = 1.0 + gross_ret
        drifted_w = (day_w * (1.0 + day_r)) / (denom if denom != 0.0 else 1.0)
        drifted_weights.iloc[i] = drifted_w

    turnover, costs = compute_turnover_and_costs(stock_weights, drifted_weights, fee_bps=fee_bps)
    portfolio_net = portfolio_gross - costs

    net_cum = (1.0 + portfolio_net).cumprod()
    strat_peak = net_cum.cummax()
    strat_dd = (net_cum - strat_peak) / strat_peak

    spy_ret = returns["SPY"] if "SPY" in returns.columns else pd.Series(0.0, index=common_idx)
    spy_cum = (1.0 + spy_ret).cumprod()
    spy_peak = spy_cum.cummax()
    spy_dd = (spy_cum - spy_peak) / spy_peak

    rsp_ret = returns["RSP"] if "RSP" in returns.columns else pd.Series(0.0, index=common_idx)
    rsp_cum = (1.0 + rsp_ret).cumprod()
    rsp_peak = rsp_cum.cummax()
    rsp_dd = (rsp_cum - rsp_peak) / rsp_peak

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


if __name__ == "__main__":
    if not PRICES_PATH.exists() or not MASK_PATH.exists():
        logging.error(f"Missing input artifacts: {PRICES_PATH} or {MASK_PATH}")
    else:
        prices_df = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
        mask_df = pd.read_csv(MASK_PATH, index_col=0, parse_dates=True).astype(bool)
        results = run_backtest(prices_df, mask_df, export_results=True)
        summary_dict = build_summary_metrics_dict(results)
        print("\n" + format_scorecard_table(summary_dict) + "\n")
