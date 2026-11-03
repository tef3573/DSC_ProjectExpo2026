"""
src/backtest.py
Vectorized Portfolio Backtesting Engine.
- Sub-issue #100: Proportional equity scaling when Nt < 3.
- Sub-issue #101: Dynamic routing of unallocated cash buffer into risk-free rate.
- Sub-issue #102: Vectorized active count tracking across timeline.
"""
from pathlib import Path
import logging
from typing import Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PRICES_PATH = DATA_DIR / "prices_daily.csv"
MASK_PATH = DATA_DIR / "universe_mask.csv"
OUTPUT_RETURNS_PATH = DATA_DIR / "portfolio_daily_returns.csv"


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads clean price matrix and lagged universe mask."""
    if not PRICES_PATH.exists():
        raise FileNotFoundError(f"Prices daily matrix missing at {PRICES_PATH}")
    if not MASK_PATH.exists():
        raise FileNotFoundError(f"Universe mask missing at {MASK_PATH}")

    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    mask = pd.read_csv(MASK_PATH, index_col=0, parse_dates=True)
    return prices, mask


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Computes daily simple arithmetic returns with numerical safety checks."""
    returns = prices.pct_change(fill_method=None)
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return returns


def compute_equal_weights(mask: pd.DataFrame, min_diversification: int = 3) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Sub-issues #100, #101, #102:
    Computes daily constituent weights with cash-drag concentration defense.
    - If Nt >= 3: Fully invested equal-weight (w_i = 1 / Nt, w_cash = 0.0).
    - If 1 <= Nt < 3: Proportional cap at 1 / min_diversification (w_i = 1/3, w_cash = (3 - Nt)/3).
    - If Nt == 0: 100% cash allocation (w_i = 0.0, w_cash = 1.0).
    """
    # Sub-issue #102: Vectorized constituent count across entire calendar
    n_t = mask.sum(axis=1)

    weights = pd.DataFrame(0.0, index=mask.index, columns=mask.columns)

    # Condition 1: Nt >= min_diversification (Standard Equal Weight)
    normal_mask = n_t >= min_diversification
    if normal_mask.any():
        weights.loc[normal_mask] = mask.loc[normal_mask].div(n_t.loc[normal_mask], axis=0)

    # Condition 2: 1 <= Nt < min_diversification (Concentration Defense)
    defensive_mask = (n_t > 0) & (n_t < min_diversification)
    if defensive_mask.any():
        # Cap each constituent at 1 / min_diversification (e.g. 1/3)
        weights.loc[defensive_mask] = mask.loc[defensive_mask].astype(float) * (1.0 / float(min_diversification))

    # Condition 3: Nt == 0 remains 0.0 for all stocks

    # Sub-issue #101: Cash buffer captures 1.0 - sum(stock_weights)
    stock_alloc = weights.sum(axis=1)
    cash_weights = (1.0 - stock_alloc).clip(lower=0.0, upper=1.0)

    return weights, cash_weights, n_t


def run_backtest(
    prices: pd.DataFrame, 
    mask: pd.DataFrame, 
    rf_annual: float = 0.015
) -> pd.DataFrame:
    """
    Simulates portfolio returns with cash drag and risk-free return attribution.
    """
    all_returns = compute_returns(prices)
    constituent_cols = [c for c in mask.columns if c in all_returns.columns]

    stock_returns = all_returns[constituent_cols]
    stock_weights, cash_weights, n_t = compute_equal_weights(mask[constituent_cols])

    # Convert annual risk-free rate to daily compounding rate (252 sessions)
    rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0

    # Sub-issue #101: Returns = equity dot product + cash buffer return
    portfolio_stock_returns = (stock_weights * stock_returns).sum(axis=1)
    portfolio_cash_returns = cash_weights * rf_daily
    total_portfolio_returns = portfolio_stock_returns + portfolio_cash_returns

    perf = pd.DataFrame(index=prices.index)
    perf["Black_CEO_EW"] = total_portfolio_returns
    perf["SPY"] = all_returns["SPY"] if "SPY" in all_returns.columns else 0.0
    perf["RSP"] = all_returns["RSP"] if "RSP" in all_returns.columns else 0.0
    perf["Active_Constituents"] = n_t
    perf["Cash_Weight"] = cash_weights
    perf["Stock_Weight"] = stock_weights.sum(axis=1)

    # Cumulative wealth series (normalized to $1.00 base)
    perf["Black_CEO_Cumulative"] = (1.0 + perf["Black_CEO_EW"]).cumprod()
    perf["SPY_Cumulative"] = (1.0 + perf["SPY"]).cumprod()
    perf["RSP_Cumulative"] = (1.0 + perf["RSP"]).cumprod()

    logging.info(
        f"Backtest completed ({len(perf)} sessions). "
        f"Cumulative Multiples: Strategy={perf['Black_CEO_Cumulative'].iloc[-1]:.3f}x | "
        f"SPY={perf['SPY_Cumulative'].iloc[-1]:.3f}x | RSP={perf['RSP_Cumulative'].iloc[-1]:.3f}x"
    )
    return perf


def export_results(perf: pd.DataFrame, path: Path = OUTPUT_RETURNS_PATH) -> Path:
    """Exports daily returns and wealth series to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    perf.to_csv(path, float_format="%.6f")
    logging.info(f"Saved portfolio daily returns to: {path}")
    return path


if __name__ == "__main__":
    df_prices, df_mask = load_inputs()
    perf_results = run_backtest(df_prices, df_mask, rf_annual=0.015)
    export_results(perf_results)

    print("\n--- ISSUE #88: CASH-DRAG DEFENSE VALIDATION ---")
    print(f"Total Sessions: {len(perf_results)}")
    print(f"Sessions with Nt < 3: {(perf_results['Active_Constituents'] < 3).sum()}")
    print(f"Average Cash Weight during Nt < 3: {perf_results.loc[perf_results['Active_Constituents'] < 3, 'Cash_Weight'].mean():.2%}")
    print("Final Growth Multiples:")
    print(f"  Black CEO Strategy: {perf_results['Black_CEO_Cumulative'].iloc[-1]:.3f}x")
    print(f"  SPY Benchmark:      {perf_results['SPY_Cumulative'].iloc[-1]:.3f}x")
    print(f"  RSP Benchmark:      {perf_results['RSP_Cumulative'].iloc[-1]:.3f}x")
