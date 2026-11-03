"""
src/backtest.py
Base Vectorized Portfolio Backtesting Engine.
- Issue #87 / Sub-issue #94: Daily 1/Nt asset return dot product.
- Issue #87 / Sub-issue #95: Dynamic cash drag allocation when Nt == 0.
- Issue #87 / Sub-issue #96: Division-by-zero protections during holidays and inactive periods.
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

# Loads clean price matrix and lagged universe mask.
def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not PRICES_PATH.exists():
        raise FileNotFoundError(f"Prices daily matrix missing at {PRICES_PATH}")
    if not MASK_PATH.exists():
        raise FileNotFoundError(f"Universe mask missing at {MASK_PATH}")

    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    mask = pd.read_csv(MASK_PATH, index_col=0, parse_dates=True)
    return prices, mask

 """
    Computes daily arithmetic simple returns: r_{i,t} = (P_{i,t} / P_{i,t-1}) - 1.
    Handles exchange halts and zero-volume days cleanly without division-by-zero errors.
"""
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change(fill_method=None)
    # Replace inf and -inf values with NaN, then fill single-day gaps with 0.0
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return returns

def compute_equal_weights(mask: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    return stock_weights, cash_weights, n_t


def run_backtest(
    prices: pd.DataFrame, 
    mask: pd.DataFrame, 
    rf_daily: float = 0.0
) -> pd.DataFrame:
    return 

# Exports daily returns and wealth series to CSV.
def export_results(perf: pd.DataFrame, path: Path = OUTPUT_RETURNS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    perf.to_csv(path, float_format="%.6f")
    logging.info(f"Saved portfolio daily returns to: {path}")
    return path

# Print
if __name__ == "__main__":
    df_prices, df_mask = load_inputs()
    perf_results = run_backtest(df_prices, df_mask, rf_daily=0.0)
    export_results(perf_results)

    print("\n--- ISSUE #87 ENGINE VALIDATION ---")
    print(f"Total Sessions: {len(perf_results)}")
    print(f"Zero Active Sessions (Cash Held): {(perf_results['Active_Constituents'] == 0).sum()}")
    print("Final Growth Multiples:")
    print(f"  Black CEO Strategy: {perf_results['Black_CEO_Cumulative'].iloc[-1]:.3f}x")
    print(f"  SPY Benchmark:      {perf_results['SPY_Cumulative'].iloc[-1]:.3f}x")
    print(f"  RSP Benchmark:      {perf_results['RSP_Cumulative'].iloc[-1]:.3f}x")