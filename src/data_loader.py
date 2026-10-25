"""
Data Ingestion Pipeline for Black CEO Portfolio Backtest
--------------------------------------------------------
This module loads the curated historical universe roster, pulls daily 
adjusted close prices for active constituents and benchmarks via yfinance, 
validates data integrity, and saves the cleaned price matrix locally.
"""

import os
import pandas as pd  # type: ignore[import-unresolved]
import yfinance as yf  # type: ignore[import-unresolved]

# Configuration paths and default parameters
ROSTER_PATH = "data/black_ceo_roster.csv"
OUTPUT_PRICES_PATH = "data/prices_daily.csv"
BENCHMARKS = ["SPY", "RSP"]  # SPY: Cap-Weighted S&P 500, RSP: Equal-Weighted S&P 500
START_DATE = "2000-01-01"


def load_universe_roster(path: str = ROSTER_PATH) -> pd.DataFrame:
    """
    Loads and standardizes the executive tenure roster.
    
    Parameters:
        path (str): Relative or absolute path to the roster CSV.
        
    Returns:
        pd.DataFrame: Cleaned dataframe with parsed datetime columns.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Roster file not found at: {path}")

    # Read roster and parse tenure boundaries as datetime objects
    df = pd.read_csv(path, parse_dates=["start_date", "end_date"])

    # Clean ticker symbols: strip any accidental whitespace and standardize uppercase
    df["ticker"] = df["ticker"].str.strip().str.upper()

    print(f"[INFO] Successfully loaded {len(df)} tenure records from {path}.")
    return df


def fetch_historical_prices(
    tickers: list[str], 
    benchmarks: list[str] = BENCHMARKS, 
    start_date: str = START_DATE
) -> pd.DataFrame:
    """
    Downloads historical adjusted closing prices from Yahoo Finance.
    Handles multi-index formatting changes in modern yfinance versions.
    
    Parameters:
        tickers (list[str]): List of stock tickers representing company tenures.
        benchmarks (list[str]): Benchmark symbols (e.g., SPY, RSP).
        start_date (str): Backtest evaluation start date (YYYY-MM-DD).
        
    Returns:
        pd.DataFrame: Daily adjusted close price series for all valid tickers.
    """
    # Combine universe tickers and benchmarks, removing duplicates
    all_symbols = sorted(list(set(tickers + benchmarks)))
    print(f"[INFO] Requesting price history for {len(all_symbols)} tickers starting from {start_date}...")

    # Download batch price history via yfinance
    # auto_adjust=False preserves the explicit 'Adj Close' column for total return adjustments
    raw_data = yf.download(
        tickers=all_symbols,
        start=start_date,
        auto_adjust=False,
        progress=True
    )

    # Inspect and normalize column structure:
    # Modern yfinance returns a MultiIndex DataFrame where level 0 is price field ('Adj Close', 'Close', etc.)
    if "Adj Close" in raw_data.columns:
        prices = raw_data["Adj Close"].copy()
    elif "Close" in raw_data.columns:
        print("[WARN] 'Adj Close' not found. Falling back to unadjusted 'Close' prices.")
        prices = raw_data["Close"].copy()
    else:
        prices = raw_data.copy()

    # Data Quality Check: Detect symbols that returned only NaNs (e.g., delisted companies, ticker renames)
    valid_tickers = prices.dropna(how="all", axis=1).columns.tolist()
    missing_tickers = set(all_symbols) - set(valid_tickers)

    if missing_tickers:
        print(f"[WARN] No market price data found for {len(missing_tickers)} tickers: {sorted(list(missing_tickers))}")
        print("[WARN] Note: Historical acquisitions/delistings may require manual corporate action mapping.")

    # Retain only columns that contain actual price observations
    cleaned_prices = prices[valid_tickers]

    # Forward-fill minor holiday/trading day calendar gaps, then drop initial empty rows
    cleaned_prices = cleaned_prices.ffill()

    return cleaned_prices


def run_pipeline():
    """
    Executes the ingestion pipeline and persists the price matrix to disk.
    """
    # Step 1: Read the executive tenure dataset
    roster_df = load_universe_roster(ROSTER_PATH)
    unique_tickers = roster_df["ticker"].unique().tolist()

    # Step 2: Fetch price histories for tickers and benchmarks
    price_matrix = fetch_historical_prices(unique_tickers)

    # Step 3: Ensure destination directory exists and save local CSV cache
    os.makedirs(os.path.dirname(OUTPUT_PRICES_PATH), exist_ok=True)
    price_matrix.to_csv(OUTPUT_PRICES_PATH)

    print(f"[SUCCESS] Price matrix cached to {OUTPUT_PRICES_PATH}")
    print(f"[SUMMARY] Total Trading Days: {len(price_matrix)} | Active Tickers Saved: {price_matrix.shape[1]}")


if __name__ == "__main__":
    run_pipeline()