"""
src/data_loader.py
Automated market data ingestion pipeline for the Black CEO portfolio universe.
- Sub-task 2.1: Batch download engine for constituents and benchmarks.
- Sub-task 2.2: QVCG corporate action fallback resolution for QRTEA.
"""
from pathlib import Path
import logging
from typing import List
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ROSTER_PATH = DATA_DIR / "black_ceo_roster.csv"
START_DATE = "2000-01-01"


def load_roster_tickers() -> List[str]:
    """Loads and strips unique constituent ticker symbols from the audited roster."""
    if not ROSTER_PATH.exists():
        raise FileNotFoundError(f"Roster not found at {ROSTER_PATH}")
    df = pd.read_csv(ROSTER_PATH)
    df.columns = df.columns.str.strip()
    tickers = sorted(df["ticker"].dropna().astype(str).str.strip().unique().tolist())
    logging.info(f"Loaded {len(tickers)} unique tickers from roster: {tickers}")
    return tickers


def resolve_corporate_actions(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Sub-task 2.2: Merges post-split QVCG prices into QRTEA for continuous history.
    """
    df = prices.copy()
    if "QVCG" in df.columns:
        if "QRTEA" in df.columns:
            df["QRTEA"] = df["QVCG"].combine_first(df["QRTEA"])
        else:
            df["QRTEA"] = df["QVCG"]
        df = df.drop(columns=["QVCG"])
        logging.info("Successfully resolved corporate action: merged QVCG into QRTEA.")
    else:
        logging.warning("QVCG was not found in downloaded columns.")
    return df


def batch_download_adjusted_prices(tickers: List[str], start_date: str = START_DATE) -> pd.DataFrame:
    """
    Downloads daily split/dividend-adjusted close prices.
    Includes SPY, RSP, and explicitly ensures QVCG is downloaded for QRTEA.
    """
    clean_tickers = [t.strip().upper() for t in tickers]
    download_symbols = list(clean_tickers)

    # Explicitly ensure QVCG is added for QRTEA
    if "QRTEA" in download_symbols and "QVCG" not in download_symbols:
        download_symbols.append("QVCG")
    elif "QVCG" not in download_symbols:
        download_symbols.append("QVCG")

    all_symbols = sorted(list(set(download_symbols + ["SPY", "RSP"])))
    logging.info(f"Initiating batch download for {len(all_symbols)} symbols: {all_symbols}")

    raw = yf.download(
        tickers=all_symbols,
        start=start_date,
        auto_adjust=False,
        progress=True,
        group_by="column",
    )

    # Extract Adj Close prices
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.levels[0]:
            prices = raw["Adj Close"].copy()
        elif "Close" in raw.columns.levels[0]:
            prices = raw["Close"].copy()
        else:
            prices = raw.copy()
    else:
        if "Adj Close" in raw.columns:
            prices = raw["Adj Close"].copy()
        elif "Close" in raw.columns:
            prices = raw["Close"].copy()
        else:
            prices = raw.copy()

    if hasattr(prices.index, "tz") and prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)

    # Resolve corporate action mapping (QVCG -> QRTEA)
    prices = resolve_corporate_actions(prices)

    logging.info(f"Price matrix shape after corporate action resolution: {prices.shape}")
    return prices


if __name__ == "__main__":
    roster_symbols = load_roster_tickers()
    df_prices = batch_download_adjusted_prices(roster_symbols)

    print("\n--- SUB-TASK 2.2 VALIDATION ---")
    if "QRTEA" in df_prices.columns:
        valid_qrtea = df_prices["QRTEA"].dropna()
        print(f"QRTEA Sessions: {len(valid_qrtea)}")
        print("Latest QRTEA (via QVCG) prices:")
        print(valid_qrtea.tail(5))
    else:
        print("QRTEA column missing from output.")
