"""
src/data_loader.py
Automated market data ingestion pipeline for the Black CEO portfolio universe.
- Sub-task #65: Batch download engine for constituents and benchmarks.
- Sub-task #66: QVCG corporate action fallback resolution for QRTEA.
- Local delisted archive integration (JCP).
- Sub-task #67: Reindex prices to SPY trading calendar and forward-fill disparities.
"""
from pathlib import Path
import logging
from typing import List
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DELISTED_DIR = DATA_DIR / "delisted"
ROSTER_PATH = DATA_DIR / "black_ceo_roster.csv"
OUTPUT_PRICES_PATH = DATA_DIR / "prices_daily.csv"
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


def load_delisted_asset(ticker: str) -> pd.Series:
    """Loads a locally archived delisted asset CSV if available."""
    candidates = [
        DELISTED_DIR / f"{ticker}.csv",
        DELISTED_DIR / f"{ticker}.CSV",
        DELISTED_DIR / f"{ticker.upper()}.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if not path:
        return pd.Series(dtype=float, name=ticker)

    logging.info(f"Loading {ticker} from local delisted archive: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    price_cols = [
        c for c in df.columns
        if any(k in c.lower() for k in ["adj close", "close", "price"])
        and "%" not in c and "change" not in c.lower()
    ]
    col = price_cols[0] if price_cols else df.columns[0]
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    series.name = ticker
    return series


def resolve_corporate_actions(prices: pd.DataFrame) -> pd.DataFrame:
    """Sub-task #66: Merges post-split QVCG prices into QRTEA for continuous history."""
    df = prices.copy()
    if "QVCG" in df.columns:
        if "QRTEA" in df.columns:
            df["QRTEA"] = df["QVCG"].combine_first(df["QRTEA"])
        else:
            df["QRTEA"] = df["QVCG"]
        df = df.drop(columns=["QVCG"])
        logging.info("Successfully resolved corporate action: merged QVCG into QRTEA.")
    return df


def align_to_trading_calendar(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Sub-task #67: Reindexes price matrix to the SPY calendar and forward-fills.
    Ensures every asset conforms to official US equity market trading dates.
    """
    if "SPY" not in prices.columns:
        raise ValueError("SPY benchmark missing from price matrix; cannot calibrate calendar.")

    df = prices.copy()
    spy_calendar = df["SPY"].dropna().index.sort_values()

    # Reindex entire universe to the official SPY trading sessions
    df = df.reindex(spy_calendar)

    # Forward-fill up to 5 consecutive sessions for holiday/reporting disparities
    df = df.ffill(limit=5)
    logging.info(f"Successfully aligned prices to SPY calendar: {len(df)} market sessions.")
    return df


def batch_download_adjusted_prices(tickers: List[str], start_date: str = START_DATE) -> pd.DataFrame:
    """Downloads daily adjusted prices, merges corporate fallbacks, and aligns to calendar."""
    clean_tickers = [t.strip().upper() for t in tickers]
    download_symbols = list(clean_tickers)

    if "QRTEA" in download_symbols and "QVCG" not in download_symbols:
        download_symbols.append("QVCG")
    elif "QVCG" not in download_symbols:
        download_symbols.append("QVCG")

    all_symbols = sorted(list(set(download_symbols + ["SPY", "RSP"])))
    logging.info(f"Initiating batch download for {len(all_symbols)} symbols...")

    raw = yf.download(
        tickers=all_symbols,
        start=start_date,
        auto_adjust=False,
        progress=True,
        group_by="column",
    )

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

    # 1. Resolve corporate action mapping (QVCG -> QRTEA) (#66)
    prices = resolve_corporate_actions(prices)

    # 2. Integrate local delisted archives (JCP)
    for t in clean_tickers:
        if t not in prices.columns or prices[t].dropna().empty:
            delisted_series = load_delisted_asset(t)
            if not delisted_series.empty:
                prices[t] = delisted_series

    # 3. Calendar Alignment & Forward-Fill (#67)
    prices = align_to_trading_calendar(prices)

    return prices


if __name__ == "__main__":
    roster_symbols = load_roster_tickers()
    df_prices = batch_download_adjusted_prices(roster_symbols)

    print("\n--- SUB-TASK #67 VALIDATION ---")
    print(f"Total Calendar Trading Sessions (SPY aligned): {len(df_prices)}")
    print(f"Index is strictly monotonic increasing: {df_prices.index.is_monotonic_increasing}")
    print("Sample aligned rows:")
    print(df_prices[["SPY", "RSP", "AXP", "QRTEA", "JCP"]].tail(3))
