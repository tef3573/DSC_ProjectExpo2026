"""
src/data_loader.py
Fast batch downloader for Black CEO portfolio constituents and market benchmarks.
Integrates local delisted archives (JCP) to eliminate survivorship bias.
"""
from pathlib import Path
import logging
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DELISTED_DIR = DATA_DIR / "delisted"
ROSTER_PATH = DATA_DIR / "black_ceo_roster.csv"
OUTPUT_PRICES_PATH = DATA_DIR / "prices_daily.csv"
START_DATE = "2000-01-01"


def load_universe_roster() -> pd.DataFrame:
    df = pd.read_csv(ROSTER_PATH)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"].fillna("2026-10-27"))
    return df


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


def build_price_matrix():
    roster_df = load_universe_roster()
    roster_tickers = sorted(roster_df["ticker"].unique().tolist())

    # Map corporate action rename (QRTEA -> QVCG)
    download_tickers = [("QVCG" if t == "QRTEA" else t) for t in roster_tickers]
    all_download_symbols = sorted(list(set(download_tickers + ["SPY", "RSP"])))

    logging.info(f"Batch downloading market price histories for {len(all_download_symbols)} symbols...")
    raw = yf.download(
        tickers=all_download_symbols,
        start=START_DATE,
        auto_adjust=False,
        progress=True,
        group_by="column"
    )

    # Extract Adjusted Close price matrix
    if "Adj Close" in raw.columns:
        prices = raw["Adj Close"].copy()
    elif "Close" in raw.columns:
        prices = raw["Close"].copy()
    else:
        prices = raw.copy()

    # Re-map QVCG column back to QRTEA for roster alignment if downloaded
    if "QVCG" in prices.columns:
        prices["QRTEA"] = prices["QVCG"]

    # Integrate local delisted archives (JCP, PBPB)
    for ticker in roster_tickers:
        if ticker not in prices.columns or prices[ticker].dropna().empty:
            delisted_series = load_delisted_asset(ticker)
            if not delisted_series.empty:
                prices[ticker] = delisted_series

    # Reindex against SPY trading calendar
    if "SPY" in prices.columns:
        spy_calendar = prices["SPY"].dropna().index
        prices = prices.reindex(spy_calendar)

    # Forward fill small calendar disparities, then export
    prices = prices.ffill()

    OUTPUT_PRICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(OUTPUT_PRICES_PATH)
    logging.info(f"[SUCCESS] Exported master price matrix: {prices.shape} to {OUTPUT_PRICES_PATH}")
    return prices


if __name__ == "__main__":
    build_price_matrix()
