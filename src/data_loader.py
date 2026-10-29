"""
src/data_loader.py
Downloads split- and dividend-adjusted closing prices from Yahoo Finance
and integrates local archives for delisted constituents (e.g., JCP).
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


def load_universe_roster() -> pd.DataFrame:
    if not ROSTER_PATH.exists():
        raise FileNotFoundError(f"Roster missing: {ROSTER_PATH}")
    df = pd.read_csv(ROSTER_PATH)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"].fillna("2026-10-27"))
    return df


def fetch_ticker_data(ticker: str, start: str = "2000-01-01", end: str = "2026-10-27") -> pd.Series:
    """Retrieves adjusted daily closing price series for a single ticker."""
    delisted_candidates = [
        DELISTED_DIR / f"{ticker}.csv",
        DELISTED_DIR / f"{ticker}.CSV",
        DELISTED_DIR / f"{ticker.upper()}.csv",
    ]
    local_csv = next((p for p in delisted_candidates if p.exists()), None)

    if local_csv:
        logging.info(f"Loading {ticker} from local delisted archive: {local_csv}")
        df_delisted = pd.read_csv(local_csv)
        df_delisted.columns = df_delisted.columns.str.strip()

        date_col = next((c for c in df_delisted.columns if "date" in c.lower()), df_delisted.columns[0])
        df_delisted[date_col] = pd.to_datetime(df_delisted[date_col], errors="coerce")
        df_delisted = df_delisted.dropna(subset=[date_col]).set_index(date_col).sort_index()

        price_cols = [
            c for c in df_delisted.columns 
            if any(k in c.lower() for k in ["adj close", "close", "price"]) 
            and "%" not in c and "change" not in c.lower()
        ]
        col = price_cols[0] if price_cols else df_delisted.columns[0]

        clean_series = pd.to_numeric(
            df_delisted[col].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip(),
            errors="coerce"
        ).dropna()
        clean_series.name = ticker
        return clean_series

    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if raw.empty or "Adj Close" not in raw:
            logging.warning(f"No price history found for {ticker}")
            return pd.Series(dtype=float, name=ticker)

        series = raw["Adj Close"].squeeze()
        series.name = ticker
        return series
    except Exception as e:
        logging.error(f"Error downloading {ticker}: {e}")
        return pd.Series(dtype=float, name=ticker)


def build_price_matrix():
    roster_df = load_universe_roster()
    unique_tickers = sorted(roster_df["ticker"].unique().tolist())
    all_symbols = list(dict.fromkeys(unique_tickers + ["SPY", "RSP"]))

    logging.info(f"Downloading/loading price series for {len(all_symbols)} symbols...")
    price_dict = {}
    failed = []

    for sym in all_symbols:
        s = fetch_ticker_data(sym)
        if s.dropna().empty:
            failed.append(sym)
        else:
            price_dict[sym] = s

    if failed:
        logging.warning(f"Could not load prices for: {failed}")

    prices_df = pd.DataFrame(price_dict).sort_index()

    if "SPY" in prices_df.columns:
        spy_calendar = prices_df["SPY"].dropna().index
        prices_df = prices_df.reindex(spy_calendar)

    OUTPUT_PRICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    prices_df.to_csv(OUTPUT_PRICES_PATH)
    logging.info(f"[SUCCESS] Saved price matrix: {prices_df.shape} to {OUTPUT_PRICES_PATH}")
    return prices_df


if __name__ == "__main__":
    build_price_matrix()
