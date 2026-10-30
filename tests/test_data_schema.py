"""
tests/test_data_schema.py
Unit tests to validate the integrity, schema, and point-in-time date consistency
of the executive roster and delisted price archives.
"""
from pathlib import Path
import unittest
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ROSTER_PATH = DATA_DIR / "black_ceo_roster.csv"
JCP_PATH = DATA_DIR / "delisted" / "JCP.csv"


class TestDataSchema(unittest.TestCase):

    def test_roster_file_and_columns(self):
        """Verify black_ceo_roster.csv exists and matches expected schema."""
        self.assertTrue(ROSTER_PATH.exists(), f"Roster file missing at {ROSTER_PATH}")
        df = pd.read_csv(ROSTER_PATH)
        df.columns = df.columns.str.strip()

        required_cols = {"ticker", "company_name", "ceo_name", "start_date", "end_date", "sector"}
        self.assertTrue(
            required_cols.issubset(set(df.columns)),
            f"Missing required columns. Found: {list(df.columns)}"
        )
        self.assertGreaterEqual(len(df), 20, f"Expected at least 20 roster entries, found {len(df)}")

    def test_roster_date_consistency(self):
        """Verify all start dates precede end dates and parse as valid timestamps."""
        df = pd.read_csv(ROSTER_PATH)
        df.columns = df.columns.str.strip()

        start_dates = pd.to_datetime(df["start_date"], errors="coerce")
        end_dates = pd.to_datetime(df["end_date"].fillna("2026-10-27"), errors="coerce")

        self.assertFalse(start_dates.isna().any(), "Found invalid or unparseable start_date entries")
        self.assertFalse(end_dates.isna().any(), "Found invalid or unparseable end_date entries")
        self.assertTrue((start_dates <= end_dates).all(), "start_date must be earlier than or equal to end_date")

    def test_delisted_jcp_schema(self):
        """Verify data/delisted/JCP.csv contains clean dates and positive closing prices."""
        self.assertTrue(JCP_PATH.exists(), f"Delisted JCP archive missing at {JCP_PATH}")
        df = pd.read_csv(JCP_PATH)
        df.columns = df.columns.str.strip()

        self.assertIn("Date", df.columns, "Date column missing from JCP.csv")
        self.assertIn("Adj Close", df.columns, "Adj Close column missing from JCP.csv")

        dates = pd.to_datetime(df["Date"], errors="coerce")
        prices = pd.to_numeric(df["Adj Close"], errors="coerce")

        self.assertFalse(dates.isna().any(), "Found invalid Date entries in JCP.csv")
        self.assertFalse(prices.isna().any(), "Found non-numeric Adj Close values in JCP.csv")
        self.assertTrue((prices > 0).all(), "All closing prices must be strictly positive")
        self.assertGreaterEqual(len(df), 700, f"Expected at least 700 NYSE trading days for JCP, found {len(df)}")


if __name__ == "__main__":
    unittest.main()
