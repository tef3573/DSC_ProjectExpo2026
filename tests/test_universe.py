"""
tests/test_universe.py
Unit tests for point-in-time universe masking, execution lag, and constituent limits.
Sub-issue #78: Verify active constituent count N_t never exceeds roster length.
"""
from pathlib import Path
import unittest
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ROSTER_PATH = DATA_DIR / "black_ceo_roster.csv"
PRICES_PATH = DATA_DIR / "prices_daily.csv"
MASK_PATH = DATA_DIR / "universe_mask.csv"

class TestUniverseMask(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
        cls.mask = pd.read_csv(MASK_PATH, index_col=0, parse_dates=True)
        cls.roster = pd.read_csv(ROSTER_PATH)
        cls.roster.columns = cls.roster.columns.astype(str).str.strip()
        cls.unique_tickers = cls.roster["ticker"].dropna().astype(str).str.strip().unique().tolist()

    def test_universe_mask_dimensions_and_calendar(self):
        """Verify mask index matches SPY trading calendar exactly."""
        self.assertEqual(len(self.mask), len(self.prices), "Universe mask row count must match prices_daily")
        self.assertTrue((self.mask.index == self.prices.index).all(), "Mask calendar index must strictly match prices")
        self.assertNotIn("SPY", self.mask.columns, "SPY benchmark must be excluded from stock universe mask")
        self.assertNotIn("RSP", self.mask.columns, "RSP benchmark must be excluded from stock universe mask")

    def test_active_constituent_count_bounds(self):
        """Sub-issue #78: Verify active constituent count N_t never exceeds roster length."""
        active_counts = self.mask.sum(axis=1)
        max_possible = len(self.unique_tickers)

        self.assertTrue((active_counts >= 0).all(), "Active count Nt cannot be negative")
        self.assertTrue((active_counts <= max_possible).all(), f"Nt cannot exceed total roster size ({max_possible})")
        self.assertLessEqual(active_counts.max(), max_possible, f"Max active Nt exceeded roster length: {active_counts.max()}")

    def test_delisted_jcp_tenure_bounds(self):
        """Verify point-in-time accuracy for Marvin Ellison at JCP (Aug 2015 to June 2018)."""
        if "JCP" in self.mask.columns:
            jcp_series = self.mask["JCP"]
            # Before Ellison tenure (e.g. 2014) must be False
            pre_tenure = jcp_series.loc["2014-01-01":"2015-07-31"]
            self.assertFalse(pre_tenure.any(), "JCP must not be active prior to Marvin Ellison appointment")

            # Active tenure (e.g. 2016) must contain active True sessions
            active_tenure = jcp_series.loc["2016-01-01":"2016-12-31"]
            self.assertTrue(active_tenure.any(), "JCP must be active during 2016 Ellison tenure")

            # After exit (e.g. 2019) must be False
            post_tenure = jcp_series.loc["2019-01-01":]
            self.assertFalse(post_tenure.any(), "JCP must not be active after Ellison departure")


if __name__ == "__main__":
    unittest.main()