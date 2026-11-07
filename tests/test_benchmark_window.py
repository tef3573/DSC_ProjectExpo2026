"""
tests/test_benchmark_window.py
Unit tests verifying benchmark window decoupling and alignment.
- Sub-issue #120: Align SPY cap-weighted comparison across full 2000-2026 horizon.
- Sub-issue #121: Isolate RSP equal-weighted comparison starting May 1, 2003.
- Sub-issue #122: Filter pre-inception zero-return entries from RSP metrics.
"""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from src.metrics import (
    align_benchmark_series,
    isolate_rsp_benchmark_series,
    compute_benchmark_metrics,
    RSP_INCEPTION_DATE
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PORTFOLIO_RETURNS_PATH = DATA_DIR / "portfolio_daily_returns.csv"


class TestBenchmarkWindowDecoupling(unittest.TestCase):

    def test_spy_full_horizon_alignment(self):
        """Sub-issue #120: Verify SPY alignment captures full simulation sessions without truncation."""
        if not PORTFOLIO_RETURNS_PATH.exists():
            self.skipTest(f"Missing {PORTFOLIO_RETURNS_PATH}")

        df = pd.read_csv(PORTFOLIO_RETURNS_PATH, index_col=0, parse_dates=True)
        self.assertIn("Black_CEO_Net", df.columns)
        self.assertIn("SPY", df.columns)

        strat, spy = align_benchmark_series(df["Black_CEO_Net"], df["SPY"])
        self.assertEqual(len(strat), len(df))
        self.assertEqual(len(spy), len(df))
        self.assertTrue((strat.index == spy.index).all())

        start_year = strat.index[0].year
        end_year = strat.index[-1].year
        self.assertEqual(start_year, 2000)
        self.assertGreaterEqual(end_year, 2026)

    def test_spy_benchmark_metrics_calculation(self):
        """Sub-issue #120: Assert compute_benchmark_metrics outputs valid alpha, beta, and tracking error."""
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        np.random.seed(42)
        spy_vals = np.random.normal(loc=0.0005, scale=0.01, size=252)
        spy_ret = pd.Series(spy_vals, index=dates)

        strat_ret = spy_ret * 1.2
        metrics = compute_benchmark_metrics(strat_ret, spy_ret, rf_annual=0.0)

        self.assertEqual(metrics["observations"], 252)
        self.assertAlmostEqual(metrics["beta"], 1.2, places=4)
        self.assertAlmostEqual(metrics["alpha_annual"], 0.0, places=4)
        expected_te = (strat_ret - spy_ret).std(ddof=1) * np.sqrt(252)
        self.assertAlmostEqual(metrics["tracking_error"], expected_te, places=5)

    def test_rsp_isolation_starts_may_first_2003(self):
        """Sub-issue #121: Verify RSP benchmark alignment strictly starts on or after May 1, 2003."""
        if not PORTFOLIO_RETURNS_PATH.exists():
            self.skipTest(f"Missing {PORTFOLIO_RETURNS_PATH}")

        df = pd.read_csv(PORTFOLIO_RETURNS_PATH, index_col=0, parse_dates=True)
        self.assertIn("RSP", df.columns)

        strat_rsp, rsp = isolate_rsp_benchmark_series(df["Black_CEO_Net"], df["RSP"])
        self.assertGreaterEqual(rsp.index[0], pd.to_datetime(RSP_INCEPTION_DATE))
        self.assertGreaterEqual(strat_rsp.index[0], pd.to_datetime(RSP_INCEPTION_DATE))
        self.assertLess(len(rsp), len(df))

        metrics = compute_benchmark_metrics(strat_rsp, rsp, rf_annual=0.015)
        self.assertEqual(metrics["start_date"], str(rsp.index[0].date()))
        self.assertGreater(metrics["observations"], 5000)

    def test_filter_pre_inception_zero_returns(self):
        """
        Sub-issue #122: Verify pre-inception synthetic zeros are filtered out
        and do not artificially dilute RSP volatility.
        """
        # Create dates from 2000-01-01 to 2004-01-01
        dates = pd.date_range("2000-01-01", "2004-01-01", freq="B")
        np.random.seed(42)

        # RSP has 0.0 until 2003-05-01, then real volatility
        rsp_raw = pd.Series(0.0, index=dates)
        post_mask = dates >= pd.to_datetime(RSP_INCEPTION_DATE)
        rsp_raw.loc[post_mask] = np.random.normal(loc=0.0004, scale=0.012, size=post_mask.sum())

        strat_raw = pd.Series(np.random.normal(loc=0.0005, scale=0.015, size=len(dates)), index=dates)

        # If uncleaned raw series is used:
        uncleaned_vol = rsp_raw.std(ddof=1) * np.sqrt(252)

        # Isolate and filter:
        strat_clean, rsp_clean = isolate_rsp_benchmark_series(strat_raw, rsp_raw)
        cleaned_vol = rsp_clean.std(ddof=1) * np.sqrt(252)

        # Cleaned volatility must be significantly higher than artificially diluted uncleaned volatility
        self.assertGreater(cleaned_vol, uncleaned_vol)
        self.assertGreaterEqual(rsp_clean.index[0], pd.to_datetime(RSP_INCEPTION_DATE))
        self.assertFalse((rsp_clean == 0.0).all())


if __name__ == "__main__":
    unittest.main()
