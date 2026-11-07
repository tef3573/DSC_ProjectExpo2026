"""
tests/test_benchmark_window.py
Unit tests verifying benchmark window decoupling and alignment.
- Sub-issue #120: Align SPY cap-weighted comparison across full 2000-2026 horizon.
"""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from src.metrics import align_benchmark_series, compute_benchmark_metrics

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PORTFOLIO_RETURNS_PATH = DATA_DIR / "portfolio_daily_returns.csv"


class TestBenchmarkWindowDecoupling(unittest.TestCase):

    def test_spy_full_horizon_alignment(self):
        """
        Sub-issue #120: Verify SPY alignment captures full simulation sessions without truncation.
        """
        if not PORTFOLIO_RETURNS_PATH.exists():
            self.skipTest(f"Missing {PORTFOLIO_RETURNS_PATH}")

        df = pd.read_csv(PORTFOLIO_RETURNS_PATH, index_col=0, parse_dates=True)
        self.assertIn("Black_CEO_Net", df.columns)
        self.assertIn("SPY", df.columns)

        strat, spy = align_benchmark_series(df["Black_CEO_Net"], df["SPY"])

        # SPY must cover the exact row count of the backtest
        self.assertEqual(len(strat), len(df))
        self.assertEqual(len(spy), len(df))
        self.assertTrue((strat.index == spy.index).all())

        # Validate start and end match the full 2000-2026 period
        start_year = strat.index[0].year
        end_year = strat.index[-1].year
        self.assertEqual(start_year, 2000)
        self.assertGreaterEqual(end_year, 2026)

    def test_spy_benchmark_metrics_calculation(self):
        """
        Sub-issue #120: Assert compute_benchmark_metrics outputs valid alpha, beta, and tracking error.
        """
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        # Provide non-zero variance in benchmark returns
        np.random.seed(42)
        spy_vals = np.random.normal(loc=0.0005, scale=0.01, size=252)
        spy_ret = pd.Series(spy_vals, index=dates)

        # Strategy with exact beta = 1.2, zero alpha (Rf = 0)
        strat_ret = spy_ret * 1.2

        metrics = compute_benchmark_metrics(strat_ret, spy_ret, rf_annual=0.0)

        self.assertEqual(metrics["observations"], 252)
        self.assertAlmostEqual(metrics["beta"], 1.2, places=4)
        self.assertAlmostEqual(metrics["alpha_annual"], 0.0, places=4)
        expected_te = (strat_ret - spy_ret).std(ddof=1) * np.sqrt(252)
        self.assertAlmostEqual(metrics["tracking_error"], expected_te, places=5)


if __name__ == "__main__":
    unittest.main()
