"""
tests/test_metrics.py
Unit tests verifying portfolio performance metrics.

- Sub-issue #114: Verify 252 annualizer in Sharpe ratio numerator.
"""
import unittest
import numpy as np
import pandas as pd
from src.metrics import compute_sharpe_ratio

class TestPerformanceMetrics(unittest.TestCase):

    def test_sharpe_ratio_annualization_scaling(self):
        """
        Sub-issue #114: Assert Sharpe ratio scales mean return by 252 and volatility by sqrt(252).
        """
        # Create synthetic series with known daily mean and std
        # 252 days: mean = +0.1% daily, std ~ 1% daily, Rf = 0.0
        np.random.seed(42)
        daily_returns = pd.Series(np.random.normal(loc=0.001, scale=0.01, size=252))

        daily_mean = daily_returns.mean()
        daily_std = daily_returns.std(ddof=1)
        expected_sharpe = (daily_mean * 252) / (daily_std * np.sqrt(252))

        calc_sharpe = compute_sharpe_ratio(daily_returns, rf_annual=0.0, periods_per_year=252)

        self.assertAlmostEqual(calc_sharpe, expected_sharpe, places=5)
        # Verify it does not equal the flawed sqrt(252)/sqrt(252) formula
        flawed_sharpe = daily_mean / (daily_std * np.sqrt(252))
        self.assertNotAlmostEqual(calc_sharpe, flawed_sharpe, places=3)


if __name__ == "__main__":
    unittest.main()