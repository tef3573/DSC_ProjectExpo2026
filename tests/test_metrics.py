"""
tests/test_metrics.py
Unit tests verifying portfolio performance metrics.

- Sub-issue #114: Verify 252 annualizer in Sharpe ratio numerator.
"""
import unittest
import numpy as np
import pandas as pd
from src.metrics import compute_downside_deviation, compute_sharpe_ratio, compute_sortino_ratio

class TestPerformanceMetrics(unittest.TestCase):

    """
    Sub-issue #114: Assert Sharpe ratio scales mean return by 252 and volatility by sqrt(252).
    """
    def test_sharpe_ratio_annualization_scaling(self):
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

    """
    Sub-issue #115: Verify downside deviation divides by total length T, not down days count
    """
    def test_downside_deviation_uses_total_sample_length_t(self):
        # 4 days total (T=4): 2 positive days (+2%, +1%), 2 negative days (-1%, -3%), Rf=0
        returns = pd.Series([0.02, 0.01, -0.01, -0.03])
        # Downside deviations: [0, 0, -0.01, -0.03]
        # Sum of squares = 0^2 + 0^2 + (-0.01)^2 + (-0.03)^2 = 0.0001 + 0.0009 = 0.0010
        # Correct (T=4): sqrt(0.0010 / 4) = sqrt(0.00025) = 0.015811
        # Flawed (N_down=2): sqrt(0.0010 / 2) = sqrt(0.0005) = 0.022360
        expected_downside_t = np.sqrt(0.0010 / 4.0)
        calc_downside = compute_downside_deviation(returns, rf_annual=0.0, periods_per_year=252)

        self.assertAlmostEqual(calc_downside, expected_downside_t, places=6)

        # Confirm Sortino incorporates this correct T-based downside
        mean_excess = returns.mean()  # (-0.01)/4 = -0.0025
        expected_sortino = (mean_excess * 252) / (expected_downside_t * np.sqrt(252))
        calc_sortino = compute_sortino_ratio(returns, rf_annual=0.0, periods_per_year=252)
        self.assertAlmostEqual(calc_sortino, expected_sortino, places=5)

if __name__ == "__main__":
    unittest.main()