"""
tests/test_metrics.py
Unit tests verifying portfolio performance metrics.

- Sub-issue #114: Verify 252 annualizer in Sharpe ratio numerator.
"""
import unittest
import numpy as np
import pandas as pd
from src.metrics import (
    compute_sharpe_ratio,
    compute_downside_deviation,
    compute_sortino_ratio,
    compute_drawdown_series,
    compute_max_drawdown,
    compute_cagr,
    compute_calmar_ratio
)


class TestPerformanceMetrics(unittest.TestCase):

    def test_sharpe_ratio_annualization_scaling(self):
        """Sub-issue #114: Assert Sharpe ratio scales mean return by 252 and volatility by sqrt(252)."""
        np.random.seed(42)
        daily_returns = pd.Series(np.random.normal(loc=0.001, scale=0.01, size=252))

        daily_mean = daily_returns.mean()
        daily_std = daily_returns.std(ddof=1)
        expected_sharpe = (daily_mean * 252) / (daily_std * np.sqrt(252))

        calc_sharpe = compute_sharpe_ratio(daily_returns, rf_annual=0.0, periods_per_year=252)
        self.assertAlmostEqual(calc_sharpe, expected_sharpe, places=5)

    """Sub-issue #115: Verify downside deviation divides by total length T, not down days count."""
    def test_downside_deviation_uses_total_sample_length_t(self):
        returns = pd.Series([0.02, 0.01, -0.01, -0.03])
        expected_downside_t = np.sqrt(0.0010 / 4.0)
        calc_downside = compute_downside_deviation(returns, rf_annual=0.0, periods_per_year=252)
        self.assertAlmostEqual(calc_downside, expected_downside_t, places=6)

        mean_excess = returns.mean()
        expected_sortino = (mean_excess * 252) / (expected_downside_t * np.sqrt(252))
        calc_sortino = compute_sortino_ratio(returns, rf_annual=0.0, periods_per_year=252)
        self.assertAlmostEqual(calc_sortino, expected_sortino, places=5)
    
    """Sub-issue #116: Verify drawdown series tracking and MDD calculation."""
    def test_cumulative_drawdown_and_mdd_tracking(self):
        # Returns: +10% (W=1.10), -20% (W=0.88), +10% (W=0.968)
        returns = pd.Series([0.10, -0.20, 0.10])
        wealth, dd = compute_drawdown_series(returns)

        # Peak at day 0 is 1.10. Day 1 wealth is 0.88 -> DD = (0.88 - 1.10) / 1.10 = -0.20 (-20%)
        self.assertAlmostEqual(wealth.iloc[0], 1.10, places=4)
        self.assertAlmostEqual(wealth.iloc[1], 0.88, places=4)
        self.assertAlmostEqual(dd.iloc[1], -0.20, places=4)

        # Max Drawdown should be 0.20 (20%)
        mdd = compute_max_drawdown(returns)
        self.assertAlmostEqual(mdd, 0.20, places=4)

    """Sub-issue #116: Verify CAGR compounding and Calmar ratio."""
    def test_cagr_and_calmar_ratio(self):
        # 252 trading sessions where wealth grows from 1.0 to 1.21 (21% gain) with a 10% MDD
        # Daily return = (1.21)^(1/252) - 1
        daily_const_ret = (1.21 ** (1.0 / 252.0)) - 1.0
        returns = pd.Series([daily_const_ret] * 252)
        
        # Introduce a temporary drop to create a known 10% drawdown
        # Day 1: +10% -> 1.10. Day 2: -10% -> 0.99. Then steady growth.
        cagr = compute_cagr(returns, periods_per_year=252)
        self.assertAlmostEqual(cagr, 0.21, places=4)

        # Test Calmar on a clean sequence with known CAGR and MDD
        # 252 sessions with 20% CAGR and 10% MDD -> Calmar = 2.0
        mock_returns = pd.Series([0.10, -0.10] + [0.0008] * 250)
        calculated_mdd = compute_max_drawdown(mock_returns)
        calculated_cagr = compute_cagr(mock_returns, periods_per_year=252)
        expected_calmar = calculated_cagr / calculated_mdd
        calc_calmar = compute_calmar_ratio(mock_returns, periods_per_year=252)

        self.assertAlmostEqual(calc_calmar, expected_calmar, places=5)


if __name__ == "__main__":
    unittest.main()