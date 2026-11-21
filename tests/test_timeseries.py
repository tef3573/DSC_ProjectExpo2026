"""
tests/test_timeseries.py
Unit and integration tests for GARCH(1,1), GJR-GARCH(1,1), walk-forward volatility
targeting (#251), ARIMA/ARIMAX diagnostics (#252), and macro regime attribution.
"""
import unittest
import numpy as np
import pandas as pd

from src.arima_diagnostics import run_walk_forward_arima_diagnostics
from src.macro_regimes import evaluate_macro_regimes
from src.timeseries import (
    compute_subperiod_performance_metrics,
    fit_garch_11,
    fit_gjr_garch_11,
    run_walk_forward_garch_overlay,
)


class TestTimeSeriesModels(unittest.TestCase):
    """Test suite for parametric volatility, walk-forward overlays, and ARIMA diagnostics."""

    def test_synthetic_garch_stationarity_and_half_life(self) -> None:
        """Verify GARCH(1,1) MLE produces stationary alpha + beta < 1 and positive half-life."""
        rng = np.random.default_rng(42)
        n_obs = 1200
        eps = np.zeros(n_obs)
        sigma2 = np.zeros(n_obs)
        omega, alpha, beta = 2e-6, 0.10, 0.85
        sigma2[0] = omega / (1.0 - alpha - beta)
        for idx in range(1, n_obs):
            sigma2[idx] = omega + alpha * eps[idx - 1] ** 2 + beta * sigma2[idx - 1]
            eps[idx] = rng.normal(0.0, np.sqrt(sigma2[idx]))

        idx_dates = pd.bdate_range("2015-01-01", periods=n_obs)
        ret_series = pd.Series(0.0004 + eps, index=idx_dates, name="Synthetic_GARCH")
        res = fit_garch_11(ret_series)

        self.assertLess(res.params.persistence, 1.0)
        self.assertGreater(res.params.persistence, 0.5)
        self.assertGreater(res.params.half_life_days, 0.0)
        self.assertEqual(len(res.conditional_vol_annualized), n_obs)

    def test_gjr_garch_detects_asymmetric_downside_leverage(self) -> None:
        """Verify GJR-GARCH(1,1) estimates positive gamma on asymmetric shock series."""
        rng = np.random.default_rng(123)
        n_obs = 1200
        eps = np.zeros(n_obs)
        sigma2 = np.zeros(n_obs)
        omega, alpha, gamma, beta = 2e-6, 0.03, 0.18, 0.82
        sigma2[0] = 1e-4
        for idx in range(1, n_obs):
            shock = alpha + (gamma if eps[idx - 1] < 0.0 else 0.0)
            sigma2[idx] = omega + shock * eps[idx - 1] ** 2 + beta * sigma2[idx - 1]
            eps[idx] = rng.normal(0.0, np.sqrt(sigma2[idx]))

        idx_dates = pd.bdate_range("2015-01-01", periods=n_obs)
        res = fit_gjr_garch_11(pd.Series(eps, index=idx_dates))
        self.assertGreater(res.params.gamma, 0.05)
        self.assertLess(res.params.persistence, 1.0)

    def test_walk_forward_vol_overlay_compresses_volatility(self) -> None:
        """Sub-issue #251: Verify causal walk-forward vol overlay reduces realized volatility."""
        rng = np.random.default_rng(7)
        idx_dates = pd.bdate_range("2015-01-01", periods=800)
        high_vol_ret = pd.Series(rng.normal(0.0006, 0.025, size=800), index=idx_dates)

        wf_res = run_walk_forward_garch_overlay(high_vol_ret, target_ann_vol=0.20)
        self.assertEqual(wf_res.oos_sessions, 800 - 252)
        self.assertLess(
            wf_res.vol_targeted_metrics["annualized_volatility"],
            wf_res.unscaled_metrics["annualized_volatility"],
        )
        self.assertTrue(np.isfinite(wf_res.qlike_loss))

    def test_walk_forward_arima_and_arimax_diagnostics(self) -> None:
        """Sub-issue #252: Verify walk-forward ARIMA and ARIMAX outputs finite metrics."""
        rng = np.random.default_rng(99)
        idx_dates = pd.bdate_range("2015-01-01", periods=700)
        ret = pd.Series(rng.normal(0.0003, 0.015, size=700), index=idx_dates)
        exog = pd.DataFrame({"spy_lag1": rng.normal(0.0, 0.01, size=700)}, index=idx_dates)

        diag = run_walk_forward_arima_diagnostics(ret, exog_lagged=exog)
        self.assertEqual(diag.oos_sessions, 700 - 252)
        self.assertIn("lag_1", diag.ljung_box_pvalues)
        self.assertGreater(diag.arima_score.rmse, 0.0)
        self.assertIsNotNone(diag.arimax_score)
        self.assertTrue(0.0 <= diag.arima_score.directional_accuracy <= 1.0)

    def test_macro_regimes_and_subperiod_metrics(self) -> None:
        """Verify 6 macro event windows evaluate cleanly on backtest_results.csv."""
        report = evaluate_macro_regimes()
        self.assertEqual(len(report), 6)
        for window in report:
            self.assertIn("Black_CEO_Net", window["series"])
            self.assertGreater(window["sessions"], 100)

        sample_perf = compute_subperiod_performance_metrics(pd.Series([0.01, -0.005, 0.008]))
        self.assertIn("sharpe_ratio", sample_perf)
        self.assertIn("sortino_ratio", sample_perf)


if __name__ == "__main__":
    unittest.main()
