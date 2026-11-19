"""
Unit and integration tests for Fama-French 5-Factor ingestion, decimal conversion,
and calendar alignment (Milestone 4, Issue #196 / Sub-issue #207).
"""
import unittest
import numpy as np
import pandas as pd
from src.attribution import (
    BACKTEST_RESULTS_PATH,
    FF5_ALL_COLUMNS,
    FF5_CACHE_PATH,
    FF5_FACTOR_COLUMNS,
    align_strategy_and_factors,
    build_ff5_regression_matrices,
    calculate_annualized_alpha_inference,
    compute_newey_west_hac_covariance,
    convert_percent_to_decimal,
    download_ff5_daily_factors,
    estimate_ff5_ols_loadings,
    evaluate_alpha_hypothesis_test,
    format_ff5_regression_markdown_table,
    SyntheticBootstrapEngine,
    recommend_newey_west_lags,
)


class TestFactorIngestionAndAlignment(unittest.TestCase):
    """Test suite verifying FF5 data normalization and strategy calendar alignment."""

    def test_convert_percent_to_decimal_accuracy(self) -> None:
        """Sub-issue #205: Verify percentage figures divide by 100.0 accurately."""
        raw_df = pd.DataFrame(
            {
                "Mkt-RF": [1.25, -0.82],
                "SMB": [0.40, 0.10],
                "HML": [-0.50, 0.00],
                "RMW": [0.68, -0.35],
                "CMA": [0.39, 0.19],
                "RF": [0.01, 0.02],
            },
            index=pd.to_datetime(["2026-09-28", "2026-09-30"]),
        )
        dec_df = convert_percent_to_decimal(raw_df)
        self.assertAlmostEqual(float(dec_df.loc["2026-09-28", "Mkt-RF"]), 0.0125, places=8)
        self.assertAlmostEqual(float(dec_df.loc["2026-09-30", "Mkt-RF"]), -0.0082, places=8)
        self.assertAlmostEqual(float(dec_df.loc["2026-09-28", "RF"]), 0.0001, places=8)

    def test_cached_ff5_factors_are_decimal_scaled(self) -> None:
        """Sub-issue #204 & #205: Verify cached FF5 dataset exists and uses decimal units."""
        self.assertTrue(FF5_CACHE_PATH.exists(), f"Missing FF5 cache at {FF5_CACHE_PATH}")
        ff5_df = download_ff5_daily_factors(force_refresh=False, as_decimal=True)
        self.assertEqual(list(ff5_df.columns), list(FF5_ALL_COLUMNS))
        self.assertGreater(len(ff5_df), 6000)
        self.assertLess(float(ff5_df.abs().max().max()), 0.25)

    def test_date_alignment_between_strategy_and_ff5_factors(self) -> None:
        """Sub-issue #207: Verify exact DatetimeIndex alignment and excess return identity."""
        self.assertTrue(
            BACKTEST_RESULTS_PATH.exists(),
            f"Missing backtest results at {BACKTEST_RESULTS_PATH}",
        )
        bt_df = pd.read_csv(BACKTEST_RESULTS_PATH, index_col=0, parse_dates=True)
        ff5_df = download_ff5_daily_factors(force_refresh=False, as_decimal=True)

        for scheme_col in ("Black_CEO_Net", "Convex_Quality_Net"):
            with self.subTest(scheme=scheme_col):
                excess_ret, aligned_ff5 = align_strategy_and_factors(
                    bt_df[scheme_col],
                    ff5_df,
                )
                # Indices must match identically and have zero NaNs
                self.assertTrue(excess_ret.index.equals(aligned_ff5.index))
                self.assertFalse(excess_ret.isna().any())
                self.assertFalse(aligned_ff5.isna().any().any())
                self.assertGreater(len(excess_ret), 6000)

                # Verify R_e = r_net - RF on every aligned session
                expected_excess = (
                    bt_df.loc[excess_ret.index, scheme_col] - aligned_ff5["RF"]
                )
                np.testing.assert_allclose(
                    excess_ret.to_numpy(),
                    expected_excess.to_numpy(),
                    rtol=1e-12,
                    atol=1e-12,
                )

    def test_alignment_handles_timezone_and_non_overlapping_dates(self) -> None:
        """Sub-issue #207: Verify tz-aware normalization and ValueError on zero overlap."""
        dates_ff5 = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
        ff5_mock = pd.DataFrame(
            {col: [0.001, -0.002, 0.003] for col in FF5_ALL_COLUMNS},
            index=dates_ff5,
        )
        # Subset overlap with tz-aware index
        strat_tz = pd.Series(
            [0.01, -0.005],
            index=pd.to_datetime(["2020-01-03", "2020-01-06"]).tz_localize("UTC"),
        )
        excess_ret, _ = align_strategy_and_factors(strat_tz, ff5_mock)
        self.assertEqual(len(excess_ret), 2)
        self.assertIsNone(excess_ret.index.tz)
        self.assertAlmostEqual(float(excess_ret.iloc[0]), 0.01 - (-0.002), places=12)
        # Disjoint dates must raise ValueError
        strat_disjoint = pd.Series([0.01], index=pd.to_datetime(["1995-01-03"]))
        with self.assertRaises(ValueError):
            align_strategy_and_factors(strat_disjoint, ff5_mock)




class TestOLSRegressionAndHAC(unittest.TestCase):
    """Sub-issues #209-#213: Verify FF5 OLS regression, HAC covariance, and equity R-squared."""

    def test_synthetic_ff5_parameter_recovery_and_hac_symmetry(self) -> None:
        """Sub-issues #209-#211: Verify exact parameter recovery and symmetric PSD HAC matrix."""
        rng = np.random.default_rng(42)
        n_days = 600
        dates = pd.bdate_range("2018-01-02", periods=n_days)

        factors = pd.DataFrame(
            {
                "Mkt-RF": rng.normal(0.0004, 0.011, size=n_days),
                "SMB": rng.normal(0.0001, 0.006, size=n_days),
                "HML": rng.normal(0.0001, 0.006, size=n_days),
                "RMW": rng.normal(0.0001, 0.005, size=n_days),
                "CMA": rng.normal(0.0000, 0.005, size=n_days),
                "RF": np.full(n_days, 0.00008),
            },
            index=dates,
        )

        true_alpha_daily = 0.05 / 252.0
        raw_noise = rng.normal(0.0, 0.003, size=n_days)
        noise = raw_noise - np.mean(raw_noise)
        strat_ret = (
            factors["RF"]
            + true_alpha_daily
            + 1.05 * factors["Mkt-RF"]
            + 0.30 * factors["SMB"]
            - 0.20 * factors["HML"]
            + 0.15 * factors["RMW"]
            + 0.10 * factors["CMA"]
            + noise
        )

        y_vec, design_x, _ = build_ff5_regression_matrices(strat_ret, factors)
        self.assertEqual(y_vec.shape, (n_days,))
        self.assertEqual(design_x.shape, (n_days, 6))

        summary = estimate_ff5_ols_loadings(
            strat_ret,
            factors,
            scheme_name="Synthetic-FF5",
            hac_max_lags=5,
        )
        self.assertAlmostEqual(summary.alpha_annualized, 0.05, delta=0.005)
        self.assertAlmostEqual(summary.betas["Mkt-RF"], 1.05, delta=0.05)
        self.assertAlmostEqual(summary.betas["SMB"], 0.30, delta=0.08)
        self.assertAlmostEqual(summary.betas["HML"], -0.20, delta=0.08)
        self.assertGreater(summary.r_squared, 0.85)

        assert summary.residuals is not None
        cov_hac = compute_newey_west_hac_covariance(design_x, summary.residuals, max_lags=5)
        np.testing.assert_allclose(cov_hac, cov_hac.T, rtol=1e-12, atol=1e-15)
        eigvals = np.linalg.eigvalsh(cov_hac)
        self.assertTrue(np.all(eigvals > 0.0))
        self.assertEqual(recommend_newey_west_lags(6726), 10)

    def test_empirical_regression_r_squared_within_standard_equity_ranges(self) -> None:
        """Sub-issue #213: Confirm daily FF5 R-squared lies within standard equity bounds."""
        bt_df = pd.read_csv(BACKTEST_RESULTS_PATH, index_col=0, parse_dates=True)
        ff5_df = download_ff5_daily_factors(force_refresh=False, as_decimal=True)

        ew_res = estimate_ff5_ols_loadings(
            bt_df["Black_CEO_Net"],
            ff5_df,
            scheme_name="Equal-Weight",
            hac_max_lags=5,
        )
        cq_res = estimate_ff5_ols_loadings(
            bt_df["Convex_Quality_Net"],
            ff5_df,
            scheme_name="Convex-Quality",
            hac_max_lags=5,
        )

        for res in (ew_res, cq_res):
            with self.subTest(scheme=res.scheme_name):
                self.assertGreaterEqual(res.r_squared, 0.10)
                self.assertLessEqual(res.r_squared, 0.90)
                self.assertLessEqual(res.adj_r_squared, res.r_squared)
                self.assertGreater(res.betas["Mkt-RF"], 0.50)
                self.assertTrue(np.isfinite(res.alpha_t_stat))
                self.assertGreaterEqual(res.alpha_p_value, 0.0)
                self.assertLessEqual(res.alpha_p_value, 1.0)

        # Risk-controlled Convex Quality portfolio should exhibit tighter factor R^2 than EW
        self.assertGreater(cq_res.r_squared, ew_res.r_squared)




class TestStatisticalSignificanceAndResiduals(unittest.TestCase):
    """Sub-issues #214-#218: Verify alpha inference, hypothesis p-values, and zero residual mean."""

    def test_regression_residual_mean_equals_zero_and_orthogonal_to_factors(self) -> None:
        """Sub-issue #218: Assert OLS residuals have zero sample mean and factor orthogonality."""
        bt_df = pd.read_csv(BACKTEST_RESULTS_PATH, index_col=0, parse_dates=True)
        ff5_df = download_ff5_daily_factors(force_refresh=False, as_decimal=True)

        for col, name in [
            ("Black_CEO_Net", "Equal-Weight"),
            ("Convex_Quality_Net", "Convex-Quality"),
        ]:
            with self.subTest(scheme=name):
                _, design_x, _ = build_ff5_regression_matrices(bt_df[col], ff5_df)
                summary = estimate_ff5_ols_loadings(
                    bt_df[col],
                    ff5_df,
                    scheme_name=name,
                    hac_max_lags=5,
                )
                assert summary.residuals is not None
                residual_mean = float(np.mean(summary.residuals))
                self.assertAlmostEqual(residual_mean, 0.0, places=14)

                # Normal equations guarantee X^T * e = 0 across intercept and all 5 FF factors
                orthogonality = design_x.T @ summary.residuals
                np.testing.assert_allclose(
                    orthogonality, np.zeros(6), atol=1e-11
                )

    def test_alpha_hypothesis_test_and_markdown_table_export(
        self,
    ) -> None:
        """Sub-issues #214-#216: Verify annualized alpha CI, one-sided p-value, and Markdown table."""
        bt_df = pd.read_csv(BACKTEST_RESULTS_PATH, index_col=0, parse_dates=True)
        ff5_df = download_ff5_daily_factors(force_refresh=False, as_decimal=True)

        ew_inf = calculate_annualized_alpha_inference(
            bt_df["Black_CEO_Net"],
            ff5_df,
            scheme_name="Equal-Weight (Net)",
            hac_max_lags=5,
        )
        self.assertLess(ew_inf.ci_95_lower_annual, ew_inf.alpha_annualized)
        self.assertGreater(ew_inf.ci_95_upper_annual, ew_inf.alpha_annualized)
        self.assertAlmostEqual(
            ew_inf.p_value_one_sided_gt_zero,
            0.5 * ew_inf.p_value_two_sided,
            places=12,
        )

        verdict = evaluate_alpha_hypothesis_test(
            bt_df["Convex_Quality_Net"],
            ff5_df,
            scheme_name="Convex-Quality (Net)",
            hac_max_lags=5,
            significance_level=0.05,
        )
        self.assertFalse(verdict["reject_h0_alpha_lte_zero"])
        self.assertEqual(len(FF5_FACTOR_COLUMNS), 5)

        ew_ols = estimate_ff5_ols_loadings(
            bt_df["Black_CEO_Net"], ff5_df, scheme_name="Equal-Weight (Net)", hac_max_lags=5
        )
        cq_ols = estimate_ff5_ols_loadings(
            bt_df["Convex_Quality_Net"], ff5_df, scheme_name="Convex-Quality (Net)", hac_max_lags=5
        )
        md_table = format_ff5_regression_markdown_table([ew_ols, cq_ols], hac_max_lags=5)
        self.assertIn(r"| **Annualized Alpha ($\hat{\alpha}_{\text{ann}}$)** |", md_table)
        self.assertIn("Equal-Weight (Net)", md_table)
        self.assertIn("Convex-Quality (Net)", md_table)




class TestMonteCarloPermutationBootstrap(unittest.TestCase):
    """Sub-issues #220-#224: Verify Monte Carlo bootstrap reproducibility and S&P convergence."""

    def test_bootstrap_distribution_mean_converges_to_equal_weight_sp(self) -> None:
        """Sub-issue #224: Confirm 10,000-trial mean return path converges to equal-weight S&P."""
        bt_df = pd.read_csv(BACKTEST_RESULTS_PATH, index_col=0, parse_dates=True)
        mask_df = pd.read_csv("data/universe_mask.csv", index_col=0, parse_dates=True)
        active_nt = mask_df.sum(axis=1)

        engine = SyntheticBootstrapEngine.from_sp_benchmark_series(
            benchmark_returns=bt_df["SPY_Daily"],
            active_counts=active_nt,
            pool_size=500,
            seed=42,
        )
        trials = engine.simulate_random_portfolios(n_trials=10_000, seed=42)
        self.assertEqual(trials.shape, (10_000, len(bt_df)))

        # Sessions with zero active Black CEOs must yield 0.0 return across all trials
        zero_mask = engine.active_counts == 0
        if np.any(zero_mask):
            np.testing.assert_array_equal(trials[:, zero_mask], 0.0)

        # Law of Large Numbers: sample mean across 10,000 trials converges to equal-weight S&P
        bootstrap_mean_daily = trials.mean(axis=0)
        expected_ew_sp_daily = engine.equal_weight_pool_returns

        daily_rmse = float(np.sqrt(np.mean((bootstrap_mean_daily - expected_ew_sp_daily) ** 2)))
        ann_diff = abs(float(bootstrap_mean_daily.mean() - expected_ew_sp_daily.mean())) * 252.0

        self.assertLess(daily_rmse, 2e-4)
        self.assertLess(ann_diff, 5e-4)

    def test_bootstrap_seed_reproducibility(self) -> None:
        """Sub-issue #222: Verify identical seeds produce bit-for-bit identical simulations."""
        rng = np.random.default_rng(7)
        pool = rng.normal(0.0004, 0.015, size=(120, 50))
        counts = np.tile([0, 1, 2, 5, 10, 15], 20)

        engine = SyntheticBootstrapEngine(constituent_returns=pool, active_counts=counts, seed=99)
        sim_one = engine.simulate_random_portfolios(n_trials=250, seed=99)
        sim_two = engine.simulate_random_portfolios(n_trials=250, seed=99)
        sim_diff = engine.simulate_random_portfolios(n_trials=250, seed=100)

        np.testing.assert_array_equal(sim_one, sim_two)
        self.assertFalse(np.array_equal(sim_one, sim_diff))


if __name__ == "__main__":
    unittest.main()
