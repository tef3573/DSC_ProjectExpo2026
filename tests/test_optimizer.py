"""
tests/test_optimizer.py
Unit tests verifying Ledoit-Wolf shrinkage, eigenvalue conditioning, and strict positive
semi-definiteness invariants for asset return covariance matrices.
- Sub-issue #153: Assert estimated covariance matrix is strictly positive semi-definite.
"""

import unittest
import numpy as np
import pandas as pd
import cvxpy as cp
from src.optimizer import (
    CovarianceEstimator,
    compute_equal_weights,
    ledoit_wolf_shrinkage,
    condition_covariance_matrix,
    compute_inverse_volatility_weights,
    ConvexPortfolioOptimizer,
    normalize_quality_scores,
    compute_fundamental_quality_scores,
    REGULARIZATION_EPSILON,
)

class TestCovarianceEstimation(unittest.TestCase):
    def setUp(self):
        self.estimator = CovarianceEstimator(lookback_window=100, min_observations=20)

    def test_estimated_covariance_is_strictly_symmetric(self):
        """Verify cov == cov.T within float precision."""
        np.random.seed(42)
        returns = pd.DataFrame(np.random.normal(0, 0.02, size=(120, 5)))
        cov, _ = self.estimator.estimate(returns)

        diff = np.abs(cov - cov.T)
        self.assertTrue((diff < 1e-12).all(), "Covariance matrix is not symmetric.")

    def test_estimated_covariance_eigenvalues_positive_semi_definite(self):
        """Sub-issue #153: Assert all eigenvalues >= epsilon > 0 (strictly PSD)."""
        np.random.seed(101)
        # 15 assets with only 25 observations (ill-conditioned N approaching T regime)
        returns = pd.DataFrame(np.random.normal(0.0005, 0.015, size=(25, 15)))
        cov, delta = self.estimator.estimate(returns)

        eigenvalues = np.linalg.eigvalsh(cov)
        min_eig = np.min(eigenvalues)

        self.assertGreaterEqual(
            min_eig,
            REGULARIZATION_EPSILON - 1e-9,
            f"Minimum eigenvalue {min_eig} violates PSD floor {REGULARIZATION_EPSILON}."
        )
        self.assertTrue(0.0 <= delta <= 1.0, f"Shrinkage intensity {delta} must be in [0, 1].")

    def test_collinear_returns_conditioned_to_psd(self):
        """Sub-issue #153: Test perfect collinearity (rank-deficient return matrix)."""
        base_series = np.random.normal(0, 0.01, size=50)
        # Stock B and C are linear multiples of Stock A
        returns = pd.DataFrame({
            "A": base_series,
            "B": base_series * 2.0,
            "C": base_series * -1.5,
        })
        cov, _ = self.estimator.estimate(returns)

        eigenvalues = np.linalg.eigvalsh(cov)
        self.assertTrue(
            (eigenvalues >= REGULARIZATION_EPSILON - 1e-9).all(),
            "Collinear assets failed to condition into positive semi-definite matrix."
        )

    def test_quadratic_form_non_negative_for_arbitrary_weights(self):
        """Assert w^T * Sigma * w >= 0 for arbitrary non-zero allocation vectors."""
        np.random.seed(99)
        returns = pd.DataFrame(np.random.normal(0, 0.02, size=(80, 4)))
        cov, _ = self.estimator.estimate(returns)

        # Generate 100 random weight vectors
        for _ in range(100):
            w = np.random.uniform(-1, 1, size=4)
            variance = w.T @ cov @ w
            self.assertGreaterEqual(
                variance,
                0.0,
                f"Quadratic form w^T Sigma w was negative: {variance}"
            )

class TestEqualWeighting(unittest.TestCase):
    """Sub-issue #155: Test dynamic 1/Nt equal weighting function."""

    def test_zero_active_constituents(self):
        mask = np.array([False, False, False, False])
        weights = compute_equal_weights(mask)
        np.testing.assert_array_equal(weights, np.zeros(4))
        self.assertEqual(np.sum(weights), 0.0)

    def test_single_constituent_concentration_defense(self):
        mask = np.array([True, False, False, False])
        weights = compute_equal_weights(mask, concentration_defense=True)
        # N_t = 1 -> capped at 1/3
        self.assertAlmostEqual(weights[0], 1.0 / 3.0, places=7)
        self.assertAlmostEqual(np.sum(weights), 1.0 / 3.0, places=7)

    def test_two_constituents_concentration_defense(self):
        mask = np.array([True, False, True, False])
        weights = compute_equal_weights(mask, concentration_defense=True)
        # N_t = 2 -> total equity 1/3, each gets 1/6
        self.assertAlmostEqual(weights[0], 1.0 / 6.0, places=7)
        self.assertAlmostEqual(weights[2], 1.0 / 6.0, places=7)
        self.assertAlmostEqual(np.sum(weights), 1.0 / 3.0, places=7)

    def test_three_or_more_full_equity_allocation(self):
        mask = np.array([True, True, True, False, True])
        weights = compute_equal_weights(mask, concentration_defense=True)
        # N_t = 4 -> each gets 0.25, sum is 1.0
        expected = np.array([0.25, 0.25, 0.25, 0.0, 0.25])
        np.testing.assert_allclose(weights, expected, atol=1e-7)
        self.assertAlmostEqual(np.sum(weights), 1.0, places=7)

    def test_2d_vectorized_mask(self):
        mask_2d = np.array([
            [False, False, False],  # N_t = 0
            [True, False, False],   # N_t = 1
            [True, True, False],    # N_t = 2
            [True, True, True],     # N_t = 3
        ])
        weights_2d = compute_equal_weights(mask_2d, concentration_defense=True)
        self.assertEqual(weights_2d.shape, (4, 3))
        self.assertAlmostEqual(np.sum(weights_2d[0]), 0.0, places=7)
        self.assertAlmostEqual(np.sum(weights_2d[1]), 1.0 / 3.0, places=7)
        self.assertAlmostEqual(np.sum(weights_2d[2]), 1.0 / 3.0, places=7)
        self.assertAlmostEqual(np.sum(weights_2d[3]), 1.0, places=7)

class TestInverseVolatilityWeighting(unittest.TestCase):
    """Sub-issues #156, #157, #159: Test inverse-volatility allocation."""
    def test_lower_volatility_receives_higher_weight(self):
        np.random.seed(42)
        r0 = np.random.normal(0, 0.01, size=(100, 1))
        r1 = np.random.normal(0, 0.05, size=(100, 1))
        r_df = pd.DataFrame(np.hstack([r0, r1]), columns=["LOW_VOL", "HIGH_VOL"])
        mask = np.ones((100, 2), dtype=bool)
        w = compute_inverse_volatility_weights(r_df, mask, lookback_window=63, concentration_defense=False)
        self.assertGreater(w[-1, 0], w[-1, 1])
        self.assertAlmostEqual(np.sum(w[-1]), 1.0, places=6)

    def test_epsilon_floor_guards_zero_volatility(self):
        r_df = pd.DataFrame(np.zeros((50, 3)), columns=["A", "B", "C"])
        mask = np.ones((50, 3), dtype=bool)
        w = compute_inverse_volatility_weights(r_df, mask, epsilon=1e-4, concentration_defense=False)
        self.assertEqual(w.shape, (50, 3))
        self.assertFalse(np.isnan(w).any())
        self.assertAlmostEqual(np.sum(w[-1]), 1.0, places=6)

    def test_weights_sum_to_unity_when_nt_gte_3(self):
        np.random.seed(42)
        r_df = pd.DataFrame(np.random.normal(0, 0.02, size=(80, 4)), columns=["A", "B", "C", "D"])
        mask = np.ones((80, 4), dtype=bool)
        w = compute_inverse_volatility_weights(r_df, mask, concentration_defense=True)
        self.assertAlmostEqual(np.sum(w[-1]), 1.0, places=6)

class TestConvexQPOptimizer(unittest.TestCase):
    """Sub-issues #162, #163, #164, #166: Test CVXPY QP formulation and solver convergence."""

    def setUp(self):
        self.optimizer = ConvexPortfolioOptimizer()

    def test_cp_variable_construction(self):
        """Sub-issue #162: Verify construct_optimization_problem initializes cp.Variable."""
        cov = np.eye(3) * 0.04
        w, problem = self.optimizer.construct_optimization_problem(cov)
        self.assertIsInstance(w, cp.Variable)
        self.assertEqual(w.shape, (3,))
        self.assertIsInstance(problem, cp.Problem)
        self.assertTrue(problem.is_dcp())

    def test_objective_is_quadratic_variance(self):
        """Sub-issue #163: Verify objective is convex minimization of variance."""
        cov = np.eye(4) * 0.05
        _, problem = self.optimizer.construct_optimization_problem(cov)
        self.assertIsInstance(problem.objective, cp.Minimize)
        self.assertTrue(problem.objective.is_dcp())

    def test_budget_and_long_only_constraints(self):
        """Sub-issue #164: Verify w >= 0 and sum(w) == equity_budget."""
        cov = np.eye(3) * 0.04
        w, problem = self.optimizer.construct_optimization_problem(cov, equity_budget=1.0 / 3.0)

        # 2 constraints: w >= 0, sum(w) == 1/3
        self.assertEqual(len(problem.constraints), 2)
        weights, _ = self.optimizer.solve(cov, equity_budget=1.0 / 3.0)

        self.assertTrue(np.all(weights >= -1e-7))
        self.assertAlmostEqual(np.sum(weights), 1.0 / 3.0, places=6)

    def test_solver_convergence_on_standard_covariance(self):
        """Sub-issue #166: Verify solver convergence on standard asset covariance input."""
        cov = np.array([
            [0.04, 0.01],
            [0.01, 0.09]
        ])
        weights, variance = self.optimizer.solve(cov, equity_budget=1.0)

        # Asset 0 (lower variance) must receive higher allocation than Asset 1
        self.assertGreater(weights[0], weights[1])
        self.assertAlmostEqual(np.sum(weights), 1.0, places=6)
        self.assertGreater(variance, 0.0)

    def test_solver_convergence_on_standard_covariance(self):
        """Sub-issue #166: Verify solver convergence on standard asset covariance input."""
        # 3-asset standard covariance matrix with realistic correlation structure
        cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.02],
            [0.005, 0.02, 0.16]
        ])
        weights, variance = self.optimizer.solve(cov, equity_budget=1.0)

        # 1. Weights must be non-negative (long-only) and sum to budget
        self.assertTrue(np.all(weights >= -1e-7))
        self.assertAlmostEqual(np.sum(weights), 1.0, places=6)

        # 2. Lower variance assets must receive strictly higher allocations
        # Asset 0 (vol=20%) > Asset 1 (vol=30%) > Asset 2 (vol=40%)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])

        # 3. Portfolio variance must be strictly positive and lower than equal-weight variance
        eq_w = np.full(3, 1.0 / 3.0)
        eq_var = float(eq_w.T @ cov @ eq_w)
        self.assertGreater(variance, 0.0)
        self.assertLess(variance, eq_var)

    def test_solver_handles_near_singular_matrix(self):
        """Sub-issue #166: Verify convergence stability on near-collinear covariance."""
        # Near rank-deficient covariance matrix
        cov = np.array([
            [0.04, 0.0399],
            [0.0399, 0.04]
        ])
        weights, variance = self.optimizer.solve(cov, equity_budget=1.0)
        self.assertFalse(np.isnan(weights).any())
        self.assertAlmostEqual(np.sum(weights), 1.0, places=6)
        self.assertGreaterEqual(variance, 0.0)

class TestQualityScoreTilts(unittest.TestCase):
    """Sub-issues #168, #169, #170, #172: Test quality scoring, peer normalization, and tilts."""

    def test_z_score_normalization_properties(self):
        """Sub-issue #170: Assert mean 0, std 1 across active peers and 0 for inactive."""
        raw = np.array([10.0, 20.0, 30.0, 40.0, 99.0])
        mask = np.array([True, True, True, True, False])
        z = normalize_quality_scores(raw, mask)

        self.assertEqual(z[4], 0.0)
        self.assertAlmostEqual(np.mean(z[:4]), 0.0, places=6)
        self.assertAlmostEqual(np.std(z[:4], ddof=1), 1.0, places=6)

    def test_quality_vector_ranks_high_drift_low_vol_highest(self):
        """Sub-issue #168: High-drift, low-downside-vol asset receives highest q_i."""
        np.random.seed(42)
        t = 300
        r_high = np.random.normal(0.0015, 0.008, size=(t, 1))
        r_mid = np.random.normal(0.0001, 0.020, size=(t, 1))
        r_low = np.random.normal(-0.0015, 0.045, size=(t, 1))

        r_df = pd.DataFrame(
            np.hstack([r_high, r_mid, r_low]),
            columns=["HIGH_Q", "MID_Q", "LOW_Q"]
        )
        mask = np.ones((t, 3), dtype=bool)

        q_mat = compute_fundamental_quality_scores(r_df, mask)
        q_last = q_mat[-1]

        self.assertEqual(q_mat.shape, (t, 3))
        self.assertGreater(q_last[0], q_last[1])
        self.assertGreater(q_last[1], q_last[2])

    def test_higher_quality_constituents_receive_positive_allocation_tilts(self):
        """Sub-issue #172: Higher quality constituents receive strictly higher weights."""
        optimizer = ConvexPortfolioOptimizer()
        # Equal variance and zero correlation across 3 assets
        cov = np.eye(3) * 0.04
        quality = np.array([-1.0, 0.0, 1.0])
        # Baseline min-variance weights without tilt (lambda=0) -> equal 1/3 each
        w_base, _ = optimizer.solve(cov, quality_scores=quality, lambda_tilt=0.0, equity_budget=1.0)
        # Tilted weights (lambda=0.05)
        w_tilted, _ = optimizer.solve(cov, quality_scores=quality, lambda_tilt=0.05, equity_budget=1.0)

        # High quality asset (index 2) gets positive tilt above baseline; low quality (index 0) gets negative tilt
        self.assertGreater(w_tilted[2], w_base[2])
        self.assertLess(w_tilted[0], w_base[0])
        self.assertGreater(w_tilted[2], w_tilted[1])
        self.assertGreater(w_tilted[1], w_tilted[0])
        self.assertAlmostEqual(np.sum(w_tilted), 1.0, places=6)



if __name__ == "__main__":
    unittest.main()
