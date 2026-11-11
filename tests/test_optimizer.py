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


if __name__ == "__main__":
    unittest.main()
