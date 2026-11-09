"""
tests/test_optimizer.py
Unit tests verifying Ledoit-Wolf shrinkage, eigenvalue conditioning, and strict positive
semi-definiteness invariants for asset return covariance matrices.
- Sub-issue #153: Assert estimated covariance matrix is strictly positive semi-definite.
"""
import unittest
import numpy as np
import pandas as pd
from src.optimizer import (
    CovarianceEstimator,
    ledoit_wolf_shrinkage,
    condition_covariance_matrix,
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


if __name__ == "__main__":
    unittest.main()
