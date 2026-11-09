"""
src/optimizer.py
Convex Quality Optimization & Covariance Estimation Module.
Milestone 3: Convex Quality Optimization (cvxpy)
- Sub-issues #149, #150, #151: Module structure, Ledoit-Wolf shrinkage, PSD conditioning.
"""
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

TRADING_DAYS_PER_YEAR = 252
DEFAULT_LOOKBACK_DAYS = 252
MIN_OBSERVATIONS_REQUIRED = 63
REGULARIZATION_EPSILON = 1e-4


def ledoit_wolf_shrinkage(returns: pd.DataFrame, shrinkage_target_identity: bool = True) -> Tuple[np.ndarray, float]:
    """
    Computes analytical Ledoit-Wolf shrinkage covariance matrix.
    """
    x = returns.values
    t, n = x.shape
    if n == 0:
        return np.empty((0, 0)), 0.0
    if n == 1:
        var = np.var(x, ddof=1) if t > 1 else np.var(x)
        return np.array([[max(var, REGULARIZATION_EPSILON)]]), 0.0

    # Demean returns
    x_mean = np.mean(x, axis=0)
    x_centered = x - x_mean

    # Sample covariance S
    sample_cov = (x_centered.T @ x_centered) / (t - 1) if t > 1 else (x_centered.T @ x_centered) / t

    # Target F = mu * I
    mean_var = np.trace(sample_cov) / n
    target = mean_var * np.eye(n)

    if not shrinkage_target_identity:
        # Structured correlation target
        d = np.sqrt(np.diag(sample_cov))
        d_outer = np.outer(d, d)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = sample_cov / d_outer
        corr = np.nan_to_num(corr, nan=0.0)
        mean_corr = (np.sum(corr) - n) / (n * (n - 1)) if n > 1 else 0.0
        target = mean_corr * d_outer
        np.fill_diagonal(target, np.diag(sample_cov))

    # Analytical optimal shrinkage intensity delta
    y = x_centered**2
    phi_mat = (y.T @ y) / t - 2.0 * (x_centered.T @ x_centered) * sample_cov / t + sample_cov**2
    phi = np.sum(phi_mat)

    gamma = np.linalg.norm(sample_cov - target, ord="fro") ** 2
    kappa = phi / gamma if gamma > 0 else 0.0
    delta = max(0.0, min(1.0, kappa / t))

    shrunk_cov = (1.0 - delta) * sample_cov + delta * target
    return shrunk_cov, delta


def condition_covariance_matrix(cov: np.ndarray, epsilon: float = REGULARIZATION_EPSILON) -> np.ndarray:
    """
    Guarantees strict positive semi-definiteness via symmetric projection and eigenvalue clipping.
    """
    if cov.size == 0:
        return cov

    # Symmetric projection
    sym_cov = 0.5 * (cov + cov.T)

    # Spectral decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(sym_cov)

    # Floor eigenvalues at epsilon
    clipped_eigenvalues = np.maximum(eigenvalues, epsilon)

    # Reconstruct conditioned covariance matrix
    psd_cov = eigenvectors @ np.diag(clipped_eigenvalues) @ eigenvectors.T
    return 0.5 * (psd_cov + psd_cov.T)


class CovarianceEstimator:
    """
    Estimator class for asset return covariance matrices with shrinkage regularization.
    """

    def __init__(
        self,
        lookback_window: int = DEFAULT_LOOKBACK_DAYS,
        min_observations: int = MIN_OBSERVATIONS_REQUIRED,
        epsilon: float = REGULARIZATION_EPSILON
    ) -> None:
        self.lookback_window = lookback_window
        self.min_observations = min_observations
        self.epsilon = epsilon

    def estimate(self, returns: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        Estimates a conditioned, positive semi-definite shrinkage covariance matrix.
        """
        if len(returns) < self.min_observations:
            logging.warning(
                "Historical observations (%d) < min required (%d). Applying strong conditioning.",
                len(returns),
                self.min_observations
            )

        shrunk_cov, delta = ledoit_wolf_shrinkage(returns)
        conditioned_cov = condition_covariance_matrix(shrunk_cov, epsilon=self.epsilon)
        return conditioned_cov, delta


if __name__ == "__main__":
    estimator = CovarianceEstimator()
    print("src/optimizer.py initialized and ready.")
