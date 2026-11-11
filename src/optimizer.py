"""
src/optimizer.py
Convex Quality Optimization & Covariance Estimation Module.
Milestone 3: Convex Quality Optimization (cvxpy)
- Sub-issue #149: Initialize src/optimizer.py module structure.
- Sub-issue #150: Implement Ledoit-Wolf shrinkage covariance estimator.
- Sub-issue #151: Condition covariance matrix for low-sample historical windows.
"""
from pathlib import Path
from typing import Tuple, Optional
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


def ledoit_wolf_shrinkage(
    returns: pd.DataFrame,
    shrinkage_target: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float]:
    """
    Sub-issue #150: Computes the analytical Ledoit-Wolf shrinkage covariance estimator.
    Shrinks sample covariance S toward structured target F (constant correlation / scaled identity):
        Sigma_shrunk = delta * F + (1 - delta) * S
    """
    x = returns.values
    t, n = x.shape

    if t < 2 or n == 0:
        eye = np.eye(max(n, 1)) * REGULARIZATION_EPSILON
        return eye, 1.0

    mean_vec = np.nanmean(x, axis=0)
    x_centered = np.nan_to_num(x - mean_vec, nan=0.0)

    # Sample covariance matrix S
    sample_cov = (x_centered.T @ x_centered) / (t - 1)

    # Shrinkage Target F: diagonal variance target with zero off-diagonal covariance
    # (or constant variance target m * I)
    mu_var = np.trace(sample_cov) / n
    if shrinkage_target is None:
        target = mu_var * np.eye(n)
    else:
        target = shrinkage_target

    # Gamma: squared Frobenius norm of (sample_cov - target)
    gamma = float(np.sum((sample_cov - target) ** 2))

    if gamma == 0.0:
        return sample_cov, 0.0

    # Phi: sum of asymptotic variances of sample covariance entries
    # y_t = x_centered_{t, i} * x_centered_{t, j}
    # Matrix formulation of phi:
    y = np.zeros((t, n * n))
    for idx in range(t):
        outer = np.outer(x_centered[idx], x_centered[idx])
        y[idx] = outer.flatten()

    y_mean = np.mean(y, axis=0)
    phi_mat = np.sum((y - y_mean) ** 2, axis=0) / t
    phi = float(np.sum(phi_mat))

    # Optimal shrinkage intensity delta in [0, 1]
    kappa = phi / gamma
    delta = float(np.clip(kappa / t, 0.0, 1.0))

    # Shrunk covariance
    shrunk_cov = delta * target + (1.0 - delta) * sample_cov
    return shrunk_cov, delta


def condition_covariance_matrix(
    cov_matrix: np.ndarray,
    epsilon: float = REGULARIZATION_EPSILON
) -> np.ndarray:
    """
    Sub-issue #151: Guarantees strict symmetry and positive semi-definiteness (PSD)
    by clipping eigenvalues to an epsilon floor: lambda_i >= epsilon > 0.
    """
    # Enforce exact symmetry
    cov_sym = 0.5 * (cov_matrix + cov_matrix.T)

    # Spectral decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_sym)

    # Condition eigenvalues: floor at epsilon to prevent rank deficiency
    clipped_eigenvalues = np.clip(np.nan_to_num(eigenvalues, nan=epsilon), epsilon, None)

    # Reconstruct positive semi-definite matrix
    psd_cov = eigenvectors @ np.diag(clipped_eigenvalues) @ eigenvectors.T
    return 0.5 * (psd_cov + psd_cov.T)

def compute_equal_weights(
    active_mask: np.ndarray,
    concentration_defense: bool = True
) -> np.ndarray:
    """
    Sub-issue #155: Computes dynamic 1/Nt equal weighting across active constituents.
    """
    active_mask = np.asarray(active_mask, dtype=bool)
    
    # Handle 1D slice (single session)
    if active_mask.ndim == 1:
        n_active = int(np.sum(active_mask))
        weights = np.zeros_like(active_mask, dtype=np.float64)
        
        if n_active == 0:
            return weights
            
        if concentration_defense and n_active < 3:
            equity_budget = 1.0 / 3.0
        else:
            equity_budget = 1.0
            
        weights[active_mask] = equity_budget / n_active
        return weights

    # Handle 2D matrix (T sessions x N assets)
    elif active_mask.ndim == 2:
        n_active = np.sum(active_mask, axis=1, keepdims=True)  # Shape (T, 1)
        weights = np.zeros_like(active_mask, dtype=np.float64)
        
        # Avoid division by zero
        safe_n = np.where(n_active > 0, n_active, 1.0)
        
        if concentration_defense:
            equity_budget = np.where(
                n_active == 0,
                0.0,
                np.where(n_active < 3, 1.0 / 3.0, 1.0)
            )
        else:
            equity_budget = np.where(n_active > 0, 1.0, 0.0)
            
        weights = np.where(active_mask, equity_budget / safe_n, 0.0)
        return weights

    else:
        raise ValueError(f"active_mask must be 1D or 2D, got shape {active_mask.shape}")

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
        Sub-issue #150 & #151: Estimates, shrinks, and conditions the covariance matrix.
        """
        shrunk_cov, delta = ledoit_wolf_shrinkage(returns)
        psd_cov = condition_covariance_matrix(shrunk_cov, epsilon=self.epsilon)
        return psd_cov, delta











if __name__ == "__main__":
    estimator = CovarianceEstimator()
    print("src/optimizer.py initialized with Ledoit-Wolf shrinkage and PSD conditioning.")
