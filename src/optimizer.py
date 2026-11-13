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
import cvxpy as cp
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


def compute_inverse_volatility_weights(
    returns: pd.DataFrame,
    active_mask: np.ndarray,
    lookback_window: int = 63,
    epsilon: float = REGULARIZATION_EPSILON,
    concentration_defense: bool = True
) -> np.ndarray:
    """
    Sub-issues #156 & #157: Computes inverse-volatility risk parity allocation.
    """
    ret_values = returns.values
    t, n = ret_values.shape
    weights = np.zeros((t, n), dtype=np.float64)
    for idx in range(t):
        current_mask = active_mask[idx]
        n_active = int(np.sum(current_mask))
        if n_active == 0:
            continue  
        start_idx = max(0, idx - lookback_window + 1)
        window_returns = ret_values[start_idx:idx + 1]
        # Realized daily volatility per asset
        if len(window_returns) > 1:
            vol = np.nanstd(window_returns, axis=0, ddof=1)
        else:
            vol = np.zeros(n)
        # Sub-issue #157: Apply epsilon floor to prevent division by zero or NaN
        safe_vol = np.where(np.isnan(vol) | (vol < epsilon), epsilon, vol)
        # Compute raw inverse-volatility for active constituents
        inv_vol = np.zeros(n, dtype=np.float64)
        inv_vol[current_mask] = 1.0 / safe_vol[current_mask]
        sum_inv_vol = np.sum(inv_vol)
        if sum_inv_vol > 0:
            norm_weights = inv_vol / sum_inv_vol
        else:
            norm_weights = np.where(current_mask, 1.0 / n_active, 0.0)
        # Apply concentration defense budget
        if concentration_defense and n_active < 3:
            equity_budget = 1.0 / 3.0
        else:
            equity_budget = 1.0
        weights[idx] = norm_weights * equity_budget
    return weights

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

    def compute_fundamental_quality_scores(
    returns: pd.DataFrame,
    active_mask: np.ndarray,
    lookback_window: int = DEFAULT_LOOKBACK_DAYS,
    skip_recent_days: int = 21,
    epsilon: float = REGULARIZATION_EPSILON
) -> np.ndarray:
        """
        Sub-issue #168: Formulates point-in-time raw fundamental quality scoring vector q_t.
        Combines two empirical quality dimensions strictly from historical data prior to t:

        1. Risk-adjusted profitability/drift proxy: 12-1 month Sharpe/Information ratio.
        2. Stability proxy: Negative trailing 63-day downside volatility.
        """
        ret_values = returns.values
        t, n = ret_values.shape
        quality_matrix = np.zeros((t, n), dtype=np.float64)

        for idx in range(t):
            current_mask = active_mask[idx]
            if not np.any(current_mask):
                continue

        # 1. Risk-Adjusted Drift (t - lookback to t - skip_recent_days)
        end_drift = max(0, idx - skip_recent_days)
        start_drift = max(0, idx - lookback_window)

        if end_drift - start_drift >= 20:
            drift_window = ret_values[start_drift:end_drift]
            mean_ret = np.nanmean(drift_window, axis=0)
            vol_ret = np.nanstd(drift_window, axis=0, ddof=1)
            safe_vol = np.where(np.isnan(vol_ret) | (vol_ret < epsilon), epsilon, vol_ret)
            raw_drift = np.nan_to_num(mean_ret / safe_vol, nan=0.0)
        else:
            raw_drift = np.zeros(n, dtype=np.float64)

        # 2. Stability / Low Downside Volatility (t - 63 to t)
        start_stab = max(0, idx - MIN_OBSERVATIONS_REQUIRED)
        if idx - start_stab >= 10:
            stab_window = ret_values[start_stab:idx]
            downside_returns = np.minimum(np.nan_to_num(stab_window, nan=0.0), 0.0)
            downside_vol = np.sqrt(np.mean(downside_returns ** 2, axis=0))
            raw_stability = -downside_vol
        else:
            raw_stability = np.zeros(n, dtype=np.float64)

        composite_raw = 0.5 * raw_drift + 0.5 * raw_stability
        quality_matrix[idx, current_mask] = composite_raw[current_mask]

        return quality_matrix







class ConvexPortfolioOptimizer:
    """
    Sub-issues #162, #163, #164: Formulates and solves the convex quadratic program (QP)
    for portfolio variance minimization with long-only budget constraints.
    """

    def __init__(
        self,
        solver: str = "CLARABEL",
        verbose: bool = False
    ) -> None:
        self.solver = solver
        self.verbose = verbose

    def construct_optimization_problem(
        self,
        cov_matrix: np.ndarray,
        quality_scores: Optional[np.ndarray] = None,
        lambda_tilt: float = 0.5,
        equity_budget: float = 1.0,
        upper_bound: Optional[float] = None
    ) -> Tuple[cp.Variable, cp.Problem]:
        """
        Sub-issue #162: Constructs convex optimization problem with cp.Variable.
        Sub-issue #163: Defines quadratic variance objective w^T @ Sigma @ w.
        Sub-issue #164: Adds long-only budget constraints (w >= 0, sum(w) == equity_budget).
        Sub-issue #169: Constructs convex optimization problem with cp.Variable, quadratic variance
        objective w^T @ Sigma @ w, linear quality tilt -lambda * q^T @ w, and long-only budget constraints.
        """
        n = cov_matrix.shape[0]
        w = cp.Variable(n, name="weights")
        sigma_psd = cp.psd_wrap(cov_matrix)
        portfolio_variance = cp.quad_form(w, sigma_psd)

        # Sub-issue #169: Add linear quality tilt term -lambda * (q^T @ w)
        if quality_scores is not None and lambda_tilt > 0.0:
            q = np.asarray(quality_scores, dtype=np.float64).flatten()
            if q.shape[0] != n:
                raise ValueError(f"quality_scores length {q.shape[0]} must match covariance dimension {n}")
            objective = cp.Minimize(0.5 * portfolio_variance - lambda_tilt * (q @ w))
        else:
            objective = cp.Minimize(0.5 * portfolio_variance)

        constraints = [
            w >= 0.0,
            cp.sum(w) == equity_budget
        ]
        if upper_bound is not None:
            constraints.append(w <= upper_bound)
        problem = cp.Problem(objective, constraints)
        return w, problem

    def solve(
        self,
        cov_matrix: np.ndarray,
        quality_scores: Optional[np.ndarray] = None,
        lambda_tilt: float = 0.5,
        equity_budget: float = 1.0,
        upper_bound: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Solves the formulated quadratic program with optional quality tilts
        and extracts optimal weights.
        """
        n = cov_matrix.shape[0]
        if n == 0 or equity_budget <= 0.0:
            return np.zeros(n), 0.0
        w, problem = self.construct_optimization_problem(
            cov_matrix=cov_matrix,
            quality_scores=quality_scores,
            lambda_tilt=lambda_tilt,
            equity_budget=equity_budget,
            upper_bound=upper_bound
        )
        try:
            problem.solve(solver=self.solver, verbose=self.verbose)
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and w.value is not None:
                weights = np.clip(w.value, 0.0, 1.0)
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights) * equity_budget
                return weights, float(problem.value)
        except Exception as e:
            logging.warning(f"Solver {self.solver} failed: {e}. Falling back to OSQP.")
            try:
                problem.solve(solver="OSQP", verbose=self.verbose)
                if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and w.value is not None:
                    weights = np.clip(w.value, 0.0, 1.0)
                    if np.sum(weights) > 0:
                        weights = weights / np.sum(weights) * equity_budget
                    return weights, float(problem.value)
            except Exception as e_osqp:
                logging.error(f"Fallback solver OSQP also failed: {e_osqp}.")
        fallback_w = np.full(n, equity_budget / n)
        fallback_var = float(fallback_w.T @ cov_matrix @ fallback_w)
        return fallback_w, fallback_var


if __name__ == "__main__":
    estimator = CovarianceEstimator()
    print("src/optimizer.py initialized with Ledoit-Wolf shrinkage and PSD conditioning.")
