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

    sample_cov = (x_centered.T @ x_centered) / (t - 1)

    mu_var = np.trace(sample_cov) / n
    if shrinkage_target is None:
        target = mu_var * np.eye(n)
    else:
        target = shrinkage_target

    gamma = float(np.sum((sample_cov - target) ** 2))

    if gamma == 0.0:
        return sample_cov, 0.0

    y = np.zeros((t, n * n))
    for idx in range(t):
        outer = np.outer(x_centered[idx], x_centered[idx])
        y[idx] = outer.flatten()

    y_mean = np.mean(y, axis=0)
    phi_mat = np.sum((y - y_mean) ** 2, axis=0) / t
    phi = float(np.sum(phi_mat))

    kappa = phi / gamma
    delta = float(np.clip(kappa / t, 0.0, 1.0))

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
    cov_sym = 0.5 * (cov_matrix + cov_matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_sym)
    clipped_eigenvalues = np.clip(np.nan_to_num(eigenvalues, nan=epsilon), epsilon, None)
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

    if active_mask.ndim == 1:
        n_active = int(np.sum(active_mask))
        weights = np.zeros_like(active_mask, dtype=np.float64)
        if n_active == 0:
            return weights
        equity_budget = (1.0 / 3.0) if (concentration_defense and n_active < 3) else 1.0
        weights[active_mask] = equity_budget / n_active
        return weights
    elif active_mask.ndim == 2:
        n_active = np.sum(active_mask, axis=1, keepdims=True)
        safe_n = np.where(n_active > 0, n_active, 1.0)
        if concentration_defense:
            equity_budget = np.where(
                n_active == 0,
                0.0,
                np.where(n_active < 3, 1.0 / 3.0, 1.0)
            )
        else:
            equity_budget = np.where(n_active > 0, 1.0, 0.0)
        return np.where(active_mask, equity_budget / safe_n, 0.0)
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
        if len(window_returns) > 1:
            vol = np.nanstd(window_returns, axis=0, ddof=1)
        else:
            vol = np.zeros(n)
        safe_vol = np.where(np.isnan(vol) | (vol < epsilon), epsilon, vol)
        inv_vol = np.zeros(n, dtype=np.float64)
        inv_vol[current_mask] = 1.0 / safe_vol[current_mask]
        sum_inv_vol = np.sum(inv_vol)
        if sum_inv_vol > 0:
            norm_weights = inv_vol / sum_inv_vol
        else:
            norm_weights = np.where(current_mask, 1.0 / n_active, 0.0)
        equity_budget = (1.0 / 3.0) if (concentration_defense and n_active < 3) else 1.0
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


def normalize_quality_scores(
    raw_scores: np.ndarray,
    active_mask: np.ndarray,
    clip_bound: float = 3.0
) -> np.ndarray:
    """
    Sub-issue #170: Cross-sectional z-score normalization of fundamental quality scores
    across active peers.
    """
    scores = np.asarray(raw_scores, dtype=np.float64).flatten()
    mask = np.asarray(active_mask, dtype=bool).flatten()
    n = len(scores)
    z_scores = np.zeros(n, dtype=np.float64)
    active_indices = np.where(mask)[0]

    if len(active_indices) < 2:
        return z_scores

    active_vals = scores[active_indices]
    mean_val = np.nanmean(active_vals)
    std_val = np.nanstd(active_vals, ddof=1)

    if np.isnan(std_val) or std_val < 1e-6:
        return z_scores

    norm_vals = (active_vals - mean_val) / std_val
    clipped_vals = np.clip(norm_vals, -clip_bound, clip_bound)
    z_scores[active_indices] = clipped_vals
    return z_scores


def compute_fundamental_quality_scores(
    returns: pd.DataFrame,
    active_mask: np.ndarray,
    lookback_window: int = DEFAULT_LOOKBACK_DAYS,
    skip_recent_days: int = 21,
    epsilon: float = REGULARIZATION_EPSILON
) -> np.ndarray:
    """
    Sub-issues #168 & #170: Formulates point-in-time fundamental quality scoring vector q_t,
    cross-sectionally z-score normalized across active peers.
    """
    ret_values = returns.values
    t, n = ret_values.shape
    quality_matrix = np.zeros((t, n), dtype=np.float64)

    for idx in range(t):
        current_mask = active_mask[idx]
        if np.sum(current_mask) < 2:
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

        z_drift = normalize_quality_scores(raw_drift, current_mask)
        z_stability = normalize_quality_scores(raw_stability, current_mask)

        composite_raw = 0.5 * z_drift + 0.5 * z_stability
        quality_matrix[idx] = normalize_quality_scores(composite_raw, current_mask)

    return quality_matrix

class ConvexPortfolioOptimizer:
    """
    Sub-issues #162, #163, #164, #169, #174, #175, #176:
    Formulates and solves the convex quadratic program (QP) for portfolio variance
    minimization with optional linear quality tilts, dynamic concentration caps,
    graceful inverse-volatility fallback, and parameterized problem caching for
    high-speed rolling execution.
    """
    def __init__(
        self,
        solver: str = "CLARABEL",
        verbose: bool = False
    ) -> None:
        self.solver = solver
        self.verbose = verbose
        # Sub-issue #176: Cache compiled parameterized cvxpy problems keyed by
        # (n_assets, has_tilt, has_upper_bound) to eliminate repeated canonicalization
        self._problem_cache: dict = {}

    @staticmethod
    def compute_dynamic_upper_cap(n_active: int, max_cap: float = 0.35) -> float:
        """
        Sub-issue #174: Computes the dynamic single-asset upper weight cap:
            w_max = min(0.35, 2 / N_t)
        """
        if n_active <= 0:
            return 0.0
        return float(min(max_cap, 2.0 / n_active))

    def construct_optimization_problem(
        self,
        cov_matrix: np.ndarray,
        quality_scores: Optional[np.ndarray] = None,
        lambda_tilt: float = 0.5,
        equity_budget: float = 1.0,
        upper_bound: Optional[float] = None,
        use_dynamic_cap: bool = True
    ) -> Tuple[cp.Variable, cp.Problem]:
        """
        Sub-issues #162, #163, #164, #169, #174:
        Constructs a standalone convex optimization problem (used for structural inspection
        and single-shot validation).
        """
        n = cov_matrix.shape[0]
        w = cp.Variable(n, name="weights")
        sigma_psd = cp.psd_wrap(cov_matrix)
        portfolio_variance = cp.quad_form(w, sigma_psd)
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
        effective_upper = upper_bound
        if effective_upper is None and use_dynamic_cap and n > 0:
            effective_upper = self.compute_dynamic_upper_cap(n_active=n)
        if effective_upper is not None:
            constraints.append(w <= effective_upper)
        problem = cp.Problem(objective, constraints)
        return w, problem

    def _get_or_build_cached_problem(
        self,
        n: int,
        has_tilt: bool,
        has_upper_bound: bool
    ) -> dict:
        """
        Sub-issue #176: Builds or retrieves a pre-canonicalized cp.Problem backed by
        cp.Parameter instances and Cholesky factor L (where Sigma = L^T L) so repeated
        solves across rolling windows bypass symbolic AST construction.
        """
        cache_key = (n, has_tilt, has_upper_bound)
        if cache_key in self._problem_cache:
            return self._problem_cache[cache_key]
        w = cp.Variable(n, name="weights")
        chol_param = cp.Parameter((n, n), name="chol_factor")
        budget_param = cp.Parameter(nonneg=True, name="equity_budget")
        portfolio_variance = cp.sum_squares(chol_param @ w)
        if has_tilt:
            tilt_param = cp.Parameter(n, name="scaled_quality")
            objective = cp.Minimize(0.5 * portfolio_variance - tilt_param @ w)
        else:
            tilt_param = None
            objective = cp.Minimize(0.5 * portfolio_variance)
        constraints = [
            w >= 0.0,
            cp.sum(w) == budget_param
        ]
        if has_upper_bound:
            upper_param = cp.Parameter(nonneg=True, name="upper_bound")
            constraints.append(w <= upper_param)
        else:
            upper_param = None
        problem = cp.Problem(objective, constraints)
        entry = {
            "w": w,
            "problem": problem,
            "chol_param": chol_param,
            "tilt_param": tilt_param,
            "budget_param": budget_param,
            "upper_param": upper_param,
        }
        self._problem_cache[cache_key] = entry
        return entry

    def _inverse_volatility_fallback(
        self,
        cov_matrix: np.ndarray,
        equity_budget: float
    ) -> Tuple[np.ndarray, float]:
        """
        Sub-issue #175: Deterministic inverse-volatility fallback when QP solvers
        are infeasible or fail to converge.
        """
        n = cov_matrix.shape[0]
        variances = np.diag(cov_matrix)
        vols = np.sqrt(np.maximum(variances, 0.0))
        safe_vols = np.where(
            np.isnan(vols) | (vols < REGULARIZATION_EPSILON),
            REGULARIZATION_EPSILON,
            vols
        )

        inv_vols = 1.0 / safe_vols
        sum_inv = float(np.sum(inv_vols))

        if sum_inv > 0.0 and np.isfinite(sum_inv):
            weights = (inv_vols / sum_inv) * equity_budget
        else:
            weights = np.full(n, equity_budget / n, dtype=np.float64)

        portfolio_var = float(weights.T @ cov_matrix @ weights)
        return weights, portfolio_var

    def solve(
        self,
        cov_matrix: np.ndarray,
        quality_scores: Optional[np.ndarray] = None,
        lambda_tilt: float = 0.5,
        equity_budget: float = 1.0,
        upper_bound: Optional[float] = None,
        use_dynamic_cap: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        Sub-issues #174, #175, #176: Solves the parameterized QP using cached problem
        canonicalization and warm-starting, with graceful inverse-volatility fallback.
        """
        n = cov_matrix.shape[0]
        if n == 0 or equity_budget <= 0.0:
            return np.zeros(n), 0.0

        has_tilt = quality_scores is not None and lambda_tilt > 0.0
        if has_tilt:
            q = np.asarray(quality_scores, dtype=np.float64).flatten()
            if q.shape[0] != n:
                raise ValueError(f"quality_scores length {q.shape[0]} must match covariance dimension {n}")

        effective_upper = upper_bound
        if effective_upper is None and use_dynamic_cap and n > 0:
            effective_upper = self.compute_dynamic_upper_cap(n_active=n)
        has_upper_bound = effective_upper is not None

        # Pre-check immediate infeasibility before invoking solver (#175 + #176 fast-path)
        if has_upper_bound and (n * effective_upper < equity_budget - 1e-9):
            logging.warning(
                f"Infeasible upper bound {effective_upper:.4f} for N={n}, budget={equity_budget:.4f}. "
                "Using inverse-vol fallback."
            )
            return self._inverse_volatility_fallback(cov_matrix, equity_budget)

        # Compute upper Cholesky factor L such that L^T @ L == Sigma
        try:
            cov_sym = 0.5 * (cov_matrix + cov_matrix.T)
            chol_upper = np.linalg.cholesky(cov_sym).T
        except np.linalg.LinAlgError:
            psd_cov = condition_covariance_matrix(cov_matrix)
            chol_upper = np.linalg.cholesky(psd_cov).T

        cached = self._get_or_build_cached_problem(
            n=n,
            has_tilt=has_tilt,
            has_upper_bound=has_upper_bound
        )
        w = cached["w"]
        problem = cached["problem"]
        cached["chol_param"].value = chol_upper
        cached["budget_param"].value = float(equity_budget)
        if has_tilt:
            cached["tilt_param"].value = lambda_tilt * q
        if has_upper_bound:
            cached["upper_param"].value = float(effective_upper)

        # 1. Primary Solver: CLARABEL (warm-started)
        try:
            problem.solve(solver=self.solver, verbose=self.verbose, warm_start=True)
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and w.value is not None:
                weights = np.clip(w.value, 0.0, 1.0)
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights) * equity_budget
                return weights, float(problem.value)
            logging.warning(f"Solver {self.solver} returned status '{problem.status}'. Trying OSQP.")
        except Exception as e:
            logging.warning(f"Solver {self.solver} raised exception: {e}. Trying OSQP.")

        # 2. Secondary Solver: OSQP (warm-started)
        try:
            problem.solve(solver="OSQP", verbose=self.verbose, warm_start=True)
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and w.value is not None:
                weights = np.clip(w.value, 0.0, 1.0)
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights) * equity_budget
                return weights, float(problem.value)
            logging.warning(f"Fallback solver OSQP returned status '{problem.status}'. Using inverse-vol fallback.")
        except Exception as e_osqp:
            logging.error(f"Fallback solver OSQP raised exception: {e_osqp}. Using inverse-vol fallback.")

        # 3. Deterministic Fallback: Inverse-Volatility Allocation (#175)
        return self._inverse_volatility_fallback(cov_matrix, equity_budget)

    
if __name__ == "__main__":
    estimator = CovarianceEstimator()
    print("src/optimizer.py initialized with Ledoit-Wolf shrinkage and PSD conditioning.")
