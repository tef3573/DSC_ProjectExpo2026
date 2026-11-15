"""
src/optimizer.py
Convex Quality Optimization & Covariance Estimation Module.

Milestone 3: Convex Quality Optimization (cvxpy)
- Sub-issue #149: Initialize src/optimizer.py module structure.
- Sub-issue #150: Implement Ledoit-Wolf shrinkage covariance estimator.
- Sub-issue #151: Condition covariance matrix for low-sample historical windows.
- Sub-issue #188: Standardize PEP 484 / numpy.typing type hints and Google-style docstrings.
- Sub-issue #189: Enforce PEP 8 and pylint compliance across optimizer and backtest engines.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple, TypedDict
import logging

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"

TRADING_DAYS_PER_YEAR: int = 252
DEFAULT_LOOKBACK_DAYS: int = 252
MIN_OBSERVATIONS_REQUIRED: int = 63
REGULARIZATION_EPSILON: float = 1e-4

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
ProblemCacheKey = Tuple[int, bool, bool]


class CachedProblemEntry(TypedDict):
    """Typed dictionary storing a pre-canonicalized CVXPY problem and its parameters."""

    w: cp.Variable
    problem: cp.Problem
    chol_param: cp.Parameter
    tilt_param: Optional[cp.Parameter]
    budget_param: cp.Parameter
    upper_param: Optional[cp.Parameter]


def ledoit_wolf_shrinkage(
    returns: pd.DataFrame,
    shrinkage_target: Optional[FloatArray] = None
) -> Tuple[FloatArray, float]:
    """
    Computes the analytical Ledoit-Wolf linear shrinkage covariance estimator.

    Shrinks sample covariance S toward structured target F (defaulting to scaled identity):
        Sigma_shrunk = delta * F + (1 - delta) * S

    Args:
        returns: Trailing asset return DataFrame of shape (T, N).
        shrinkage_target: Optional custom (N, N) target covariance matrix F.

    Returns:
        Tuple (shrunk_cov, delta) of the regularized covariance matrix and shrinkage intensity.
    """
    x: FloatArray = np.asarray(returns.values, dtype=np.float64)
    t, n = x.shape

    if t < 2 or n == 0:
        eye: FloatArray = np.eye(max(n, 1), dtype=np.float64) * REGULARIZATION_EPSILON
        return eye, 1.0

    x_centered: FloatArray = np.nan_to_num(x - np.nanmean(x, axis=0), nan=0.0)
    sample_cov: FloatArray = (x_centered.T @ x_centered) / (t - 1)

    target: FloatArray = (
        (float(np.trace(sample_cov) / n)) * np.eye(n, dtype=np.float64)
        if shrinkage_target is None
        else np.asarray(shrinkage_target, dtype=np.float64)
    )

    gamma: float = float(np.sum((sample_cov - target) ** 2))
    if gamma == 0.0:
        return sample_cov, 0.0

    y: FloatArray = np.einsum("ti,tj->tij", x_centered, x_centered).reshape(t, n * n)
    phi: float = float(np.sum(np.mean((y - np.mean(y, axis=0)) ** 2, axis=0)))
    delta: float = float(np.clip((phi / gamma) / t, 0.0, 1.0))

    shrunk_cov: FloatArray = delta * target + (1.0 - delta) * sample_cov
    return shrunk_cov, delta


def condition_covariance_matrix(
    cov_matrix: FloatArray,
    epsilon: float = REGULARIZATION_EPSILON
) -> FloatArray:
    """
    Enforces strict symmetry and positive semi-definiteness (PSD) via spectral flooring.

    Args:
        cov_matrix: Input (N, N) symmetric or near-symmetric covariance matrix.
        epsilon: Minimum strictly positive eigenvalue threshold.

    Returns:
        Conditioned (N, N) symmetric positive-definite covariance matrix.
    """
    cov_sym: FloatArray = 0.5 * (cov_matrix + cov_matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_sym)
    clipped_eigenvalues: FloatArray = np.clip(
        np.nan_to_num(eigenvalues, nan=epsilon), epsilon, None
    )
    psd_cov: FloatArray = eigenvectors @ np.diag(clipped_eigenvalues) @ eigenvectors.T
    return 0.5 * (psd_cov + psd_cov.T)


def compute_equal_weights(
    active_mask: BoolArray,
    concentration_defense: bool = True
) -> FloatArray:
    """
    Computes dynamic 1 / N_t equal target weights across active constituents.

    Args:
        active_mask: 1D boolean vector (N,) or 2D boolean matrix (T, N).
        concentration_defense: Whether to enforce the sparse-universe N_t / 3 equity cap.

    Returns:
        Floating-point target weight array matching active_mask.shape.

    Raises:
        ValueError: If active_mask is neither 1D nor 2D.
    """
    mask_arr: BoolArray = np.asarray(active_mask, dtype=bool)

    if mask_arr.ndim == 1:
        n_active: int = int(np.sum(mask_arr))
        weights: FloatArray = np.zeros_like(mask_arr, dtype=np.float64)
        if n_active == 0:
            return weights
        equity_budget: float = (1.0 / 3.0) if (concentration_defense and n_active < 3) else 1.0
        weights[mask_arr] = equity_budget / n_active
        return weights

    if mask_arr.ndim == 2:
        n_active_2d = np.sum(mask_arr, axis=1, keepdims=True)
        safe_n = np.where(n_active_2d > 0, n_active_2d, 1.0)
        if concentration_defense:
            equity_budget_2d = np.where(
                n_active_2d == 0,
                0.0,
                np.where(n_active_2d < 3, 1.0 / 3.0, 1.0)
            )
        else:
            equity_budget_2d = np.where(n_active_2d > 0, 1.0, 0.0)
        return np.where(mask_arr, equity_budget_2d / safe_n, 0.0)

    raise ValueError(f"active_mask must be 1D or 2D, got shape {mask_arr.shape}")


def compute_inverse_volatility_weights(
    returns: pd.DataFrame,
    active_mask: BoolArray,
    lookback_window: int = MIN_OBSERVATIONS_REQUIRED,
    epsilon: float = REGULARIZATION_EPSILON,
    concentration_defense: bool = True
) -> FloatArray:
    """
    Computes rolling inverse-volatility risk parity weights across active constituents.

    Args:
        returns: Daily asset return DataFrame of shape (T, N).
        active_mask: 2D boolean membership matrix of shape (T, N).
        lookback_window: Trailing window length in trading sessions for sample volatility.
        epsilon: Minimum volatility floor guarding against zero-variance division.
        concentration_defense: Whether to cap total equity at 1/3 when N_t < 3.

    Returns:
        2D target equity weight matrix of shape (T, N).
    """
    ret_values: FloatArray = np.asarray(returns.values, dtype=np.float64)
    t, n = ret_values.shape
    weights: FloatArray = np.zeros((t, n), dtype=np.float64)

    for idx in range(t):
        current_mask: BoolArray = active_mask[idx]
        n_active: int = int(np.sum(current_mask))
        if n_active == 0:
            continue

        window_returns: FloatArray = ret_values[max(0, idx - lookback_window + 1):idx + 1]
        vol: FloatArray = (
            np.nanstd(window_returns, axis=0, ddof=1)
            if len(window_returns) > 1
            else np.zeros(n, dtype=np.float64)
        )
        safe_vol: FloatArray = np.where(np.isnan(vol) | (vol < epsilon), epsilon, vol)
        inv_vol: FloatArray = np.where(current_mask, 1.0 / safe_vol, 0.0)
        sum_inv: float = float(np.sum(inv_vol))

        norm_weights: FloatArray = (
            inv_vol / sum_inv
            if sum_inv > 0.0
            else np.where(current_mask, 1.0 / n_active, 0.0)
        )
        equity_budget: float = (1.0 / 3.0) if (concentration_defense and n_active < 3) else 1.0
        weights[idx] = norm_weights * equity_budget

    return weights


class CovarianceEstimator:  # pylint: disable=too-few-public-methods
    """
    Rolling covariance matrix estimator combining Ledoit-Wolf shrinkage and PSD conditioning.

    Attributes:
        lookback_window: Default trailing window length in trading days.
        min_observations: Minimum sample count before relying on full shrinkage estimate.
        epsilon: Minimum eigenvalue floor for PSD conditioning.
    """

    def __init__(
        self,
        lookback_window: int = DEFAULT_LOOKBACK_DAYS,
        min_observations: int = MIN_OBSERVATIONS_REQUIRED,
        epsilon: float = REGULARIZATION_EPSILON
    ) -> None:
        """Initializes the CovarianceEstimator with window and regularization settings."""
        self.lookback_window: int = lookback_window
        self.min_observations: int = min_observations
        self.epsilon: float = epsilon

    def estimate(self, returns: pd.DataFrame) -> Tuple[FloatArray, float]:
        """
        Estimates, shrinks, and spectrally conditions the return covariance matrix.

        Args:
            returns: Trailing asset return DataFrame of shape (W, N).

        Returns:
            Tuple (psd_cov, delta) of the strictly PSD covariance matrix and shrinkage intensity.
        """
        shrunk_cov, delta = ledoit_wolf_shrinkage(returns)
        psd_cov: FloatArray = condition_covariance_matrix(shrunk_cov, epsilon=self.epsilon)
        return psd_cov, delta


def normalize_quality_scores(
    raw_scores: FloatArray,
    active_mask: BoolArray,
    clip_bound: float = 3.0
) -> FloatArray:
    """
    Cross-sectionally standardizes raw quality metrics across active peers to z-scores.

    Args:
        raw_scores: 1D array of raw cross-sectional quality signals (N,).
        active_mask: 1D boolean mask indicating active constituents (N,).
        clip_bound: Symmetric Winsorization ceiling applied to standardized z-scores.

    Returns:
        1D array of Winsorized cross-sectional z-scores of shape (N,).
    """
    scores: FloatArray = np.asarray(raw_scores, dtype=np.float64).flatten()
    mask: BoolArray = np.asarray(active_mask, dtype=bool).flatten()
    z_scores: FloatArray = np.zeros(len(scores), dtype=np.float64)
    active_indices: NDArray[np.intp] = np.where(mask)[0]

    if len(active_indices) < 2:
        return z_scores

    active_vals: FloatArray = scores[active_indices]
    mean_val: float = float(np.nanmean(active_vals))
    std_val: float = float(np.nanstd(active_vals, ddof=1))

    if np.isnan(std_val) or std_val < 1e-6:
        return z_scores

    z_scores[active_indices] = np.clip((active_vals - mean_val) / std_val, -clip_bound, clip_bound)
    return z_scores


def compute_fundamental_quality_scores(
    returns: pd.DataFrame,
    active_mask: BoolArray,
    lookback_window: int = DEFAULT_LOOKBACK_DAYS,
    skip_recent_days: int = 21,
    epsilon: float = REGULARIZATION_EPSILON
) -> FloatArray:
    """
    Computes rolling point-in-time composite quality z-scores q_t across active peers.

    Args:
        returns: Daily asset return DataFrame of shape (T, N).
        active_mask: 2D boolean membership matrix of shape (T, N).
        lookback_window: Trailing lookback horizon W for medium-term drift estimation.
        skip_recent_days: Most recent trading days excluded from drift to avoid reversal.
        epsilon: Minimum volatility floor preventing division by zero.

    Returns:
        2D quality score matrix Q in R^(T x N) standardized per session.
    """
    ret_values: FloatArray = np.asarray(returns.values, dtype=np.float64)
    t, n = ret_values.shape
    quality_matrix: FloatArray = np.zeros((t, n), dtype=np.float64)

    for idx in range(t):
        current_mask: BoolArray = active_mask[idx]
        if np.sum(current_mask) < 2:
            continue

        end_drift: int = max(0, idx - skip_recent_days)
        start_drift: int = max(0, idx - lookback_window)
        if end_drift - start_drift >= 20:
            drift_slice: FloatArray = ret_values[start_drift:end_drift]
            vol_ret: FloatArray = np.nanstd(drift_slice, axis=0, ddof=1)
            safe_vol: FloatArray = np.where(np.isnan(vol_ret) | (vol_ret < epsilon), epsilon, vol_ret)
            raw_drift: FloatArray = np.nan_to_num(np.nanmean(drift_slice, axis=0) / safe_vol, nan=0.0)
        else:
            raw_drift = np.zeros(n, dtype=np.float64)

        start_stab: int = max(0, idx - MIN_OBSERVATIONS_REQUIRED)
        if idx - start_stab >= 10:
            downside: FloatArray = np.minimum(np.nan_to_num(ret_values[start_stab:idx], nan=0.0), 0.0)
            raw_stability: FloatArray = -np.sqrt(np.mean(downside ** 2, axis=0))
        else:
            raw_stability = np.zeros(n, dtype=np.float64)

        composite: FloatArray = (
            0.5 * normalize_quality_scores(raw_drift, current_mask)
            + 0.5 * normalize_quality_scores(raw_stability, current_mask)
        )
        quality_matrix[idx] = normalize_quality_scores(composite, current_mask)

    return quality_matrix


class ConvexPortfolioOptimizer:
    """
    Convex quadratic programming (QP) portfolio optimizer with quality tilts and dynamic caps.

    Attributes:
        solver: Primary CVXPY solver backend (default "CLARABEL").
        verbose: Whether to emit detailed solver diagnostic logs.
    """

    def __init__(
        self,
        solver: str = "CLARABEL",
        verbose: bool = False
    ) -> None:
        """Initializes the optimizer and its parameterized problem cache."""
        self.solver: str = solver
        self.verbose: bool = verbose
        self._problem_cache: Dict[ProblemCacheKey, CachedProblemEntry] = {}

    @staticmethod
    def compute_dynamic_upper_cap(n_active: int, max_cap: float = 0.35) -> float:
        """
        Computes the dynamic single-asset upper weight ceiling w_max = min(0.35, 2 / N_t).

        Args:
            n_active: Number of active constituents N_t.
            max_cap: Hard institutional single-issuer ceiling (default 0.35).

        Returns:
            Upper bound per active asset in [0.0, max_cap].
        """
        if n_active <= 0:
            return 0.0
        return float(min(max_cap, 2.0 / n_active))

    def construct_optimization_problem(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        cov_matrix: FloatArray,
        quality_scores: Optional[FloatArray] = None,
        lambda_tilt: float = 0.5,
        equity_budget: float = 1.0,
        upper_bound: Optional[float] = None,
        use_dynamic_cap: bool = True
    ) -> Tuple[cp.Variable, cp.Problem]:
        """
        Constructs a standalone symbolic CVXPY optimization problem for inspection or testing.

        Args:
            cov_matrix: Conditioned (N, N) positive-definite covariance matrix.
            quality_scores: Optional 1D cross-sectional quality vector q in R^N.
            lambda_tilt: Linear quality tilt weight lambda_tilt >= 0.
            equity_budget: Total target equity allocation B_t in [0, 1].
            upper_bound: Optional explicit per-asset upper weight bound.
            use_dynamic_cap: Whether to apply min(0.35, 2/N) when upper_bound is None.

        Returns:
            Tuple (w, problem) containing the CVXPY decision variable and problem instance.

        Raises:
            ValueError: If quality_scores length does not match cov_matrix dimension.
        """
        n: int = cov_matrix.shape[0]
        w: cp.Variable = cp.Variable(n, name="weights")
        portfolio_variance = cp.quad_form(w, cp.psd_wrap(cov_matrix))

        if quality_scores is not None and lambda_tilt > 0.0:
            q: FloatArray = np.asarray(quality_scores, dtype=np.float64).flatten()
            if q.shape[0] != n:
                raise ValueError(
                    f"quality_scores length {q.shape[0]} must match covariance dimension {n}"
                )
            objective = cp.Minimize(0.5 * portfolio_variance - lambda_tilt * (q @ w))
        else:
            objective = cp.Minimize(0.5 * portfolio_variance)

        constraints = [w >= 0.0, cp.sum(w) == equity_budget]
        effective_upper: Optional[float] = upper_bound
        if effective_upper is None and use_dynamic_cap and n > 0:
            effective_upper = self.compute_dynamic_upper_cap(n_active=n)
        if effective_upper is not None:
            constraints.append(w <= effective_upper)

        return w, cp.Problem(objective, constraints)

    def _get_or_build_cached_problem(
        self,
        n: int,
        has_tilt: bool,
        has_upper_bound: bool
    ) -> CachedProblemEntry:
        """
        Retrieves or builds a pre-canonicalized CVXPY problem backed by cp.Parameter objects.

        Args:
            n: Active universe dimension N_t.
            has_tilt: Whether the linear quality tilt term is active.
            has_upper_bound: Whether an explicit per-asset upper cap constraint is active.

        Returns:
            CachedProblemEntry dictionary containing the problem and parameter handles.
        """
        cache_key: ProblemCacheKey = (n, has_tilt, has_upper_bound)
        if cache_key in self._problem_cache:
            return self._problem_cache[cache_key]

        w: cp.Variable = cp.Variable(n, name="weights")
        chol_param: cp.Parameter = cp.Parameter((n, n), name="chol_factor")
        budget_param: cp.Parameter = cp.Parameter(nonneg=True, name="equity_budget")
        portfolio_variance = cp.sum_squares(chol_param @ w)

        tilt_param: Optional[cp.Parameter] = (
            cp.Parameter(n, name="scaled_quality") if has_tilt else None
        )
        objective = (
            cp.Minimize(0.5 * portfolio_variance - tilt_param @ w)
            if tilt_param is not None
            else cp.Minimize(0.5 * portfolio_variance)
        )

        constraints = [w >= 0.0, cp.sum(w) == budget_param]
        upper_param: Optional[cp.Parameter] = (
            cp.Parameter(nonneg=True, name="upper_bound") if has_upper_bound else None
        )
        if upper_param is not None:
            constraints.append(w <= upper_param)

        entry: CachedProblemEntry = {
            "w": w,
            "problem": cp.Problem(objective, constraints),
            "chol_param": chol_param,
            "tilt_param": tilt_param,
            "budget_param": budget_param,
            "upper_param": upper_param,
        }
        self._problem_cache[cache_key] = entry
        return entry

    def _inverse_volatility_fallback(
        self,
        cov_matrix: FloatArray,
        equity_budget: float
    ) -> Tuple[FloatArray, float]:
        """
        Computes a deterministic inverse-volatility allocation when the QP is infeasible.

        Args:
            cov_matrix: (N, N) covariance matrix whose diagonal provides asset variances.
            equity_budget: Target total equity weight B_t.

        Returns:
            Tuple (weights, portfolio_var) with normalized weights and realized variance.
        """
        n: int = cov_matrix.shape[0]
        vols: FloatArray = np.sqrt(np.maximum(np.diag(cov_matrix), 0.0))
        safe_vols: FloatArray = np.where(
            np.isnan(vols) | (vols < REGULARIZATION_EPSILON),
            REGULARIZATION_EPSILON,
            vols
        )

        inv_vols: FloatArray = 1.0 / safe_vols
        sum_inv: float = float(np.sum(inv_vols))
        weights: FloatArray = (
            (inv_vols / sum_inv) * equity_budget
            if sum_inv > 0.0 and np.isfinite(sum_inv)
            else np.full(n, equity_budget / n, dtype=np.float64)
        )
        return weights, float(weights.T @ cov_matrix @ weights)

    @staticmethod
    def _compute_cholesky_upper(cov_matrix: FloatArray) -> FloatArray:
        """Computes upper Cholesky factor L such that L.T @ L == Sigma."""
        try:
            cov_sym: FloatArray = 0.5 * (cov_matrix + cov_matrix.T)
            return np.linalg.cholesky(cov_sym).T
        except np.linalg.LinAlgError:
            psd_cov: FloatArray = condition_covariance_matrix(cov_matrix)
            return np.linalg.cholesky(psd_cov).T

    def _try_solve_backend(
        self,
        problem: cp.Problem,
        w: cp.Variable,
        solver_name: str,
        equity_budget: float
    ) -> Optional[Tuple[FloatArray, float]]:
        """Attempts to solve the parameterized problem with a single CVXPY solver backend."""
        try:
            problem.solve(solver=solver_name, verbose=self.verbose, warm_start=True)
            if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and w.value is not None:
                weights: FloatArray = np.clip(w.value, 0.0, 1.0)
                if np.sum(weights) > 0.0:
                    weights = weights / np.sum(weights) * equity_budget
                return weights, float(problem.value)
            logging.warning(
                "Solver %s returned status '%s'.", solver_name, problem.status
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logging.warning("Solver %s raised exception: %s.", solver_name, exc)
        return None

    def solve(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        cov_matrix: FloatArray,
        quality_scores: Optional[FloatArray] = None,
        lambda_tilt: float = 0.5,
        equity_budget: float = 1.0,
        upper_bound: Optional[float] = None,
        use_dynamic_cap: bool = True
    ) -> Tuple[FloatArray, float]:
        """
        Solves the quality-tilted minimum-variance QP with cached canonicalization and fallback.

        Args:
            cov_matrix: (N, N) positive-definite asset covariance matrix.
            quality_scores: Optional 1D cross-sectional quality z-score vector (N,).
            lambda_tilt: Quality tilt multiplier lambda_tilt >= 0.
            equity_budget: Target total equity exposure B_t in [0, 1].
            upper_bound: Optional explicit per-asset upper weight limit.
            use_dynamic_cap: Whether to enforce min(0.35, 2/N) when upper_bound is None.

        Returns:
            Tuple (weights, objective_value) where weights sums to equity_budget.

        Raises:
            ValueError: If quality_scores length does not match cov_matrix dimension.
        """
        n: int = cov_matrix.shape[0]
        if n == 0 or equity_budget <= 0.0:
            return np.zeros(n, dtype=np.float64), 0.0

        has_tilt: bool = quality_scores is not None and lambda_tilt > 0.0
        q: FloatArray = np.zeros(n, dtype=np.float64)
        if has_tilt:
            q = np.asarray(quality_scores, dtype=np.float64).flatten()
            if q.shape[0] != n:
                raise ValueError(
                    f"quality_scores length {q.shape[0]} must match covariance dimension {n}"
                )

        effective_upper: Optional[float] = upper_bound
        if effective_upper is None and use_dynamic_cap:
            effective_upper = self.compute_dynamic_upper_cap(n_active=n)
        has_upper_bound: bool = effective_upper is not None

        if (
            has_upper_bound
            and effective_upper is not None
            and (n * effective_upper < equity_budget - 1e-9)
        ):
            logging.warning(
                "Infeasible upper bound %.4f for N=%d, budget=%.4f. Using inverse-vol fallback.",
                effective_upper,
                n,
                equity_budget,
            )
            return self._inverse_volatility_fallback(cov_matrix, equity_budget)

        cached: CachedProblemEntry = self._get_or_build_cached_problem(
            n=n, has_tilt=has_tilt, has_upper_bound=has_upper_bound
        )
        cached["chol_param"].value = self._compute_cholesky_upper(cov_matrix)
        cached["budget_param"].value = float(equity_budget)
        if has_tilt and cached["tilt_param"] is not None:
            cached["tilt_param"].value = lambda_tilt * q
        if has_upper_bound and effective_upper is not None and cached["upper_param"] is not None:
            cached["upper_param"].value = float(effective_upper)

        for solver_name in (self.solver, "OSQP"):
            solution = self._try_solve_backend(
                cached["problem"], cached["w"], solver_name, equity_budget
            )
            if solution is not None:
                return solution

        return self._inverse_volatility_fallback(cov_matrix, equity_budget)


if __name__ == "__main__":
    estimator = CovarianceEstimator()
    print("src/optimizer.py initialized with Ledoit-Wolf shrinkage and PSD conditioning.")
