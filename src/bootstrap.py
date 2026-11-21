"""
Non-Parametric Monte Carlo Permutation Bootstrap Engine & Artifact Export Utilities.

- Sub-issues #220-#224: Breadth-matched synthetic S&P pool and deterministic resampling.
- Sub-issues #226-#230: Sharpe percentile rank, Davison-Hinkley empirical p-values, and
  parallel SeedSequence execution.
- Sub-issue #233: Export bootstrap distribution percentiles to data/mc_results.json.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
import os
import numpy as np
from numpy.typing import NDArray
import pandas as pd

BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
TRADING_DAYS_PER_YEAR: int = 252

FloatArray = NDArray[np.float64]


class SyntheticBootstrapEngine:
    """
    Synthetic universe Monte Carlo permutation resampling engine.
    
    Resamples randomized equal-weight portfolios matching exact
    daily active constituent count N_t with deterministic SeedSequence reproducibility.
    """

    DEFAULT_SEED: int = 42

    def __init__(
        self,
        constituent_returns: FloatArray,
        active_counts: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        seed: int = DEFAULT_SEED,
    ) -> None:
        """
        Initializes the synthetic universe resampling engine with a fixed default seed.
        """
        if constituent_returns.ndim != 2:
            raise ValueError("constituent_returns must be a 2D array of shape (T, M).")

        n_sessions, n_assets = constituent_returns.shape
        if active_counts.ndim != 1 or active_counts.shape[0] != n_sessions:
            raise ValueError("active_counts must be a 1D array matching session count T.")

        if np.any(active_counts < 0) or np.any(active_counts > n_assets):
            raise ValueError(
                f"active_counts must lie within [0, {n_assets}], got "
                f"[{int(np.min(active_counts))}, {int(np.max(active_counts))}]."
            )

        self.constituent_returns: FloatArray = np.asarray(constituent_returns, dtype=np.float64)
        self.active_counts: np.ndarray = np.asarray(active_counts, dtype=np.int64)
        self.dates: Optional[pd.DatetimeIndex] = dates
        self.pool_size: int = int(n_assets)
        self.seed: int = int(seed)

    @property
    def n_sessions(self) -> int:
        """Returns number of trading sessions T."""
        return int(self.constituent_returns.shape[0])

    @property
    def equal_weight_pool_returns(self) -> FloatArray:
        """
        Computes the equal-weighted pool return series scaled by active equity budget
        B_t = min(1.0, N_t / 3.0), which equals the theoretical expectation of random
        N_t-stock sampling on every session t.
        """
        budget_t: FloatArray = np.minimum(1.0, self.active_counts / 3.0)
        return budget_t * np.mean(self.constituent_returns, axis=1)

    def create_rng(self, seed: Optional[int] = None) -> np.random.Generator:
        """
        Creates a deterministic numpy PCG64 Generator from an explicit SeedSequence.

        Guarantees bit-for-bit reproducibility across repeated calls
        and platforms without mutating global numpy random state.
        """
        resolved_seed: int = self.seed if seed is None else int(seed)
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence(resolved_seed)))

    def _simulate_breadth_regime(
        self,
        count_k: int,
        n_trials: int,
        child_seed: np.random.SeedSequence,
    ) -> Tuple[np.ndarray, FloatArray]:
        """
        Simulates all sessions sharing active constituent count k using an independent
        spawned SeedSequence stream for thread-safe deterministic reproducibility.
        """
        session_indices = np.flatnonzero(self.active_counts == count_k)
        n_regime_days: int = int(session_indices.shape[0])
        budget_k: float = min(1.0, count_k / 3.0)
        pool_subset: FloatArray = self.constituent_returns[session_indices]

        rng = np.random.Generator(np.random.PCG64(child_seed))
        sampled_idx = rng.integers(
            low=0,
            high=self.pool_size,
            size=(n_regime_days, n_trials, count_k),
        )
        picked_returns: FloatArray = np.take_along_axis(
            pool_subset[:, np.newaxis, :],
            sampled_idx,
            axis=2,
        )
        regime_returns: FloatArray = budget_k * picked_returns.mean(axis=2).T
        return session_indices, regime_returns

    def simulate_random_portfolios(
        self,
        n_trials: int = 10_000,
        seed: Optional[int] = None,
        n_jobs: int = -1,
    ) -> FloatArray:
        """
        Generates B randomized equal-weight portfolios matching exact daily active breadth N_t.
        
        Parallelizes independent breadth-regime permutation loops across
        worker threads using spawned SeedSequence streams so multi-worker execution is
        bit-for-bit identical to single-threaded execution (n_jobs=1).

        """
        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {n_trials}.")

        resolved_seed: int = self.seed if seed is None else int(seed)
        non_zero_counts: List[int] = [
            int(k) for k in np.unique(self.active_counts) if int(k) > 0
        ]
        simulated_returns: FloatArray = np.zeros((n_trials, self.n_sessions), dtype=np.float64)
        if not non_zero_counts:
            return simulated_returns

        child_seeds = np.random.SeedSequence(resolved_seed).spawn(len(non_zero_counts))
        tasks = list(zip(non_zero_counts, child_seeds))

        max_workers: int = (
            max(1, os.cpu_count() or 1) if n_jobs == -1 else max(1, int(n_jobs))
        )
        if max_workers == 1 or len(tasks) == 1:
            results = [
                self._simulate_breadth_regime(k_val, n_trials, c_seed)
                for k_val, c_seed in tasks
            ]
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as executor:
                futures = [
                    executor.submit(self._simulate_breadth_regime, k_val, n_trials, c_seed)
                    for k_val, c_seed in tasks
                ]
                results = [fut.result() for fut in futures]

        for session_indices, regime_returns in results:
            simulated_returns[:, session_indices] = regime_returns

        return simulated_returns

    @classmethod
    def from_sp_benchmark_series(
        cls,
        benchmark_returns: pd.Series,
        active_counts: pd.Series,
        pool_size: int = 500,
        seed: int = DEFAULT_SEED,
    ) -> "SyntheticBootstrapEngine":
        """
        Constructs a calibrated synthetic S&P 500 cross-sectional return pool around an
        observed equal-weight S&P benchmark series (e.g., RSP / SPY) so cross-sectional
        mean returns match the benchmark exactly on every trading day t.
        """
        aligned_df = pd.DataFrame(
            {"bench": benchmark_returns, "n_active": active_counts}
        ).dropna()
        bench_arr: FloatArray = aligned_df["bench"].to_numpy(dtype=np.float64)
        counts_arr: np.ndarray = aligned_df["n_active"].to_numpy(dtype=np.int64)

        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed))))
        eps: FloatArray = rng.normal(loc=0.0, scale=0.015, size=(bench_arr.shape[0], pool_size))
        eps -= np.mean(eps, axis=1, keepdims=True)
        pool_matrix: FloatArray = bench_arr[:, np.newaxis] + eps

        return cls(
            constituent_returns=pool_matrix,
            active_counts=counts_arr,
            dates=pd.DatetimeIndex(aligned_df.index),
            seed=int(seed),
        )


def compute_annualized_sharpe_array(
    returns_matrix: FloatArray,
    risk_free_annual: float = 0.015,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> FloatArray:
    """
    Vectorized annualized Sharpe ratio calculation across (B, T) return paths.
    """
    arr: FloatArray = np.asarray(returns_matrix, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    rf_daily: float = risk_free_annual / float(periods_per_year)
    excess: FloatArray = arr - rf_daily
    mean_excess: FloatArray = np.mean(excess, axis=1)
    std_excess: FloatArray = np.std(excess, axis=1, ddof=1)
    sharpes: FloatArray = np.zeros_like(mean_excess, dtype=np.float64)
    valid = std_excess > 1e-15
    sharpes[valid] = (mean_excess[valid] / std_excess[valid]) * np.sqrt(float(periods_per_year))
    return sharpes


def compute_sharpe_percentile_rank(
    strategy_returns: pd.Series,
    bootstrap_trials: FloatArray,
    risk_free_annual: float = 0.015,
) -> Dict[str, float]:
    """
    Computes the empirical percentile rank of the Black CEO portfolio's Sharpe ratio
    against the synthetic Monte Carlo null Sharpe distribution.

    Returns observed Sharpe, null distribution mean/median/std, and
    empirical percentile rank in [0.0, 100.0].
    """
    strat_arr: FloatArray = np.asarray(strategy_returns.dropna(), dtype=np.float64)
    observed_sharpe: float = float(
        compute_annualized_sharpe_array(strat_arr, risk_free_annual=risk_free_annual)[0]
    )
    null_sharpes: FloatArray = compute_annualized_sharpe_array(
        bootstrap_trials,
        risk_free_annual=risk_free_annual,
    )
    percentile_rank: float = float(np.mean(null_sharpes <= observed_sharpe) * 100.0)
    return {
        "observed_sharpe": observed_sharpe,
        "null_sharpe_mean": float(np.mean(null_sharpes)),
        "null_sharpe_median": float(np.median(null_sharpes)),
        "null_sharpe_std": float(np.std(null_sharpes, ddof=1)),
        "sharpe_percentile_rank": percentile_rank,
        "n_trials": float(null_sharpes.shape[0]),
    }


def extract_empirical_bootstrap_p_value(
    strategy_returns: pd.Series,
    bootstrap_trials: FloatArray,
    metric: str = "sharpe",
    risk_free_annual: float = 0.015,
) -> Dict[str, float]:
    """
    Extracts finite-sample empirical p-values comparing the Black CEO strategy against
    the synthetic Monte Carlo permutation null distribution.
    """
    trials_arr: FloatArray = np.asarray(bootstrap_trials, dtype=np.float64)
    if trials_arr.ndim != 2 or trials_arr.shape[0] < 1:
        raise ValueError("bootstrap_trials must be a non-empty 2D array of shape (B, T).")
    strat_arr: FloatArray = np.asarray(strategy_returns.dropna(), dtype=np.float64)
    metric_key: str = metric.strip().lower()
    if metric_key == "sharpe":
        obs_stat: float = float(
            compute_annualized_sharpe_array(strat_arr, risk_free_annual=risk_free_annual)[0]
        )
        null_stats: FloatArray = compute_annualized_sharpe_array(
            trials_arr,
            risk_free_annual=risk_free_annual,
        )
    elif metric_key == "cagr":
        obs_stat = float(
            np.prod(1.0 + strat_arr) ** (TRADING_DAYS_PER_YEAR / strat_arr.shape[0]) - 1.0
        )
        null_stats = (
            np.prod(1.0 + trials_arr, axis=1)
            ** (TRADING_DAYS_PER_YEAR / trials_arr.shape[1])
            - 1.0
        )
    else:
        raise ValueError(f"Unsupported bootstrap metric '{metric}'. Expected 'sharpe' or 'cagr'.")
    n_trials: int = int(null_stats.shape[0])
    null_mean: float = float(np.mean(null_stats))
    right_tail_count: int = int(np.sum(null_stats >= obs_stat))
    two_sided_count: int = int(
        np.sum(np.abs(null_stats - null_mean) >= abs(obs_stat - null_mean))
    )
    p_right: float = float((1 + right_tail_count) / (n_trials + 1))
    p_two_sided: float = float((1 + two_sided_count) / (n_trials + 1))
    return {
        "observed_stat": obs_stat,
        "null_mean": null_mean,
        "null_p025": float(np.percentile(null_stats, 2.5)),
        "null_p975": float(np.percentile(null_stats, 97.5)),
        "percentile_rank": float(np.mean(null_stats <= obs_stat) * 100.0),
        "p_value_right_tailed": p_right,
        "p_value_two_sided": p_two_sided,
        "n_trials": float(n_trials),
    }


def _summarize_bootstrap_metric(
    strategy_returns: pd.Series,
    bootstrap_trials: FloatArray,
    metric: str,
    risk_free_annual: float = 0.015,
) -> Dict[str, object]:
    """Builds full percentile ladder and empirical p-value summary for one bootstrap metric."""
    inference: Dict[str, float] = extract_empirical_bootstrap_p_value(
        strategy_returns=strategy_returns,
        bootstrap_trials=bootstrap_trials,
        metric=metric,
        risk_free_annual=risk_free_annual,
    )
    strat_arr: FloatArray = np.asarray(strategy_returns.dropna(), dtype=np.float64)
    if metric == "sharpe":
        null_dist: FloatArray = compute_annualized_sharpe_array(
            bootstrap_trials,
            risk_free_annual=risk_free_annual,
        )
    else:
        null_dist = (
            np.prod(1.0 + bootstrap_trials, axis=1)
            ** (TRADING_DAYS_PER_YEAR / bootstrap_trials.shape[1])
            - 1.0
        )

    pct_keys = (1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 97.5, 99.0)
    percentiles: Dict[str, float] = {
        f"p{str(q).replace('.', '_')}": float(np.percentile(null_dist, q))
        for q in pct_keys
    }
    return {
        "observed": inference["observed_stat"],
        "null_mean": inference["null_mean"],
        "null_std": float(np.std(null_dist, ddof=1)),
        "percentile_rank": inference["percentile_rank"],
        "p_value_right_tailed": inference["p_value_right_tailed"],
        "p_value_two_sided": inference["p_value_two_sided"],
        "two_tailed_95_bounds": [inference["null_p025"], inference["null_p975"]],
        "percentiles": percentiles,
        "n_sessions": int(strat_arr.shape[0]),
    }


def export_mc_results_json(
    strategy_series_map: Dict[str, pd.Series],
    bootstrap_trials: FloatArray,
    output_path: Path = DATA_DIR / "mc_results.json",
    seed: int = SyntheticBootstrapEngine.DEFAULT_SEED,
    risk_free_annual: float = 0.015,
) -> Dict[str, object]:
    """
    Exports Monte Carlo bootstrap distribution percentiles, confidence bounds, and
    empirical p-values to data/mc_results.json.

    Generates structured JSON artifacts for statistical auditing and
    Milestone 5 interactive Expo distribution visualizations.
    """
    if not strategy_series_map:
        raise ValueError("strategy_series_map must contain at least one strategy return series.")

    payload: Dict[str, object] = {
        "metadata": {
            "n_trials": int(bootstrap_trials.shape[0]),
            "n_sessions": int(bootstrap_trials.shape[1]),
            "seed": int(seed),
            "risk_free_annual": float(risk_free_annual),
            "p_value_estimator": "(1 + count) / (B + 1)",
        },
        "strategies": {},
    }

    strategies_dict: Dict[str, object] = {}
    for scheme_label, ret_series in strategy_series_map.items():
        strategies_dict[scheme_label] = {
            "sharpe": _summarize_bootstrap_metric(
                ret_series,
                bootstrap_trials,
                metric="sharpe",
                risk_free_annual=risk_free_annual,
            ),
            "cagr": _summarize_bootstrap_metric(
                ret_series,
                bootstrap_trials,
                metric="cagr",
                risk_free_annual=risk_free_annual,
            ),
        }

    payload["strategies"] = strategies_dict
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logging.info("Sub-issue #233: Exported Monte Carlo results to %s", output_path)
    return payload


