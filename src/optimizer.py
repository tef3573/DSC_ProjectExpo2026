"""
src/optimizer.py
Convex Quality Optimization & Covariance Estimation Module.
Milestone 3: Convex Quality Optimization (cvxpy)
- Sub-issue #149: Initialize src/optimizer.py module structure.
- Sub-issue #150: Implement Ledoit-Wolf shrinkage covariance estimator.
"""
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import logging
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

TRADING_DAYS_PER_YEAR = 252
DEFAULT_LOOKBACK_DAYS = 252
MIN_OBSERVATIONS_REQUIRED = 63
REGULARIZATION_EPSILON = 1e-4


class CovarianceEstimator:
    """
    Estimator class for asset return covariance matrices using Ledoit-Wolf shrinkage.
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

    def estimate_covariance(
        self,
        returns: pd.DataFrame,
        annualize: bool = True
    ) -> Tuple[pd.DataFrame, float]:
        """
        Sub-issue #150: Computes Ledoit-Wolf shrinkage covariance matrix.
        
        Args:
            returns: Daily percentage return matrix (T x N)
            annualize: Scale daily covariance by 252 if True
            
        Returns:
            Tuple of (Covariance DataFrame (N x N), shrinkage intensity delta in [0, 1])
        """
        clean_returns = returns.dropna(axis=0, how="all").dropna(axis=1, how="all")
        n_obs, n_assets = clean_returns.shape

        if n_obs < self.min_observations:
            logging.warning(
                "Sample size T=%d is below minimum threshold %d. Shrinkage may have higher variance.",
                n_obs, self.min_observations
            )

        if n_assets == 0:
            raise ValueError("No active assets provided for covariance estimation.")

        if n_assets == 1:
            var_daily = float(np.var(clean_returns.iloc[:, 0], ddof=1))
            var_out = var_daily * (TRADING_DAYS_PER_YEAR if annualize else 1.0)
            cov_df = pd.DataFrame(
                [[var_out]],
                index=clean_returns.columns,
                columns=clean_returns.columns
            )
            return cov_df, 0.0

        # Fit Ledoit-Wolf optimal shrinkage
        lw = LedoitWolf(assume_centered=False)
        lw.fit(clean_returns.values)

        cov_matrix = lw.covariance_
        shrinkage_intensity = float(lw.shrinkage_)

        if annualize:
            cov_matrix = cov_matrix * TRADING_DAYS_PER_YEAR

        cov_df = pd.DataFrame(
            cov_matrix,
            index=clean_returns.columns,
            columns=clean_returns.columns
        )
        return cov_df, shrinkage_intensity


if __name__ == "__main__":
    estimator = CovarianceEstimator()
    np.random.seed(42)
    fake_rets = pd.DataFrame(
        np.random.normal(0.0005, 0.015, size=(252, 4)),
        columns=["AAPL", "MSFT", "AMZN", "GOOG"]
    )
    cov, delta = estimator.estimate_covariance(fake_rets)
    print(f"Ledoit-Wolf Shrinkage Intensity (delta): {delta:.4f}")
    print("\nEstimated Annualized Covariance Matrix:\n", cov.round(5))
