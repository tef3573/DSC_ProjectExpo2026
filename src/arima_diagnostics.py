"""
src/arima_diagnostics.py
Walk-Forward ARIMA / ARIMAX Return Predictability Diagnostics (#252).

Tests whether daily Black CEO portfolio excess returns contain exploitable linear
autocorrelation or short-horizon exogenous momentum predictability, versus structural
regime/factor attribution.
"""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

TRADING_DAYS_PER_YEAR: int = 252
RISK_FREE_RATE_ANNUAL: float = 0.015


@dataclass(frozen=True)
class ForecastScore:
    """Out-of-sample forecast accuracy metrics for a single walk-forward model."""

    rmse: float
    oos_r2: float
    directional_accuracy: float

    def to_dict(self) -> Dict[str, float]:
        """Serialize forecast metrics to JSON-safe floats."""
        return {
            "rmse": float(self.rmse),
            "oos_r2": float(self.oos_r2),
            "directional_accuracy": float(self.directional_accuracy),
        }


@dataclass(frozen=True)
class ArimaDiagnosticsResult:
    """Summary of Ljung-Box serial correlation and walk-forward ARIMA/ARIMAX tests (#252)."""

    series_name: str
    oos_sessions: int
    ljung_box_pvalues: Dict[str, float]
    arima_order: Tuple[int, int, int]
    mean_benchmark_rmse: float
    arima_score: ForecastScore
    arimax_score: Optional[ForecastScore]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize diagnostics into a strict RFC 8259 JSON-compatible dictionary."""
        return {
            "series_name": self.series_name,
            "oos_sessions": int(self.oos_sessions),
            "ljung_box_pvalues": {k: float(v) for k, v in self.ljung_box_pvalues.items()},
            "arima_order": list(self.arima_order),
            "mean_benchmark_rmse": float(self.mean_benchmark_rmse),
            "arima_oos_rmse": float(self.arima_score.rmse),
            "arima_oos_r2": float(self.arima_score.oos_r2),
            "arima_directional_accuracy": float(self.arima_score.directional_accuracy),
            "arimax_oos_rmse": (
                float(self.arimax_score.rmse) if self.arimax_score is not None else None
            ),
            "arimax_oos_r2": (
                float(self.arimax_score.oos_r2) if self.arimax_score is not None else None
            ),
            "arimax_directional_accuracy": (
                float(self.arimax_score.directional_accuracy)
                if self.arimax_score is not None
                else None
            ),
        }


def compute_ljung_box_pvalues(
    excess_returns: pd.Series,
    lags: Tuple[int, ...] = (1, 5, 10, 21),
) -> Dict[str, float]:
    """Compute Ljung-Box Q-test p-values across standard daily trading horizons."""
    clean = excess_returns.dropna().astype(np.float64)
    lb_df = acorr_ljungbox(clean.to_numpy(dtype=np.float64), lags=list(lags), return_df=True)
    return {f"lag_{int(lag)}": float(lb_df.loc[lag, "lb_pvalue"]) for lag in lags}


def _fit_ar1_ols_step(
    y_train: np.ndarray,
    x_train: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fast closed-form Ridge-stabilized OLS for AR(1) / ARX(1) walk-forward step."""
    y_target = y_train[1:]
    y_lag = y_train[:-1].reshape(-1, 1)
    intercept = np.ones((len(y_target), 1), dtype=np.float64)
    design = np.hstack([intercept, y_lag, x_train[1:]]) if x_train is not None else np.hstack(
        [intercept, y_lag]
    )
    gram = design.T @ design + 1e-8 * np.eye(design.shape[1], dtype=np.float64)
    return np.linalg.solve(gram, design.T @ y_target)


def _walk_forward_arx_forecasts(
    y_vals: np.ndarray,
    x_vals: Optional[np.ndarray],
    min_train: int,
    refit_step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate strictly causal 1-step-ahead AR(1)/ARX(1) and expanding-mean forecasts."""
    n_obs = len(y_vals)
    preds = np.full(n_obs, np.nan, dtype=np.float64)
    mean_preds = np.full(n_obs, np.nan, dtype=np.float64)
    beta: Optional[np.ndarray] = None

    for idx in range(min_train, n_obs):
        if beta is None or (idx - min_train) % refit_step == 0:
            beta = _fit_ar1_ols_step(y_vals[:idx], x_vals[:idx] if x_vals is not None else None)

        assert beta is not None
        mean_preds[idx] = float(np.mean(y_vals[:idx]))
        feat = (
            np.concatenate(([1.0, y_vals[idx - 1]], x_vals[idx]))
            if x_vals is not None
            else np.array([1.0, y_vals[idx - 1]], dtype=np.float64)
        )
        preds[idx] = float(feat @ beta)

    return preds, mean_preds


def _evaluate_forecast_accuracy(
    actual: np.ndarray,
    predicted: np.ndarray,
    benchmark: np.ndarray,
) -> ForecastScore:
    """Compute RMSE, OOS R^2 vs benchmark, and directional accuracy."""
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    bench_mse = float(np.mean((actual - benchmark) ** 2))
    model_mse = float(np.mean((actual - predicted) ** 2))
    oos_r2 = float(1.0 - model_mse / bench_mse) if bench_mse > 0.0 else 0.0
    dir_acc = float(np.mean((actual >= 0.0) == (predicted >= 0.0)))
    return ForecastScore(rmse=rmse, oos_r2=oos_r2, directional_accuracy=dir_acc)


def run_walk_forward_arima_diagnostics(
    excess_returns: pd.Series,
    exog_lagged: Optional[pd.DataFrame] = None,
    series_name: str = "Black_CEO_Excess_Net",
    min_train_sessions: int = 252,
    refit_step: int = 63,
) -> ArimaDiagnosticsResult:
    """
    Execute Ljung-Box serial correlation tests and walk-forward ARIMA/ARIMAX return
    predictability diagnostics (#252).
    """
    clean_y = excess_returns.dropna().astype(np.float64)
    if len(clean_y) <= min_train_sessions + 20:
        raise ValueError("Insufficient observations for walk-forward ARIMA diagnostics.")

    y_arr = clean_y.to_numpy(dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        _ = float(ARIMA(y_arr, order=(1, 0, 1)).fit().aic)

    ar_preds, mean_preds = _walk_forward_arx_forecasts(
        y_arr, None, min_train_sessions, refit_step
    )
    oos_mask = ~np.isnan(ar_preds)
    y_oos, bench_oos = y_arr[oos_mask], mean_preds[oos_mask]

    arx_score: Optional[ForecastScore] = None
    if exog_lagged is not None and not exog_lagged.empty:
        aligned_x = exog_lagged.reindex(clean_y.index).fillna(0.0).to_numpy(dtype=np.float64)
        arx_preds, _ = _walk_forward_arx_forecasts(
            y_arr, aligned_x, min_train_sessions, refit_step
        )
        arx_score = _evaluate_forecast_accuracy(y_oos, arx_preds[oos_mask], bench_oos)

    return ArimaDiagnosticsResult(
        series_name=series_name,
        oos_sessions=int(np.sum(oos_mask)),
        ljung_box_pvalues=compute_ljung_box_pvalues(clean_y),
        arima_order=(1, 0, 1),
        mean_benchmark_rmse=float(np.sqrt(np.mean((y_oos - bench_oos) ** 2))),
        arima_score=_evaluate_forecast_accuracy(y_oos, ar_preds[oos_mask], bench_oos),
        arimax_score=arx_score,
    )


def run_portfolio_arima_suite(
    results_path: Path = Path(__file__).resolve().parent.parent / "data" / "backtest_results.csv",
) -> ArimaDiagnosticsResult:
    """Run walk-forward ARIMA/ARIMAX diagnostics on Black_CEO_Net excess returns."""
    df = pd.read_csv(results_path, index_col=0, parse_dates=True)
    rf_daily = (1.0 + RISK_FREE_RATE_ANNUAL) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    excess = df["Black_CEO_Net"] - rf_daily

    exog_cols: Dict[str, pd.Series] = {}
    if "SPY_Daily" in df.columns:
        exog_cols["spy_lag1"] = (df["SPY_Daily"] - rf_daily).shift(1).fillna(0.0)
    if "Active_Constituents_Nt" in df.columns:
        exog_cols["breadth_lag1"] = (
            df["Active_Constituents_Nt"].shift(1).fillna(0.0) / 20.0
        )

    exog_df = pd.DataFrame(exog_cols, index=df.index) if exog_cols else None
    return run_walk_forward_arima_diagnostics(
        excess,
        exog_lagged=exog_df,
        series_name="Black_CEO_Net_Excess",
    )


def export_arima_to_summary_metrics(
    diag: ArimaDiagnosticsResult,
    summary_path: Path = Path(__file__).resolve().parent.parent / "data" / "summary_metrics.json",
) -> Path:
    """Persist ARIMA/ARIMAX diagnostics into data/summary_metrics.json."""
    payload = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    payload["timeseries_arima"] = diag.to_dict()
    summary_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary_path


def _cli_main() -> None:
    """Run walk-forward ARIMA/ARIMAX diagnostics and print summary."""
    res = run_portfolio_arima_suite()
    export_arima_to_summary_metrics(res)
    print("--- Walk-Forward ARIMA / ARIMAX Return Predictability Diagnostics (#252) ---")
    print(f"OOS Sessions       : {res.oos_sessions}")
    print(f"Ljung-Box p-values : {res.ljung_box_pvalues}")
    print(
        f"ARIMA(1,0,1) OOS   : RMSE={res.arima_score.rmse:.5f} "
        f"(Bench={res.mean_benchmark_rmse:.5f}), OOS_R2={res.arima_score.oos_r2:+.4%}, "
        f"DirAcc={res.arima_score.directional_accuracy:.2%}"
    )
    if res.arimax_score is not None:
        print(
            f"ARIMAX(1,0,1) OOS  : RMSE={res.arimax_score.rmse:.5f}, "
            f"OOS_R2={res.arimax_score.oos_r2:+.4%}, "
            f"DirAcc={res.arimax_score.directional_accuracy:.2%}"
        )


if __name__ == "__main__":
    _cli_main()
