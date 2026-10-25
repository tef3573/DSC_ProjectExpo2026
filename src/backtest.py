"""
Vectorized Backtesting Engine for Black CEO Portfolio
-----------------------------------------------------
Calculates daily 1/N dynamic allocation returns with:
  1. Cash drag substitution (Rf,t) when Nt == 0
  2. Weight drift tracking and 10 bps turnover friction deduction
  3. Comprehensive risk diagnostics: Sharpe, Sortino, Max Drawdown, Calmar
  4. Benchmark comparison against SPY (Cap-Weight) and RSP (Equal-Weight)
"""

import json
import os
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from src.universe import create_point_in_time_mask

# Output and parameter constants
RESULTS_CSV_PATH = "data/backtest_results.csv"
METRICS_JSON_PATH = "data/summary_metrics.json"
ANNUAL_RF = 0.03            # Baseline 3.0% annualized risk-free rate
TRANSACTION_FEE_BPS = 10     # 10 bps = 0.10% = 0.0010


def compute_risk_metrics(return_series: pd.Series, rf_annual: float = ANNUAL_RF) -> Dict[str, float]:
    """
    Computes annualized institutional risk-adjusted return metrics.
    """
    clean_returns = return_series.dropna()
    n_days = len(clean_returns)
    if n_days == 0:
        return {}

    # Daily compounding risk-free rate
    rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0
    excess_returns = clean_returns - rf_daily

    # Cumulative growth curve and drawdown
    cum_returns = (1.0 + clean_returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = float(drawdown.min())

    # Annualized CAGR
    cagr = float(cum_returns.iloc[-1] ** (252.0 / n_days) - 1.0)

    # Annualized Volatility
    ann_vol = float(clean_returns.std() * np.sqrt(252.0))

    # Sharpe Ratio
    sharpe = float((excess_returns.mean() / clean_returns.std()) * np.sqrt(252.0)) if clean_returns.std() > 0 else 0.0

    # Downside semi-deviation (Sortino)
    downside_diff = excess_returns[excess_returns < 0.0]
    downside_vol = float(np.sqrt((downside_diff ** 2).mean()) * np.sqrt(252.0)) if len(downside_diff) > 0 else 0.0
    sortino = float((excess_returns.mean() * np.sqrt(252.0)) / downside_vol) if downside_vol > 0 else 0.0

    # Calmar Ratio
    calmar = float(cagr / abs(max_drawdown)) if max_drawdown < 0 else 0.0

    return {
        "CAGR": cagr,
        "Annualized_Volatility": ann_vol,
        "Sharpe_Ratio": sharpe,
        "Sortino_Ratio": sortino,
        "Max_Drawdown": max_drawdown,
        "Calmar_Ratio": calmar
    }


def run_vectorized_backtest() -> Tuple[pd.DataFrame, Dict]:
    """
    Executes the daily rebalanced 1/N point-in-time backtest with friction.
    """
    print("[INFO] Loading prices and constructing point-in-time mask...")
    stock_prices, mask, benchmarks = create_point_in_time_mask()

    # Calculate raw daily returns for all universe assets
    asset_returns = stock_prices.pct_change().fillna(0.0)

    # Calculate daily compounding risk-free rate
    rf_daily = (1.0 + ANNUAL_RF) ** (1.0 / 252.0) - 1.0

    # Initialize tracking arrays
    dates = stock_prices.index
    num_assets = stock_prices.shape[1]
    net_strategy_returns = []
    gross_strategy_returns = []
    turnover_series = []
    active_counts = []

    # Initialize prior allocation weight vector (starting 100% in cash buffer)
    prev_weights = np.zeros(num_assets)
    friction_coefficient = TRANSACTION_FEE_BPS / 10000.0  # 10 bps -> 0.0010

    print("[INFO] Simulating 1/N daily rebalancing with 10 bps friction...")
    for t_idx, current_date in enumerate(dates):
        # Boolean vector of eligible assets today
        active_mask_vec = mask.iloc[t_idx].values.astype(bool)
        n_active = int(active_mask_vec.sum())
        active_counts.append(n_active)

        # Asset returns on date t
        ret_vec = asset_returns.iloc[t_idx].values

        if n_active > 0:
            # Target 1/N weight for each active constituent
            target_weights = np.zeros(num_assets)
            target_weights[active_mask_vec] = 1.0 / n_active

            # Gross return is inner product of target weights and returns
            gross_ret = float(np.dot(target_weights, ret_vec))
        else:
            # Zero active constituents -> allocate 100% to risk-free cash buffer
            target_weights = np.zeros(num_assets)
            gross_ret = float(rf_daily)

        # Calculate weight drift from t-1 returns
        if t_idx == 0:
            turnover = float(np.sum(np.abs(target_weights)))
        else:
            # Drifted prior weights: w_{i, t-1} * (1 + R_{i, t-1})
            prior_ret = asset_returns.iloc[t_idx - 1].values
            drifted_weights = prev_weights * (1.0 + prior_ret)
            drift_sum = drifted_weights.sum()
            if drift_sum > 0:
                drifted_weights = drifted_weights / drift_sum

            # Portfolio turnover: 0.5 * sum(|w_target - w_drifted|)
            turnover = float(0.5 * np.sum(np.abs(target_weights - drifted_weights)))

        # Net return after 10 bps friction fee on turnover
        friction_cost = friction_coefficient * turnover
        net_ret = gross_ret - friction_cost

        # Record values
        gross_strategy_returns.append(gross_ret)
        net_strategy_returns.append(net_ret)
        turnover_series.append(turnover)
        prev_weights = target_weights

    # Benchmark returns
    bm_returns = benchmarks.pct_change().fillna(0.0)

    # Compile unified backtest DataFrame
    results_df = pd.DataFrame(
        {
            "Black_CEO_Gross": gross_strategy_returns,
            "Black_CEO_Net": net_strategy_returns,
            "SPY": bm_returns["SPY"],
            "RSP": bm_returns["RSP"],
            "Turnover": turnover_series,
            "Active_Count": active_counts
        },
        index=dates
    ).dropna()

    # Ensure output directory exists and export CSV
    os.makedirs(os.path.dirname(RESULTS_CSV_PATH), exist_ok=True)
    results_df.to_csv(RESULTS_CSV_PATH)
    print(f"[SUCCESS] Exported backtest time series to {RESULTS_CSV_PATH}")

    # Compute risk profiles across strategies and benchmarks
    summary_metrics = {
        "Black_CEO_Net": compute_risk_metrics(results_df["Black_CEO_Net"]),
        "Black_CEO_Gross": compute_risk_metrics(results_df["Black_CEO_Gross"]),
        "SPY": compute_risk_metrics(results_df["SPY"]),
        "RSP": compute_risk_metrics(results_df["RSP"]),
    }

    # Save summary JSON for frontend ingestion
    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(summary_metrics, f, indent=4)
    print(f"[SUCCESS] Exported scorecard metrics to {METRICS_JSON_PATH}\n")

    # Display comparison table
    display_df = pd.DataFrame(summary_metrics)
    
   # Format for clean console printing
    formatted_df = display_df.astype(object).copy()
    formatted_df.loc["CAGR"] = formatted_df.loc["CAGR"].apply(
        lambda x: f"{x:.2%}" if pd.notnull(x) else "--"
    )
    formatted_df.loc["Annualized_Volatility"] = formatted_df.loc[
        "Annualized_Volatility"
    ].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "--")
    formatted_df.loc["Sharpe_Ratio"] = formatted_df.loc["Sharpe_Ratio"].apply(
        lambda x: f"{x:.2f}" if pd.notnull(x) else "--"
    )
    formatted_df.loc["Sortino_Ratio"] = formatted_df.loc[
        "Sortino_Ratio"
    ].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "--")
    formatted_df.loc["Max_Drawdown"] = formatted_df.loc["Max_Drawdown"].apply(
        lambda x: f"{x:.2%}" if pd.notnull(x) else "--"
    )
    formatted_df.loc["Calmar_Ratio"] = formatted_df.loc["Calmar_Ratio"].apply(
        lambda x: f"{x:.2f}" if pd.notnull(x) else "--"
    )

    print("=" * 68)
    print("      QUANTITATIVE RISK-ADJUSTED PERFORMANCE (2000 - 2026)")
    print("=" * 68)
    print(formatted_df.to_string())
    print("=" * 68)

    return results_df, summary_metrics


if __name__ == "__main__":
    run_vectorized_backtest()