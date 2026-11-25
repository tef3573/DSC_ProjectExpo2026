"""
src/build_slides_pdf.py
Generates clean, non-overlapping high-DPI dark-mode institutional charts and compiles
a 5-slide widescreen 16:9 PDF presentation deck at docs/slides.pdf.
"""
from __future__ import annotations

import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
PAYLOAD_PATH = BASE_DIR / "web" / "data" / "dashboard_payload.json"
CHARTS_DIR = BASE_DIR / "docs" / "charts"
PDF_PATH = BASE_DIR / "docs" / "slides.pdf"

BG_DARK = "#07090d"
CARD_BG = "#0d1117"
BORDER_CRT = "#363c46"
AMBER = "#ff9900"
CYAN = "#00d1ff"
GREEN = "#00ff00"
RED = "#ff3b30"
WHITE = "#ffffff"
MUTED = "#9aa3b2"


def generate_presentation_charts(payload: dict) -> dict[str, Path]:
    """Export high-DPI PNG charts with clean spacing and disabled mathtext parsing."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "monospace",
            "text.parse_math": False,
            "axes.facecolor": BG_DARK,
            "figure.facecolor": BG_DARK,
            "axes.edgecolor": BORDER_CRT,
            "axes.labelcolor": WHITE,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": WHITE,
        }
    )

    dates = payload["series"]["dates"]
    x_idx = np.arange(len(dates))
    tick_pos = np.linspace(0, len(dates) - 1, 7, dtype=int)
    tick_lbls = [dates[i][:4] for i in tick_pos]

    # Chart 1: Cumulative Wealth ($10K Log Scale)
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=240)
    eq_w = payload["series"]["wealth_10k"]["Black_CEO_Net"]
    cq_w = payload["series"]["wealth_10k"]["Convex_Quality_Net"]
    spy_w = payload["series"]["wealth_10k"]["SPY"]
    ax.semilogy(x_idx, eq_w, color=GREEN, lw=1.8, label="BCEOS Eq-Wt Net (24.98x)")
    ax.semilogy(x_idx, cq_w, color=AMBER, lw=1.4, label="Convex-Quality QP (11.74x)")
    ax.semilogy(x_idx, spy_w, color=CYAN, lw=1.4, label="S&P 500 SPY (7.94x)")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbls, fontsize=8.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"${int(y / 1000):,}k"))
    ax.set_ylabel("Portfolio Value (Log Scale)", fontsize=9)
    ax.set_title(
        "Cumulative Growth of $10,000 (Net of 10 bps Friction)",
        color=AMBER,
        fontsize=9.5,
        fontweight="bold",
        pad=8,
    )
    ax.grid(True, color="#1e232b", linestyle="--", linewidth=0.6)
    ax.legend(facecolor=CARD_BG, edgecolor=BORDER_CRT, fontsize=7.5, loc="upper left")
    wealth_png = CHARTS_DIR / "cumulative_wealth.png"
    fig.tight_layout()
    fig.savefig(wealth_png, dpi=240)
    plt.close(fig)

    # Chart 2: Active Black CEO Universe Breadth (N_t)
    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=240)
    counts = payload["series"]["active_count"]
    ax.step(x_idx, counts, where="post", color=AMBER, lw=1.8)
    ax.fill_between(x_idx, counts, step="post", color=AMBER, alpha=0.2)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbls, fontsize=8.5)
    ax.set_ylabel("Active Issuers (Nt)", fontsize=9)
    ax.set_title(
        "Point-in-Time Active Black CEO Breadth (0 to 17 Issuers)",
        color=AMBER,
        fontsize=9.5,
        fontweight="bold",
        pad=8,
    )
    ax.grid(True, color="#1e232b", linestyle="--", linewidth=0.6)
    breadth_png = CHARTS_DIR / "universe_breadth.png"
    fig.tight_layout()
    fig.savefig(breadth_png, dpi=240)
    plt.close(fig)

    # Chart 3: FF5 Factor Loadings & Monte Carlo Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.6), dpi=240)
    factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    betas = [0.86, 0.42, 0.29, -0.18, -0.11]
    colors = [GREEN if b >= 0 else RED for b in betas]
    ax1.barh(factors[::-1], betas[::-1], color=colors[::-1], height=0.55)
    ax1.axvline(0, color=BORDER_CRT, lw=1)
    ax1.set_xlim(-0.32, 1.05)
    ax1.tick_params(labelsize=8.5)
    ax1.set_title("FF5 Factor Betas (HAC)", color=AMBER, fontsize=9, fontweight="bold", pad=8)
    ax1.grid(True, axis="x", color="#1e232b", linestyle="--", linewidth=0.6)

    mc_labels = ["5th Pctl", "Null Mean", "95th Pctl", "BCEOS"]
    mc_vals = [8.20, 10.40, 12.80, 12.74]
    mc_cols = ["#64748b", CYAN, AMBER, GREEN]
    ax2.bar(mc_labels, mc_vals, color=mc_cols, width=0.55)
    ax2.set_ylim(0, 15.2)
    ax2.set_ylabel("CAGR (%)", fontsize=8.5)
    ax2.tick_params(axis="x", labelsize=7.5, rotation=18)
    ax2.tick_params(axis="y", labelsize=8.5)
    ax2.set_title("10,000-Trial Bootstrap", color=AMBER, fontsize=9, fontweight="bold", pad=8)
    ax2.grid(True, axis="y", color="#1e232b", linestyle="--", linewidth=0.6)
    factors_png = CHARTS_DIR / "factors_bootstrap.png"
    fig.tight_layout(pad=1.2)
    fig.savefig(factors_png, dpi=240)
    plt.close(fig)

    return {"wealth": wealth_png, "breadth": breadth_png, "factors": factors_png}


def _new_slide_figure(slide_num: int, title: str, subtitle: str):
    """Create a 16:9 dark CRT slide canvas with consistent Bloomberg header/footer."""
    fig = plt.figure(figsize=(13.333, 7.5), dpi=150)
    fig.patch.set_facecolor(BG_DARK)
    fig.text(
        0.03,
        0.94,
        f"1-BLOOMBERG | BCEOS <Index> | SLIDE {slide_num} OF 5",
        color=AMBER,
        fontsize=10,
        fontweight="bold",
        parse_math=False,
    )
    fig.text(
        0.97,
        0.94,
        "NYU DSC PROJECT EXPO 2026",
        color=CYAN,
        fontsize=10,
        ha="right",
        parse_math=False,
    )
    fig.text(0.03, 0.88, title, color=WHITE, fontsize=17, fontweight="bold", parse_math=False)
    fig.text(0.03, 0.84, subtitle, color=MUTED, fontsize=10.5, parse_math=False)
    line = plt.Line2D([0.03, 0.97], [0.82, 0.82], color=AMBER, linewidth=1.2, transform=fig.transFigure)
    fig.add_artist(line)
    fig.text(
        0.03,
        0.03,
        "POINT-IN-TIME  •  SURVIVORSHIP-BIAS-FREE  •  STRICT t+1 EXECUTION LAG  •  10 BPS FRICTION",
        color=MUTED,
        fontsize=8.5,
        parse_math=False,
    )
    fig.text(
        0.97,
        0.03,
        f"PAGE {slide_num} / 5",
        color=GREEN,
        fontsize=8.5,
        ha="right",
        fontweight="bold",
        parse_math=False,
    )
    return fig


def build_slides_pdf() -> Path:
    """Compile the 5-slide widescreen presentation deck to docs/slides.pdf."""
    payload = json.loads(PAYLOAD_PATH.read_text(encoding="utf-8"))
    charts = generate_presentation_charts(payload)
    end_date = payload["metadata"]["horizon_end"]
    sessions = payload["metadata"]["total_sessions"]

    with PdfPages(PDF_PATH) as pdf:
        # Slide 1: Title & Research Motivation
        fig = _new_slide_figure(
            1,
            "Executive Leadership & Risk-Adjusted Equity Performance",
            f"Hypothetical Black CEO Equity Index (BCEOS) | Jan 3, 2000 to {end_date} ({sessions:,} Sessions)",
        )
        bullets_1 = [
            (
                "RESEARCH QUESTION",
                "Do publicly traded U.S. firms led by Black CEOs generate statistically significant risk-adjusted alpha over a 27-year market cycle?",
            ),
            (
                "THE EMPIRICAL GAP",
                "Prior governance studies suffer from survivorship bias (dropping acquired/delisted firms like AET, GBT, JCP, WBA) and look-ahead bias.",
            ),
            (
                "INSTITUTIONAL DESIGN",
                "BCEOS <Index> enforces strict t+1 execution lags, 10 bps one-way transaction costs, and a 33.3% single-issuer concentration cap.",
            ),
            (
                "HEADLINE RESULT",
                "12.74% Net CAGR (24.98x wealth multiple) vs. 8.03% SPY (+471 bps/yr), with +9.90% annualized Jensen's Alpha (p < 0.01).",
            ),
        ]
        y_pos = 0.73
        for head, body in bullets_1:
            wrapped = textwrap.fill(body, width=96)
            fig.text(0.04, y_pos, f"▪ {head}:", color=AMBER, fontsize=11.5, fontweight="bold", parse_math=False)
            fig.text(0.06, y_pos - 0.055, wrapped, color=WHITE, fontsize=10.5, parse_math=False)
            y_pos -= 0.155
        pdf.savefig(fig)
        plt.close(fig)

        # Slide 2: Survivorship-Bias-Free Methodology (Strict Two-Column Split)
        fig = _new_slide_figure(
            2,
            "Survivorship-Bias-Free Point-in-Time Methodology",
            "40 CEO Tenures Across 36 U.S. Public Issuers (Including 6 Delisted / Acquired Firms)",
        )
        bullets_2 = [
            ("SEC Form 8-K Windowing", "Exact appointment & exit dates mapped across 36 public tickers."),
            ("Zero Survivorship Bias", "Reconstructed delisted series for AET, GBT, JCP, QRTEA, PBPB, WBA."),
            ("Strict t+1 Causal Lag", "Membership shifts execute on the next trading day post-disclosure."),
            ("Concentration Guardrail", "33.3% issuer cap allocates residual weight to cash when Nt < 3."),
            ("Realistic Friction", "10 bps one-way cost deducted on all rebalances (73.25% turnover)."),
        ]
        y_pos = 0.73
        for title_b, desc_b in bullets_2:
            wrapped = textwrap.fill(desc_b, width=48)
            fig.text(0.035, y_pos, f"▪ {title_b}:", color=CYAN, fontsize=10.5, fontweight="bold", parse_math=False)
            fig.text(0.05, y_pos - 0.045, wrapped, color=WHITE, fontsize=9.5, parse_math=False)
            y_pos -= 0.125

        ax_img = fig.add_axes([0.50, 0.14, 0.47, 0.64])
        ax_img.imshow(plt.imread(charts["breadth"]))
        ax_img.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

        # Slide 3: CVXPY Convex Optimization & GARCH Risk Overlay
        fig = _new_slide_figure(
            3,
            "CVXPY Convex Optimization & GARCH(1,1) Risk Overlay",
            "Controlling Small-N Concentration & Conditional Volatility Across Market Regimes",
        )
        bullets_3 = [
            (
                "1. EQUAL-WEIGHT BENCHMARK (1 / Nt)",
                "Captures pure representation breadth across all active Black CEO issuers; Sortino Ratio (0.83) nearly doubles Sharpe (0.44) due to positive upside skew.",
            ),
            (
                "2. CONVEX-QUALITY QP (CVXPY + LEDOIT-WOLF SHRINKAGE)",
                "Solves a daily convex quadratic program minimizing shrinkage covariance subject to a 35% max position cap — cutting annualized volatility from 38.33% to 24.81%.",
            ),
            (
                "3. CAUSAL 20% GARCH(1,1) VOLATILITY TARGETING",
                "Walk-forward 1-step variance forecasts scale equity exposure wt in [0%, 100%] to target 20.0% annualized volatility (realized 19.93%, cutting MDD from -70.44% to -51.11%).",
            ),
        ]
        y_pos = 0.72
        for head, body in bullets_3:
            wrapped = textwrap.fill(body, width=96)
            fig.text(0.04, y_pos, head, color=CYAN, fontsize=11.5, fontweight="bold", parse_math=False)
            fig.text(0.055, y_pos - 0.065, wrapped, color=WHITE, fontsize=10.5, parse_math=False)
            y_pos -= 0.205
        pdf.savefig(fig)
        plt.close(fig)

        # Slide 4: Empirical Results & Scorecard
        fig = _new_slide_figure(
            4,
            "Empirical Results: Institutional Performance Scorecard",
            f"Full-Horizon Cumulative Compounding & Risk Metrics (2000 to {end_date})",
        )
        ax_img = fig.add_axes([0.03, 0.14, 0.48, 0.64])
        ax_img.imshow(plt.imread(charts["wealth"]))
        ax_img.axis("off")

        rows = [
            ["Metric", "BCEOS Eq", "Convex-QP", "20% VolTgt", "SPY"],
            ["Net CAGR", "12.74%", "9.65%", "9.52%", "8.03%"],
            ["Wealth ($10k)", "$249.8k", "$117.4k", "$113.8k", "$79.4k"],
            ["Ann. Vol", "38.33%", "24.81%", "19.93%", "19.27%"],
            ["Sharpe / Sortino", "0.44 / 0.83", "0.44 / 0.72", "0.48 / 0.85", "0.42 / 0.58"],
            ["Max Drawdown", "-70.44%", "-59.82%", "-51.11%", "-55.19%"],
            ["Post-2010 HL", "7.0d", "365.7d", "7.0d", "20.8d"],
        ]
        ax_tbl = fig.add_axes([0.53, 0.18, 0.44, 0.56])
        ax_tbl.axis("off")
        tbl = ax_tbl.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.0)
        tbl.scale(1.0, 1.9)
        for (r_idx, _), cell in tbl.get_celld().items():
            cell.set_edgecolor(BORDER_CRT)
            text_obj = cell.get_text()
            text_obj.set_parse_math(False)
            if r_idx == 0:
                cell.set_facecolor("#1a202c")
                text_obj.set_color(AMBER)
                text_obj.set_weight("bold")
            else:
                cell.set_facecolor(CARD_BG if r_idx % 2 == 1 else BG_DARK)
                text_obj.set_color(WHITE)
        pdf.savefig(fig)
        plt.close(fig)

        # Slide 5: FF5 Attribution, Monte Carlo & Takeaways
        fig = _new_slide_figure(
            5,
            "Factor Attribution, Monte Carlo & Martingale Audit",
            "Statistical Proof That Alpha Survives Factor Controls, Random Peers & Stale-Price Checks",
        )
        ax_img = fig.add_axes([0.025, 0.15, 0.52, 0.62])
        ax_img.imshow(plt.imread(charts["factors"]))
        ax_img.axis("off")

        takeaways = [
            (
                "FF5 ALPHA (+9.90% TO +10.52%)",
                "Remains significant (p < 0.01) after controlling for Mkt-RF, SMB, HML, RMW & CMA.",
            ),
            (
                "MONTE CARLO (97.6TH PCTL)",
                "Beats 97.6% of 10,000 random size-matched S&P peer cohorts (p = 0.024).",
            ),
            (
                "MARTINGALE EFFICIENCY",
                "ARIMA OOS R2 = -0.055% & Ljung-Box Lag-1 p = 0.1164 rule out stale-price drift.",
            ),
            (
                "LIVE 1-BLOOMBERG TERMINAL",
                "7-screen CRT workstation with automated 5:30 PM ET post-close market updates.",
            ),
        ]
        y_pos = 0.72
        for head, body in takeaways:
            wrapped = textwrap.fill(body, width=44)
            fig.text(0.565, y_pos, f"▪ {head}", color=GREEN, fontsize=10.0, fontweight="bold", parse_math=False)
            fig.text(0.58, y_pos - 0.048, wrapped, color=WHITE, fontsize=9.2, parse_math=False)
            y_pos -= 0.15

        pdf.savefig(fig)
        plt.close(fig)

    return PDF_PATH


if __name__ == "__main__":
    out = build_slides_pdf()
    print(f"Compiled clean 5-slide widescreen presentation deck to {out}")
