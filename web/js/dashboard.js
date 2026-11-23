(function () {
  "use strict";

  const REGIME_WINDOWS = {
    full_horizon: null,
    post_2010: { start: "2010-01-04", end: "2026-12-31" },
    pre_2010: { start: "2000-01-03", end: "2009-12-31" },
    dot_com_bust: { start: "2000-03-10", end: "2002-10-09" },
    gfc_2008: { start: "2007-10-09", end: "2009-03-09" },
    zirp_qe_era: { start: "2009-03-10", end: "2020-02-19" },
    covid_shock_stimulus: { start: "2020-02-20", end: "2021-12-31" },
    inflation_rate_hikes: { start: "2022-01-03", end: "2023-10-27" },
    ai_high_rate_regime: { start: "2023-10-30", end: "2026-12-31" },
  };

  const state = {
    payload: null,
    strategy: "Black_CEO_Net",
    regime: "full_horizon",
    scale: "log",
    search: "",
    statusFilter: "ALL",
  };

  function buildSparklineSvg(points, isPositive) {
    if (!Array.isArray(points) || points.length < 2) {
      return '<span class="exec-sub">&mdash;</span>';
    }
    const w = 96;
    const h = 26;
    const min = Math.min(...points);
    const max = Math.max(...points);
    const span = max - min || 1;
    const coords = points
      .map((val, idx) => {
        const x = ((idx / (points.length - 1)) * w).toFixed(1);
        const y = (h - 3 - ((val - min) / span) * (h - 6)).toFixed(1);
        return `${x},${y}`;
      })
      .join(" ");
    const stroke = isPositive ? "#00e676" : "#ff5252";
    return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-hidden="true">
      <polyline fill="none" stroke="${stroke}" stroke-width="1.75" points="${coords}" />
    </svg>`;
  }

  function renderMacroTable(payload) {
    const tbody = document.getElementById("macro-regime-tbody");
    const regimes = payload?.timeseries_diagnostics?.macro_regimes;
    if (!tbody || !Array.isArray(regimes) || regimes.length === 0) {
      return;
    }
    tbody.innerHTML = regimes
      .map((r) => {
        const stratRet = Number(r.strategy_cum_return_pct || 0);
        const spyRet = Number(r.spy_cum_return_pct || 0);
        const rec = r.recovery_days !== null && r.recovery_days !== undefined ? `${r.recovery_days}d` : "Unrec.";
        return `<tr>
          <td>${r.regime_name}</td>
          <td class="num">${r.nt_min}–${r.nt_max}</td>
          <td class="num ${stratRet >= 0 ? "num--pos" : "num--neg"}">${stratRet >= 0 ? "+" : ""}${stratRet.toFixed(2)}%</td>
          <td class="num ${spyRet >= 0 ? "num--pos" : "num--neg"}">${spyRet >= 0 ? "+" : ""}${spyRet.toFixed(2)}%</td>
          <td class="num">${Number(r.max_drawdown_pct || 0).toFixed(2)}%</td>
          <td class="num">${rec}</td>
          <td class="num">${Number(r.downside_capture || 0).toFixed(2)}&times;</td>
        </tr>`;
      })
      .join("");
  }

  function renderRosterTable(payload) {
    const tbody = document.getElementById("roster-tbody");
    if (!tbody || !Array.isArray(payload?.ceo_dossiers)) {
      return;
    }
    const query = state.search.trim().toLowerCase();
    const filtered = payload.ceo_dossiers.filter((d) => {
      if (state.statusFilter !== "ALL" && d.status !== state.statusFilter) {
        return false;
      }
      if (!query) {
        return true;
      }
      return (
        d.ticker.toLowerCase().includes(query) ||
        d.ceo_name.toLowerCase().includes(query) ||
        d.company_name.toLowerCase().includes(query) ||
        d.sector.toLowerCase().includes(query)
      );
    });

    tbody.innerHTML = filtered
      .map((d) => {
        const statusCls =
          d.status === "ACTIVE"
            ? "status-pill--active"
            : d.status === "DELISTED"
              ? "status-pill--delisted"
              : "status-pill--exited";
        const retVal = d.tenure_return_pct;
        const retText = retVal !== null && retVal !== undefined ? `${retVal >= 0 ? "+" : ""}${retVal.toFixed(2)}%` : "—";
        const retCls = retVal !== null && retVal >= 0 ? "num--pos" : "num--neg";
        const peakText = d.peak_price !== null ? `$${d.peak_price.toFixed(2)}` : "—";
        const startText = d.start_price !== null ? `$${d.start_price.toFixed(2)}` : "—";

        return `<tr>
          <td>
            <div class="exec-cell">
              <span class="exec-avatar" style="background:${d.avatar.bg};color:${d.avatar.fg}">${d.avatar.initials}</span>
              <div class="exec-meta">
                <span class="exec-name">${d.ceo_name}</span>
                <span class="exec-sub">${d.ticker} &bull; ${d.company_name}</span>
              </div>
            </div>
          </td>
          <td>${d.sector}</td>
          <td class="num">${d.start_date} &rarr; ${d.end_date}</td>
          <td><span class="status-pill ${statusCls}">${d.status}</span></td>
          <td class="num">${d.current_weight_pct > 0 ? `${d.current_weight_pct.toFixed(2)}%` : "0.00%"}</td>
          <td class="num">${startText}</td>
          <td class="num"><strong>${peakText}</strong><span class="peak-sub">${d.peak_date || ""}</span></td>
          <td class="num ${retCls}">${retText}</td>
          <td>${buildSparklineSvg(d.sparkline, (retVal || 0) >= 0)}</td>
        </tr>`;
      })
      .join("");
  }

  function refreshAllCharts() {
    if (!state.payload || !window.BlackCeoCharts) {
      return;
    }
    const opts = {
      strategy: state.strategy,
      scale: state.scale,
      range: REGIME_WINDOWS[state.regime] || null,
    };
    window.BlackCeoCharts.renderEquityCurve("chart-equity-curve", state.payload, opts);
    window.BlackCeoCharts.renderDrawdown("chart-drawdown", state.payload, opts);
    window.BlackCeoCharts.renderConstituentBreadth("chart-constituent-count", state.payload, opts);
    window.BlackCeoCharts.renderFF5Factors("chart-ff5-factors", state.payload);
    window.BlackCeoCharts.renderMonteCarlo("chart-mc-bootstrap", state.payload);
  }

  function bindControls() {
    const strategySelect = document.getElementById("strategy-select");
    const regimeSelect = document.getElementById("regime-select");
    const rosterSearch = document.getElementById("roster-search");
    const statusFilter = document.getElementById("roster-status-filter");

    if (strategySelect) {
      strategySelect.addEventListener("change", (e) => {
        state.strategy = e.target.value;
        refreshAllCharts();
      });
    }
    if (regimeSelect) {
      regimeSelect.addEventListener("change", (e) => {
        state.regime = e.target.value;
        refreshAllCharts();
      });
    }
    document.querySelectorAll(".pill-btn[data-scale]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".pill-btn[data-scale]").forEach((b) => b.classList.remove("pill-btn--active"));
        btn.classList.add("pill-btn--active");
        state.scale = btn.getAttribute("data-scale");
        refreshAllCharts();
      });
    });
    if (rosterSearch) {
      rosterSearch.addEventListener("input", (e) => {
        state.search = e.target.value;
        renderRosterTable(state.payload);
      });
    }
    if (statusFilter) {
      statusFilter.addEventListener("change", (e) => {
        state.statusFilter = e.target.value;
        renderRosterTable(state.payload);
      });
    }

    const fnKeys = Array.from(document.querySelectorAll(".fn-key[data-jump]"));
    fnKeys.forEach((btn) => {
      btn.addEventListener("click", () => {
        fnKeys.forEach((b) => b.classList.remove("fn-key--active"));
        btn.classList.add("fn-key--active");
        const target = document.getElementById(btn.getAttribute("data-jump"));
        if (target) {
          target.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      });
    });

    window.addEventListener("keydown", (e) => {
      const idx = ["F1", "F2", "F3", "F4", "F5"].indexOf(e.key);
      if (idx >= 0 && fnKeys[idx]) {
        e.preventDefault();
        fnKeys[idx].click();
      }
    });
  }

  async function init() {
    bindControls();
    try {
      const resp = await fetch("data/dashboard_payload.json");
      if (!resp.ok) {
        return;
      }
      state.payload = await resp.json();
      renderMacroTable(state.payload);
      renderRosterTable(state.payload);
      refreshAllCharts();
    } catch (err) {
      console.error("Failed to load dashboard payload:", err);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
