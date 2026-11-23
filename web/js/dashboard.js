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

  const PAGE_TITLES = {
    "01": "01 OVERVIEW",
    "02": "02 EQUITY & DRAWDOWN",
    "03": "03 UNIVERSE BREADTH",
    "04": "04 MACRO REGIMES",
    "05": "05 FF5 & BOOTSTRAP",
    "06": "06 40-CEO ROSTER",
  };

  const state = {
    payload: null,
    activePage: "01",
    strategy: "Black_CEO_Net",
    regime: "full_horizon",
    scale: "log",
    search: "",
    statusFilter: "ALL",
    railMode: "tenure",
  };

  function buildSparklineSvg(points, isPositive, w = 44, h = 16) {
    if (!Array.isArray(points) || points.length < 2) {
      return "<span>&mdash;</span>";
    }
    const min = Math.min(...points);
    const max = Math.max(...points);
    const span = max - min || 1;
    const coords = points
      .map((val, idx) => {
        const x = ((idx / (points.length - 1)) * w).toFixed(1);
        const y = (h - 2 - ((val - min) / span) * (h - 4)).toFixed(1);
        return `${x},${y}`;
      })
      .join(" ");
    const stroke = isPositive ? "#00e676" : "#ff5252";
    return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-hidden="true">
      <polyline fill="none" stroke="${stroke}" stroke-width="1.4" points="${coords}" />
    </svg>`;
  }

  function switchPage(pageId) {
    if (!PAGE_TITLES[pageId]) return;
    state.activePage = pageId;

    document.querySelectorAll(".bb-screen").forEach((el) => {
      el.classList.toggle("bb-screen--active", el.id === `screen-${pageId}`);
    });
    document.querySelectorAll(".bb-nav-item[data-page]").forEach((btn) => {
      btn.classList.toggle("bb-nav-item--active", btn.getAttribute("data-page") === pageId);
    });
    const badge = document.getElementById("bb-active-screen-badge");
    if (badge) badge.textContent = PAGE_TITLES[pageId];

    const cmdInput = document.getElementById("bb-cmd-input");
    if (cmdInput) cmdInput.value = `${Number(pageId)}<GO>`;

    requestAnimationFrame(() => {
      refreshAllCharts();
      window.dispatchEvent(new Event("resize"));
    });
  }

  function renderRightRail(payload) {
    const driversEl = document.getElementById("rail-drivers");
    const metricsEl = document.getElementById("rail-metrics");
    const constEl = document.getElementById("rail-constituents");
    if (!driversEl || !metricsEl || !constEl || !payload) return;

    const eqSeries = payload.series.wealth_10k.Black_CEO_Net.slice(-30);
    const ddSeries = payload.series.drawdown_pct.Black_CEO_Net.slice(-30);
    const ntSeries = payload.series.active_count.slice(-30);

    const drivers = [
      { num: "31)", name: "Net CAGR (2000-26)", val: "12.99%", cls: "bb-green", pts: eqSeries, pos: true, page: "02" },
      { num: "32)", name: "Wealth Multiple", val: "26.48x", cls: "bb-amber", pts: eqSeries, pos: true, page: "02" },
      { num: "33)", name: "Jensen Alpha (SPY)", val: "+10.27%", cls: "bb-green", pts: eqSeries, pos: true, page: "05" },
      { num: "34)", name: "Post-2010 CAGR", val: "16.62%", cls: "bb-green", pts: eqSeries, pos: true, page: "04" },
      { num: "35)", name: "Active Black CEOs", val: "17", cls: "bb-cyan", pts: ntSeries, pos: true, page: "03" },
    ];

    const metrics = [
      { num: "36)", name: "Annualized Vol", val: "38.32%", cls: "bb-amber", pts: ddSeries, pos: false, page: "04" },
      { num: "37)", name: "Max Drawdown", val: "70.44%", cls: "bb-red", pts: ddSeries, pos: false, page: "02" },
      { num: "38)", name: "Sharpe / Sortino", val: "0.45/0.84", cls: "bb-amber", pts: eqSeries, pos: true, page: "04" },
      { num: "39)", name: "GARCH(1,1) a+b", val: "0.9982", cls: "bb-amber", pts: ddSeries, pos: false, page: "04" },
      { num: "40)", name: "Post-2010 GARCH HL", val: "7.0d", cls: "bb-green", pts: eqSeries, pos: true, page: "03" },
    ];

    const renderIndicatorRow = (item) => `<div class="bb-rail-row" data-jump-page="${item.page}">
      <span class="bb-rail-num">${item.num}</span>
      <span class="bb-rail-name">${item.name}</span>
      <span>${buildSparklineSvg(item.pts, item.pos)}</span>
      <span class="bb-rail-val ${item.cls}">${item.val}</span>
    </div>`;

    driversEl.innerHTML = drivers.map(renderIndicatorRow).join("");
    metricsEl.innerHTML = metrics.map(renderIndicatorRow).join("");

    document.querySelectorAll("[data-jump-page]").forEach((row) => {
      row.addEventListener("click", () => switchPage(row.getAttribute("data-jump-page")));
    });

    const activeDossiers = payload.ceo_dossiers.filter((d) => d.status === "ACTIVE");
    constEl.innerHTML = activeDossiers
      .map((d, idx) => {
        const ret = Number(d.tenure_return_pct || 0);
        let valStr = `${ret >= 0 ? "+" : ""}${ret.toFixed(1)}%`;
        let cls = ret >= 0 ? "bb-green" : "bb-red";
        if (state.railMode === "peak") {
          valStr = d.peak_price !== null ? `$${d.peak_price.toFixed(0)}` : "—";
          cls = "bb-amber";
        } else if (state.railMode === "weight") {
          valStr = `${d.current_weight_pct.toFixed(2)}%`;
          cls = "bb-cyan";
        }
        return `<div class="bb-rail-row" data-ceo-ticker="${d.ticker}">
          <span class="bb-rail-num">${51 + idx})</span>
          <span class="bb-rail-name" title="${d.ceo_name} (${d.company_name})">${d.ticker} &bull; ${d.ceo_name.split(" ").slice(-1)[0]}</span>
          <span>${buildSparklineSvg(d.sparkline, ret >= 0)}</span>
          <span class="bb-rail-val ${cls}">${valStr}</span>
        </div>`;
      })
      .join("");

    constEl.querySelectorAll("[data-ceo-ticker]").forEach((row) => {
      row.addEventListener("click", () => {
        const t = row.getAttribute("data-ceo-ticker");
        state.search = t;
        const searchEl = document.getElementById("roster-search");
        if (searchEl) searchEl.value = t;
        renderRosterTable(state.payload);
        switchPage("06");
      });
    });
  }

  function renderResearchFeed() {
    const feedEl = document.getElementById("bb-research-feed");
    if (!feedEl) return;
    const items = [
      { n: "1)", text: "Full-Horizon Net CAGR outpaces SPY (+515 bps) & RSP (+198 bps)", val: "12.99%", page: "02" },
      { n: "2)", text: "Post-2010 Modern Universe CAGR accelerates as Nt expands to 17", val: "16.62%", page: "03" },
      { n: "3)", text: "Convex-Quality QP Optimizer compresses volatility by -1,351 bps", val: "24.81%", page: "04" },
      { n: "4)", text: "Fama-French 5-Factor Annualized Alpha (Newey-West HAC t=1.51)", val: "+10.52%", page: "05" },
      { n: "5)", text: "GARCH(1,1) post-2010 volatility shock half-life drops 95%", val: "7.0d", page: "04" },
      { n: "6)", text: "AI & High-Rate Regime (2023–2026) cumulative return vs SPY +74.2%", val: "+110.4%", page: "04" },
      { n: "7)", text: "Top Active Tenure Performer: Alex Karp (PLTR) since Sep 2020", val: "+1,680%", page: "06" },
      { n: "8)", text: "Top Healthcare Tenure Performer: Herriot Tabuteau (AXSM)", val: "+1,513%", page: "06" },
    ];
    feedEl.innerHTML = items
      .map(
        (it) => `<div class="bb-feed-row" data-feed-page="${it.page}">
          <span class="bb-feed-row__num">${it.n}</span>
          <span class="bb-feed-row__text">${it.text}</span>
          <span class="bb-feed-row__val">${it.val}</span>
        </div>`
      )
      .join("");
    feedEl.querySelectorAll("[data-feed-page]").forEach((row) => {
      row.addEventListener("click", () => switchPage(row.getAttribute("data-feed-page")));
    });
  }

  function renderMacroTable(payload) {
    const tbody = document.getElementById("macro-regime-tbody");
    const regimes = payload?.timeseries_diagnostics?.macro_regimes;
    if (!tbody || !Array.isArray(regimes)) return;
    const stratKey = state.strategy || "Black_CEO_Net";
    tbody.innerHTML = regimes
      .map((r) => {
        const stratStats = r.series?.[stratKey] || r.series?.Black_CEO_Net || {};
        const spyStats = r.series?.SPY_Daily || {};
        const stratRet = Number(stratStats.cumulative_return || 0) * 100;
        const spyRet = Number(spyStats.cumulative_return || 0) * 100;
        const mdd = Number(stratStats.max_drawdown || 0) * 100;
        const rec =
          stratStats.recovery_days !== null && stratStats.recovery_days !== undefined
            ? `${stratStats.recovery_days}d`
            : "Unrec.";
        const downCap = Number(stratStats.downside_capture_vs_spy || 0);
        return `<tr>
          <td>${r.label}</td>
          <td class="num">${r.active_constituents_min}–${r.active_constituents_max}</td>
          <td class="num ${stratRet >= 0 ? "num--pos" : "num--neg"}">${stratRet >= 0 ? "+" : ""}${stratRet.toFixed(2)}%</td>
          <td class="num ${spyRet >= 0 ? "num--pos" : "num--neg"}">${spyRet >= 0 ? "+" : ""}${spyRet.toFixed(2)}%</td>
          <td class="num">${mdd.toFixed(2)}%</td>
          <td class="num">${rec}</td>
          <td class="num">${downCap.toFixed(2)}&times;</td>
        </tr>`;
      })
      .join("");
  }

  function renderRosterTable(payload) {
    const tbody = document.getElementById("roster-tbody");
    if (!tbody || !Array.isArray(payload?.ceo_dossiers)) return;
    const query = state.search.trim().toLowerCase();
    const filtered = payload.ceo_dossiers.filter((d) => {
      if (state.statusFilter !== "ALL" && d.status !== state.statusFilter) return false;
      if (!query) return true;
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
          <td>${buildSparklineSvg(d.sparkline, (retVal || 0) >= 0, 84, 22)}</td>
        </tr>`;
      })
      .join("");
  }

  function refreshAllCharts() {
    if (!state.payload || !window.BlackCeoCharts) return;
    const opts = {
      strategy: state.strategy,
      scale: state.scale,
      range: REGIME_WINDOWS[state.regime] || null,
    };
    window.BlackCeoCharts.renderEquityCurve("chart-equity-curve", state.payload, opts);
    window.BlackCeoCharts.renderDrawdown("chart-drawdown", state.payload, opts);
    window.BlackCeoCharts.renderConstituentBreadth("chart-constituent-count", state.payload, opts);
    window.BlackCeoCharts.renderFF5Factors("chart-ff5-factors", state.payload, opts);
    window.BlackCeoCharts.renderMonteCarlo("chart-mc-bootstrap", state.payload, opts);
  }

  function bindControls() {
    const strategySelect = document.getElementById("strategy-select");
    const regimeSelect = document.getElementById("regime-select");
    const rosterSearch = document.getElementById("roster-search");
    const statusFilter = document.getElementById("roster-status-filter");
    const railModeSelect = document.getElementById("rail-metric-mode");
    const toggleSidebarBtn = document.getElementById("toggle-sidebar-btn");
    const cmdForm = document.getElementById("bb-cmd-form");
    const cmdInput = document.getElementById("bb-cmd-input");

    if (cmdForm && cmdInput) {
      cmdForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const raw = cmdInput.value.trim();
        const numMatch = raw.match(/^0?([1-6])/);
        if (numMatch) {
          switchPage(`0${numMatch[1]}`);
        } else {
          state.search = raw;
          if (rosterSearch) rosterSearch.value = raw;
          renderRosterTable(state.payload);
          switchPage("06");
        }
      });
    }

    if (strategySelect) {
      strategySelect.addEventListener("change", (e) => {
        state.strategy = e.target.value;
        renderMacroTable(state.payload);
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
    if (railModeSelect) {
      railModeSelect.addEventListener("change", (e) => {
        state.railMode = e.target.value;
        renderRightRail(state.payload);
      });
    }
    if (toggleSidebarBtn) {
      toggleSidebarBtn.addEventListener("click", () => {
        const ws = document.getElementById("bb-workstation");
        if (ws) ws.classList.toggle("bb-workstation--collapsed");
        window.dispatchEvent(new Event("resize"));
      });
    }

    document.querySelectorAll("[data-page]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const targetPage = btn.getAttribute("data-page");
        const nextStatus = btn.getAttribute("data-filter-status");
        if (nextStatus) {
          state.statusFilter = nextStatus;
          if (statusFilter) statusFilter.value = nextStatus;
          renderRosterTable(state.payload);
        }
        switchPage(targetPage);
      });
    });

    window.addEventListener("keydown", (e) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "SELECT") return;
      if (["1", "2", "3", "4", "5", "6"].includes(e.key)) {
        switchPage(`0${e.key}`);
      }
    });
  }

  async function init() {
    bindControls();
    renderResearchFeed();
    try {
      const resp = await fetch("data/dashboard_payload.json");
      if (!resp.ok) return;
      state.payload = await resp.json();
      renderRightRail(state.payload);
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
