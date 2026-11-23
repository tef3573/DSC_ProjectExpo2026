(function () {
  "use strict";

  const SCREEN_HEADERS = {
    "0": "WEI / TOP <GO> LIVE MARKET MONITOR",
    "1": "BCEOS Index DES <GO> SECURITY DESCRIPTION",
    "2": "BCEOS Index MEMB <GO> 40-CEO CONSTITUENTS",
    "3": "BCEOS Index GP <GO> WEALTH & DRAWDOWN LAB",
    "4": "BCEOS Index ECST <GO> MACRO REGIME ATTRIBUTION",
    "5": "BCEOS Index VOL <GO> GARCH(1,1) RISK OVERLAY",
    "6": "BCEOS Index ALPH <GO> FF5, BOOTSTRAP & ARIMA",
  };

  const state = {
    payload: null,
    activeScreen: "0",
    strategy: "Black_CEO_Net",
    scale: "log",
    search: "",
    statusFilter: "ALL",
    simulateTicks: false,
    prevQuotes: {},
    marketData: null,
    miniCursor: 6762,
    miniPlaying: false,
  };

  function updateClock() {
    const el = document.getElementById("bb-nyc-clock");
    if (!el) return;
    try {
      const nycTime = new Date().toLocaleTimeString("en-US", {
        timeZone: "America/New_York",
        hour12: false,
      });
      el.textContent = `NYC ${nycTime} EST`;
    } catch {
      el.textContent = "NYC LIVE EST";
    }
  }

  function switchScreen(screenId) {
    if (!SCREEN_HEADERS[screenId]) return;
    state.activeScreen = screenId;

    document.querySelectorAll(".bb-screen").forEach((sec) => {
      sec.classList.toggle("bb-screen--active", sec.id === `screen-${screenId}`);
    });
    document.querySelectorAll(".bb-fn-btn[data-screen]").forEach((btn) => {
      btn.classList.toggle("bb-fn-btn--active", btn.getAttribute("data-screen") === screenId);
    });

    const hdr = document.getElementById("bb-header-security");
    if (hdr) hdr.textContent = SCREEN_HEADERS[screenId];

    const cmdInput = document.getElementById("bb-cmd-input");
    if (cmdInput) {
      cmdInput.value =
        screenId === "0"
          ? "BCEOS Index DES <GO>"
          : screenId === "1"
            ? "BCEOS Index DES <GO>"
            : `${screenId} <GO>`;
    }

    requestAnimationFrame(() => {
      refreshCharts();
      window.dispatchEvent(new Event("resize"));
    });
  }

  function animateTypeAndLaunchDes() {
    const cmdInput = document.getElementById("bb-cmd-input");
    if (!cmdInput) {
      switchScreen("1");
      return;
    }
    const targetCmd = "BCEOS Index DES <GO>";
    cmdInput.value = "";
    let i = 0;
    const timer = setInterval(() => {
      i += 1;
      cmdInput.value = targetCmd.slice(0, i);
      if (i >= targetCmd.length) {
        clearInterval(timer);
        setTimeout(() => switchScreen("1"), 140);
      }
    }, 18);
  }

  function renderIntradaySparkline(intraday) {
    const host = document.getElementById("wei-intraday-svg");
    if (!host || !intraday || !Array.isArray(intraday.spx)) return;
    const pts = intraday.spx;
    const w = 360;
    const h = 72;
    const min = Math.min(...pts);
    const max = Math.max(...pts);
    const span = max - min || 1;
    const coords = pts
      .map((val, idx) => {
        const x = ((idx / Math.max(1, pts.length - 1)) * (w - 8) + 4).toFixed(1);
        const y = (h - 6 - ((val - min) / span) * (h - 12)).toFixed(1);
        return `${x},${y}`;
      })
      .join(" ");
    const lastCoord = coords.split(" ").pop() || "350,36";
    const [lx, ly] = lastCoord.split(",");

    host.innerHTML = `<svg width="100%" height="${h}" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
      <line x1="0" y1="${h / 2}" x2="${w}" y2="${h / 2}" stroke="#1e232b" stroke-dasharray="3,3" />
      <polyline fill="none" stroke="#00ff00" stroke-width="1.6" points="${coords}" />
      <circle cx="${lx}" cy="${ly}" r="3" fill="#00ff00" />
    </svg>`;
  }

  function renderMarketTables(data) {
    if (!data) return;
    const weiBody = document.getElementById("wei-tbody");
    const fitBody = document.getElementById("fit-tbody");
    const feedTag = document.getElementById("wei-feed-tag");
    const footerTag = document.getElementById("footer-feed-status");

    if (feedTag) {
      feedTag.textContent = state.simulateTicks
        ? "SIMULATED LIVE TICKS ●"
        : data.is_live
          ? "LIVE API ●"
          : "SNAPSHOT (2026-11-20)";
      feedTag.className = data.is_live || state.simulateTicks ? "bb-green" : "bb-amber";
    }
    if (footerTag) {
      footerTag.textContent = `FEED: ${feedTag ? feedTag.textContent : "READY"}`;
    }

    if (weiBody && Array.isArray(data.equities)) {
      weiBody.innerHTML = data.equities
        .map((row) => {
          const prev = state.prevQuotes[row.symbol];
          let flashCls = "";
          if (prev !== undefined && prev !== row.last) {
            flashCls = row.last > prev ? "tick-flash-up" : "tick-flash-down";
          }
          state.prevQuotes[row.symbol] = row.last;
          const pos = row.chg >= 0;
          const cls = row.symbol === "^VIX" ? (pos ? "bb-red" : "bb-green") : pos ? "bb-green" : "bb-red";
          return `<tr class="${flashCls}">
            <td class="bb-amber bb-bold">${row.name}</td>
            <td class="r bb-bold">${Number(row.last).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
            <td class="r ${cls}">${pos ? "+" : ""}${Number(row.chg).toFixed(2)}</td>
            <td class="r ${cls}">${pos ? "+" : ""}${Number(row.pct_chg).toFixed(2)}%</td>
            <td class="r ${row.ytd >= 0 ? "bb-green" : "bb-red"}">${row.ytd >= 0 ? "+" : ""}${Number(row.ytd).toFixed(2)}%</td>
          </tr>`;
        })
        .join("");

      const vixRow = data.equities.find((e) => e.symbol === "^VIX");
      const vixBadge = document.getElementById("vix-context-badge");
      const volNote = document.getElementById("wei-vol-commentary");
      if (vixRow && vixBadge && volNote) {
        vixBadge.textContent = `VIX: ${Number(vixRow.last).toFixed(2)}`;
        volNote.textContent = `Live VIX (${Number(vixRow.last).toFixed(2)}) vs. BCEOS unconditional GARCH(1,1) vol (38.32%) and causal 20% vol-targeted overlay (19.93%).`;
      }
    }

    if (fitBody && Array.isArray(data.rates)) {
      fitBody.innerHTML = data.rates
        .map((r) => {
          const cls = r.chg_bps <= 0 ? "bb-green" : "bb-red";
          return `<tr>
            <td class="bb-cyan">${r.name}</td>
            <td class="r bb-bold">${Number(r.yield_pct).toFixed(2)}%</td>
            <td class="r ${cls}">${r.chg_bps >= 0 ? "+" : ""}${Number(r.chg_bps).toFixed(1)}</td>
            <td class="r">${r.spread_vs_3m_bps >= 0 ? "+" : ""}${Number(r.spread_vs_3m_bps).toFixed(1)} bps</td>
          </tr>`;
        })
        .join("");
    }

    renderIntradaySparkline(data.intraday);
  }

  function renderDesMiniChart() {
    const host = document.getElementById("des-mini-chart");
    if (!host || !state.payload) return;
    const dates = state.payload.series.dates;
    const endIdx = Math.min(dates.length - 1, Math.max(1, state.miniCursor));
    const step = Math.max(1, Math.floor(endIdx / 140));

    const bceos = [];
    const spx = [];
    for (let i = 0; i <= endIdx; i += step) {
      bceos.push(state.payload.series.wealth_10k[state.strategy][i]);
      spx.push(state.payload.series.wealth_10k.SPY[i]);
    }
    bceos.push(state.payload.series.wealth_10k[state.strategy][endIdx]);
    spx.push(state.payload.series.wealth_10k.SPY[endIdx]);

    const w = 420;
    const h = 145;
    const maxVal = Math.max(...bceos, ...spx, 12000);
    const minVal = 5000;
    const span = maxVal - minVal || 1;

    const toPolyline = (arr) =>
      arr
        .map((val, idx) => {
          const x = ((idx / Math.max(1, arr.length - 1)) * (w - 14) + 6).toFixed(1);
          const y = (h - 10 - ((val - minVal) / span) * (h - 20)).toFixed(1);
          return `${x},${y}`;
        })
        .join(" ");

    const bCoords = toPolyline(bceos);
    const sCoords = toPolyline(spx);
    const tip = bCoords.split(" ").pop() || "410,20";
    const [tx, ty] = tip.split(",");

    host.innerHTML = `<svg width="100%" height="100%" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
      <line x1="0" y1="${h * 0.25}" x2="${w}" y2="${h * 0.25}" stroke="#1e232b" />
      <line x1="0" y1="${h * 0.5}" x2="${w}" y2="${h * 0.5}" stroke="#1e232b" />
      <line x1="0" y1="${h * 0.75}" x2="${w}" y2="${h * 0.75}" stroke="#1e232b" />
      <polyline fill="none" stroke="#00d1ff" stroke-width="1.4" points="${sCoords}" />
      <polyline fill="none" stroke="#00ff00" stroke-width="2" points="${bCoords}" />
      <circle cx="${tx}" cy="${ty}" r="3.5" fill="#00ff00" />
    </svg>`;

    const dateEl = document.getElementById("des-mini-cursor-date");
    if (dateEl) dateEl.textContent = dates[endIdx] || "2026-11-20";
  }

  function buildSparklineSvg(points, isPositive, w = 78, h = 18) {
    if (!Array.isArray(points) || points.length < 2) return "<span>&mdash;</span>";
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
    const stroke = isPositive ? "#00ff00" : "#ff3b30";
    return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"><polyline fill="none" stroke="${stroke}" stroke-width="1.4" points="${coords}" /></svg>`;
  }

  function renderMacroTable(payload) {
    const tbody = document.getElementById("macro-regime-tbody");
    const regimes = payload?.timeseries_diagnostics?.macro_regimes;
    if (!tbody || !Array.isArray(regimes)) return;
    tbody.innerHTML = regimes
      .map((r) => {
        const s = r.series?.[state.strategy] || r.series?.Black_CEO_Net || {};
        const b = r.series?.SPY_Daily || {};
        const sRet = Number(s.cumulative_return || 0) * 100;
        const bRet = Number(b.cumulative_return || 0) * 100;
        return `<tr>
          <td class="bb-amber bb-bold">${r.label}</td>
          <td class="r">${r.active_constituents_min}–${r.active_constituents_max}</td>
          <td class="r ${sRet >= 0 ? "bb-green" : "bb-red"}">${sRet >= 0 ? "+" : ""}${sRet.toFixed(2)}%</td>
          <td class="r ${bRet >= 0 ? "bb-green" : "bb-red"}">${bRet >= 0 ? "+" : ""}${bRet.toFixed(2)}%</td>
          <td class="r">${(Number(s.max_drawdown || 0) * 100).toFixed(2)}%</td>
          <td class="r">${s.recovery_days !== null ? `${s.recovery_days}d` : "Unrec."}</td>
          <td class="r">${Number(s.downside_capture_vs_spy || 0).toFixed(2)}x</td>
        </tr>`;
      })
      .join("");
  }

  function renderRosterTable(payload) {
    const tbody = document.getElementById("roster-tbody");
    if (!tbody || !Array.isArray(payload?.ceo_dossiers)) return;
    const q = state.search.trim().toLowerCase();
    const filtered = payload.ceo_dossiers.filter((d) => {
      if (state.statusFilter !== "ALL" && d.status !== state.statusFilter) return false;
      if (!q) return true;
      return (
        d.ticker.toLowerCase().includes(q) ||
        d.ceo_name.toLowerCase().includes(q) ||
        d.company_name.toLowerCase().includes(q) ||
        d.sector.toLowerCase().includes(q)
      );
    });

    tbody.innerHTML = filtered
      .map((d) => {
        const ret = d.tenure_return_pct;
        const stCls = d.status === "ACTIVE" ? "bb-green" : d.status === "DELISTED" ? "bb-amber" : "bb-muted";
        return `<tr>
          <td><strong class="bb-amber">${d.ticker}</strong> &bull; <strong>${d.ceo_name}</strong> <span class="bb-muted">(${d.company_name})</span></td>
          <td>${d.sector}</td>
          <td>${d.start_date} &rarr; ${d.end_date}</td>
          <td class="${stCls} bb-bold">${d.status}</td>
          <td class="r">${d.current_weight_pct > 0 ? `${d.current_weight_pct.toFixed(2)}%` : "0.00%"}</td>
          <td class="r">${d.start_price !== null ? `$${d.start_price.toFixed(2)}` : "—"}</td>
          <td class="r bb-amber"><strong>${d.peak_price !== null ? `$${d.peak_price.toFixed(2)}` : "—"}</strong> <span class="bb-muted">(${d.peak_date || ""})</span></td>
          <td class="r ${(ret || 0) >= 0 ? "bb-green" : "bb-red"} bb-bold">${ret !== null ? `${ret >= 0 ? "+" : ""}${ret.toFixed(2)}%` : "—"}</td>
          <td>${buildSparklineSvg(d.sparkline, (ret || 0) >= 0)}</td>
        </tr>`;
      })
      .join("");
  }

  function refreshCharts() {
    if (!state.payload || !window.BlackCeoCharts) return;
    const opts = { strategy: state.strategy, scale: state.scale, range: null };
    window.BlackCeoCharts.renderEquityCurve("chart-equity-curve", state.payload, opts);
    window.BlackCeoCharts.renderDrawdown("chart-drawdown", state.payload, opts);
    window.BlackCeoCharts.renderConstituentBreadth("chart-constituent-count", state.payload, opts);
    window.BlackCeoCharts.renderFF5Factors("chart-ff5-factors", state.payload, opts);
    window.BlackCeoCharts.renderMonteCarlo("chart-mc-bootstrap", state.payload, opts);
  }

  async function pollLiveMarket() {
    try {
      const resp = await fetch("/api/live-market");
      if (resp.ok) {
        state.marketData = await resp.json();
        renderMarketTables(state.marketData);
      }
    } catch {
      // Fallback already populated
    }
  }

  function bindControls() {
    const cmdForm = document.getElementById("bb-cmd-form");
    const cmdInput = document.getElementById("bb-cmd-input");

    if (cmdForm && cmdInput) {
      cmdForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const raw = cmdInput.value.trim().toUpperCase();
        if (raw.includes("WEI") || raw.includes("TOP") || raw.startsWith("0")) switchScreen("0");
        else if (raw.includes("DES") || raw.startsWith("1")) switchScreen("1");
        else if (raw.includes("MEMB") || raw.startsWith("2")) switchScreen("2");
        else if (raw.includes("GP") || raw.startsWith("3")) switchScreen("3");
        else if (raw.includes("ECST") || raw.startsWith("4")) switchScreen("4");
        else if (raw.includes("VOL") || raw.startsWith("5")) switchScreen("5");
        else if (raw.includes("ALPH") || raw.startsWith("6")) switchScreen("6");
        else {
          state.search = raw;
          renderRosterTable(state.payload);
          switchScreen("2");
        }
      });
    }

    document.querySelectorAll("[data-screen]").forEach((btn) => {
      btn.addEventListener("click", () => switchScreen(btn.getAttribute("data-screen")));
    });

    const launchBtn = document.getElementById("launch-des-btn");
    const launchHeadline = document.getElementById("launch-bceos-headline");
    if (launchBtn) launchBtn.addEventListener("click", animateTypeAndLaunchDes);
    if (launchHeadline) launchHeadline.addEventListener("click", animateTypeAndLaunchDes);

    const strategySelect = document.getElementById("strategy-select");
    if (strategySelect) {
      strategySelect.addEventListener("change", (e) => {
        state.strategy = e.target.value;
        renderDesMiniChart();
        renderMacroTable(state.payload);
        refreshCharts();
      });
    }

    document.querySelectorAll("[data-scale]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("[data-scale]").forEach((b) => b.classList.remove("bb-mini-btn--active"));
        btn.classList.add("bb-mini-btn--active");
        state.scale = btn.getAttribute("data-scale");
        refreshCharts();
      });
    });

    const simBtn = document.getElementById("sim-ticks-btn");
    if (simBtn) {
      simBtn.addEventListener("click", () => {
        state.simulateTicks = !state.simulateTicks;
        simBtn.textContent = `[SIMULATE TICKS: ${state.simulateTicks ? "ON" : "OFF"}]`;
        renderMarketTables(state.marketData);
      });
    }

    const rosterSearch = document.getElementById("roster-search");
    const statusFilter = document.getElementById("roster-status-filter");
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

    const miniStart = document.getElementById("des-mini-start");
    const miniPlay = document.getElementById("des-mini-play");
    const miniEnd = document.getElementById("des-mini-end");
    if (miniStart) {
      miniStart.addEventListener("click", () => {
        state.miniPlaying = false;
        state.miniCursor = 50;
        renderDesMiniChart();
      });
    }
    if (miniPlay) {
      miniPlay.addEventListener("click", () => {
        state.miniPlaying = !state.miniPlaying;
        if (state.miniPlaying && state.miniCursor >= 6700) state.miniCursor = 50;
      });
    }
    if (miniEnd) {
      miniEnd.addEventListener("click", () => {
        state.miniPlaying = false;
        state.miniCursor = 6762;
        renderDesMiniChart();
      });
    }

    window.addEventListener("keydown", (e) => {
      const tag = (e.target || {}).tagName;
      if (tag === "INPUT" || tag === "SELECT") return;
      if (["0", "1", "2", "3", "4", "5", "6"].includes(e.key)) {
        switchScreen(e.key);
      }
    });
  }

  async function init() {
    bindControls();
    updateClock();
    setInterval(updateClock, 1000);

    setInterval(() => {
      if (state.miniPlaying && state.payload) {
        state.miniCursor = Math.min(state.payload.series.dates.length - 1, state.miniCursor + 90);
        if (state.miniCursor >= state.payload.series.dates.length - 1) state.miniPlaying = false;
        renderDesMiniChart();
      }
    }, 30);

    setInterval(() => {
      if (state.simulateTicks && state.marketData?.equities) {
        state.marketData.equities.forEach((eq) => {
          const delta = Number(((Math.random() - 0.48) * (eq.last * 0.0012)).toFixed(2));
          eq.last = Number((eq.last + delta).toFixed(2));
          eq.chg = Number((eq.chg + delta).toFixed(2));
        });
        if (state.marketData.intraday?.spx) {
          const spx = state.marketData.equities[0].last;
          state.marketData.intraday.spx.push(spx);
          state.marketData.intraday.spx.shift();
        }
        renderMarketTables(state.marketData);
      }
    }, 1800);

    await pollLiveMarket();
    setInterval(pollLiveMarket, 20000);

    try {
      const resp = await fetch("data/dashboard_payload.json");
      if (!resp.ok) throw new Error("Payload HTTP error");
      state.payload = await resp.json();
      state.miniCursor = state.payload.series.dates.length - 1;
      renderDesMiniChart();
      renderMacroTable(state.payload);
      renderRosterTable(state.payload);
      refreshCharts();
    } catch (err) {
      const stage = document.getElementById("main-content");
      if (stage) {
        stage.insertAdjacentHTML(
          "afterbegin",
          `<div class="bb-narrator-bar"><strong class="bb-red">TELEMETRY ERROR:</strong> ${err.message}</div>`
        );
      }
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
