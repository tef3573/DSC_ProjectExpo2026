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
    showGross: true,
    showYieldOverlay: true,
    selectedRegime: "ALL",
    search: "",
    statusFilter: "ALL",
    simulateTicks: true,
    prevQuotes: {},
    marketData: null,
    miniCursor: 6763,
    miniPlaying: false,
    gpCursor: 6763,
    gpPlaying: false,
    volCursor: 6763,
    volPlaying: false,
    garchSpinAngle: 0.7,
    garchAutoSpin: true,
    garchShowPoints: false,
    shockSign: 1,
    mcResampleOffset: 0,
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
        screenId === "0" || screenId === "1" ? "BCEOS Index DES <GO>" : `${screenId} <GO>`;
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
    const h = 70;
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
    const lastCoord = coords.split(" ").pop() || "350,35";
    const [lx, ly] = lastCoord.split(",");

    host.innerHTML = `<svg width="100%" height="${h}" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
      <line x1="0" y1="${h / 2}" x2="${w}" y2="${h / 2}" stroke="#1e232b" stroke-dasharray="3,3" />
      <polyline fill="none" stroke="#00ff00" stroke-width="1.6" points="${coords}" />
      <circle cx="${lx}" cy="${ly}" r="3" fill="#00ff00" />
    </svg>`;
  }

  function renderNewsWire(headlines) {
    const list = document.getElementById("live-news-list");
    if (!list) return;
    const liveItems = Array.isArray(headlines) ? headlines : [];
    const pinnedBi = `<div class="bb-news-item bb-news-item--flash" id="launch-bceos-headline">
      <div class="bb-news-time">LIVE <span class="bb-badge-bi">[BI QUANT RESEARCH]</span></div>
      <div class="bb-news-title">
        NEW HYPOTHETICAL INDEX: BLACK CEO LEADERSHIP INDEX (BCEOS) OUTPERFORMS SPX BY +515 BPS/YR &rarr;
      </div>
      <div class="bb-news-cta">[CLICK TO LAUNCH: BCEOS Index DES &lt;GO&gt;]</div>
    </div>`;

    const dynamicHtml = liveItems
      .map((h) => {
        const titleHtml = h.url
          ? `<a href="${h.url}" target="_blank" rel="noopener noreferrer">${h.title}</a>`
          : h.title;
        return `<div class="bb-news-item">
          <div class="bb-news-time">${h.time} <span class="bb-cyan">[${h.tag}]</span></div>
          <div class="bb-news-title">${titleHtml}</div>
        </div>`;
      })
      .join("");

    list.innerHTML = pinnedBi + dynamicHtml;
    const pinnedEl = document.getElementById("launch-bceos-headline");
    if (pinnedEl) pinnedEl.addEventListener("click", animateTypeAndLaunchDes);
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
          : "SNAPSHOT";
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
    renderNewsWire(data.headlines);
  }

  function syncMetadataDates() {
    if (!state.payload || !state.payload.metadata) return;
    const meta = state.payload.metadata;
    const end = meta.horizon_end;
    const sessions = Number(meta.total_sessions || 6764).toLocaleString("en-US");
    const eqEnd = state.payload.series.wealth_10k.Black_CEO_Net.slice(-1)[0] || 264850;
    const mult = (eqEnd / 10000).toFixed(2);

    const topAsof = document.getElementById("top-asof-badge");
    const topLast = document.getElementById("top-last-badge");
    const launchH = document.getElementById("launch-horizon-val");
    const desH = document.getElementById("des-horizon-val");
    const scMult = document.getElementById("sc-bceos-mult");

    if (topAsof) topAsof.textContent = end;
    if (topLast) topLast.textContent = `${(eqEnd / 1000).toFixed(2)} (${mult}x)`;
    if (launchH) launchH.textContent = `${meta.horizon_start} to ${end} (${sessions} Sessions)`;
    if (desH) desH.textContent = `${meta.horizon_start} - ${end} (${sessions}d)`;
    if (scMult) scMult.textContent = `${mult}x`;
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
    const lastB = state.payload.series.wealth_10k[state.strategy][endIdx];
    const lastS = state.payload.series.wealth_10k.SPY[endIdx];
    bceos.push(lastB);
    spx.push(lastS);

    const w = 440;
    const h = 135;
    const maxVal = Math.max(...bceos, ...spx, 12000);
    const minVal = 5000;
    const span = maxVal - minVal || 1;

    const toPolyline = (arr) =>
      arr
        .map((val, idx) => {
          const x = ((idx / Math.max(1, arr.length - 1)) * (w - 46) + 40).toFixed(1);
          const y = (h - 18 - ((val - minVal) / span) * (h - 26)).toFixed(1);
          return `${x},${y}`;
        })
        .join(" ");

    const bCoords = toPolyline(bceos);
    const sCoords = toPolyline(spx);
    const tip = bCoords.split(" ").pop() || "430,20";
    const [tx, ty] = tip.split(",");

    host.innerHTML = `<svg width="100%" height="100%" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
      <line x1="38" y1="8" x2="38" y2="${h - 16}" stroke="#363c46" />
      <line x1="38" y1="${h - 16}" x2="${w - 4}" y2="${h - 16}" stroke="#363c46" />
      <text x="2" y="14" fill="#9aa3b2" font-size="8.5">$${Math.round(maxVal / 1000)}k</text>
      <text x="2" y="${h / 2}" fill="#9aa3b2" font-size="8.5">$${Math.round((maxVal + minVal) / 2000)}k</text>
      <text x="2" y="${h - 18}" fill="#9aa3b2" font-size="8.5">$10k</text>
      <text x="42" y="${h - 4}" fill="#9aa3b2" font-size="8.5">2000-01-03</text>
      <text x="${w / 2 - 20}" y="${h - 4}" fill="#9aa3b2" font-size="8.5">2013</text>
      <text x="${w - 64}" y="${h - 4}" fill="#ff9900" font-size="8.5">${dates[endIdx]}</text>
      <polyline fill="none" stroke="#00d1ff" stroke-width="1.4" points="${sCoords}" />
      <polyline fill="none" stroke="#00ff00" stroke-width="2" points="${bCoords}" />
      <circle cx="${tx}" cy="${ty}" r="3.5" fill="#00ff00" />
    </svg>`;

    const dateEl = document.getElementById("des-mini-cursor-date");
    const bValEl = document.getElementById("des-mini-bceos-val");
    const sValEl = document.getElementById("des-mini-spx-val");
    if (dateEl) dateEl.textContent = dates[endIdx];
    if (bValEl) bValEl.textContent = `$${(lastB / 1000).toFixed(1)}k`;
    if (sValEl) sValEl.textContent = `$${(lastS / 1000).toFixed(1)}k`;
  }

  function updateGpCursorTelemetry() {
    if (!state.payload) return;
    const dates = state.payload.series.dates;
    const idx = Math.min(dates.length - 1, Math.max(0, state.gpCursor));
    const dt = dates[idx];
    const bNet = state.payload.series.wealth_10k[state.strategy][idx];
    const bGross = (state.payload.series.wealth_10k.Black_CEO_Gross || [])[idx] || bNet;
    const spx = state.payload.series.wealth_10k.SPY[idx];
    const activeN = state.payload.series.active_count[idx];

    const telem = document.getElementById("gp-cursor-telemetry");
    if (telem) {
      telem.innerHTML = `CURSOR: <strong class="bb-amber">${dt} (${(idx + 1).toLocaleString()} / ${dates.length.toLocaleString()})</strong> | `
        + `BCEOS NET: <strong class="bb-green">$${Math.round(bNet).toLocaleString()}</strong> | `
        + `GROSS: <strong class="bb-cyan">$${Math.round(bGross).toLocaleString()}</strong> | `
        + `SPX: <strong>$${Math.round(spx).toLocaleString()}</strong> | `
        + `ACTIVE N: <strong class="bb-amber">${activeN}</strong>`;
    }

    // Chapter-based live narrator switching
    const year = Number(dt.slice(0, 4));
    const gpBar = document.getElementById("commentary-gp");
    if (gpBar) {
      if (year <= 2002) {
        gpBar.innerHTML = `<strong class="bb-amber">[CHAPTER 1: 2000–2002 DOT-COM RESILIENCE]</strong> Holding +12.36% cumulative gains while SPX falls -42.64% despite concentrated N=4–5 breadth.`;
      } else if (year >= 2007 && year <= 2009) {
        gpBar.innerHTML = `<strong class="bb-amber">[CHAPTER 2: 2007–2009 GFC LIQUIDITY TROUGH]</strong> Maximum full-horizon drawdown (-70.44%) occurs here as credit liquidity freezes across N=5 issuers.`;
      } else if (year >= 2010 && year <= 2019) {
        gpBar.innerHTML = `<strong class="bb-amber">[CHAPTER 3: 2010–2019 POST-2010 SCALABILITY]</strong> Breadth expands toward N=10+; annualized return accelerates to 16.62% with a 1.01 Sortino Ratio and 7.0d shock half-life.`;
      } else if (year >= 2020) {
        gpBar.innerHTML = `<strong class="bb-amber">[CHAPTER 4: 2020–2026 COVID V-RECOVERY &amp; AI ERA]</strong> Rapid V-shaped recovery after March 2020, followed by +110.37% gains in the 2023–2026 AI &amp; High-Rate regime (76.99% turnover = only 2.07% cumulative drag).`;
      }
    }
  }

  function updateVolCursorReadout() {
    if (!state.payload) return;
    const dates = state.payload.series.dates;
    const idx = Math.min(dates.length - 1, Math.max(0, state.volCursor));
    const wt = (state.payload.series.vol_target_weights_pct || [])[idx] || 100.0;
    const wtEl = document.getElementById("vol-wt-readout");
    const fillEl = document.getElementById("vol-wt-fill");
    const dtEl = document.getElementById("vol-date-readout");
    if (wtEl) wtEl.textContent = `${wt.toFixed(1)}%`;
    if (fillEl) fillEl.style.width = `${Math.min(100, Math.max(5, wt))}%`;
    if (dtEl) dtEl.textContent = dates[idx];
  }

  function renderDynamicCommentary() {
    if (!state.payload || !state.payload.analyst_commentary) return;
    const block =
      state.payload.analyst_commentary[state.strategy] ||
      state.payload.analyst_commentary.Black_CEO_Net ||
      {};
    const mapping = {
      "commentary-memb": block.memb_breadth,
      "commentary-gp": block.gp_wealth,
      "commentary-ecst": block.ecst_macro,
      "commentary-vol": block.vol_garch,
      "commentary-alph": block.alph_factors,
    };
    Object.entries(mapping).forEach(([id, text]) => {
      const el = document.getElementById(id);
      if (el && text) {
        const idx = text.indexOf(":");
        if (idx !== -1) {
          el.innerHTML = `<strong class="bb-amber">${text.slice(0, idx + 1)}</strong>${text.slice(idx + 1)}`;
        } else {
          el.textContent = text;
        }
      }
    });
  }

  function buildSparklineSvg(points, isPositive, w = 74, h = 18) {
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
    const filtered =
      state.selectedRegime === "ALL"
        ? regimes
        : regimes.filter((r) => r.regime_id === state.selectedRegime);

    tbody.innerHTML = filtered
      .map((r) => {
        const s = r.series?.[state.strategy] || r.series?.Black_CEO_Net || {};
        const b = r.series?.SPY_Daily || {};
        const sRet = Number(s.cumulative_return || 0) * 100;
        const bRet = Number(b.cumulative_return || 0) * 100;
        return `<tr class="bb-clickable-row" data-regime-id="${r.regime_id}">
          <td class="bb-amber bb-bold">${r.label}</td>
          <td class="r">${r.active_constituents_min}–${r.active_constituents_max}</td>
          <td class="r ${sRet >= 0 ? "bb-green" : "bb-red"}">${sRet >= 0 ? "+" : ""}${sRet.toFixed(1)}%</td>
          <td class="r ${bRet >= 0 ? "bb-green" : "bb-red"}">${bRet >= 0 ? "+" : ""}${bRet.toFixed(1)}%</td>
          <td class="r">${(Number(s.max_drawdown || 0) * 100).toFixed(1)}%</td>
          <td class="r">${Number(s.downside_capture_vs_spy || 0).toFixed(2)}x</td>
        </tr>`;
      })
      .join("");

    tbody.querySelectorAll("tr[data-regime-id]").forEach((tr) => {
      tr.addEventListener("click", () => {
        const rid = tr.getAttribute("data-regime-id");
        const reg = regimes.find((item) => item.regime_id === rid);
        const inlineBox = document.getElementById("ecst-inline-detail");
        if (reg && inlineBox) {
          const s = reg.series?.[state.strategy] || reg.series?.Black_CEO_Net || {};
          const b = reg.series?.SPY_Daily || {};
          const sRet = (Number(s.cumulative_return || 0) * 100).toFixed(2);
          const bRet = (Number(b.cumulative_return || 0) * 100).toFixed(2);
          const rec = s.recovery_days !== null ? `${s.recovery_days} trading days` : "unrecovered within window";
          inlineBox.innerHTML = `<div class="bb-amber bb-bold">[${reg.label.toUpperCase()} — REGIME BRIEFING]</div>`
            + `<div>During this macro window (${reg.start_date || ""} to ${reg.end_date || ""}), BCEOS returned <strong class="${Number(sRet) >= 0 ? "bb-green" : "bb-red"}">${sRet}%</strong> vs. SPX <strong>${bRet}%</strong> `
            + `across <strong>N=${reg.active_constituents_min}–${reg.active_constituents_max}</strong> active issuers, with a max drawdown of <strong>${(Number(s.max_drawdown || 0) * 100).toFixed(2)}%</strong> (${rec}) and <strong>${Number(s.downside_capture_vs_spy || 0).toFixed(2)}x</strong> downside capture.</div>`;
        }
      });
    });
  }

  function openCeoModal(dossier) {
    const modal = document.getElementById("ceo-modal");
    const body = document.getElementById("ceo-modal-body");
    if (!modal || !body || !dossier) return;
    const p = dossier.profile || {};
    const ret = dossier.tenure_return_pct;
    body.innerHTML = `
      <div class="bb-ceo-hero">
        <div class="bb-ceo-avatar" style="color:${dossier.avatar.fg};background:${dossier.avatar.bg};overflow:hidden;">
          ${dossier.avatar.photo_url ? `<img src="${dossier.avatar.photo_url}" alt="${dossier.ceo_name}" style="width:100%;height:100%;object-fit:cover;" onerror="this.remove();" />` : dossier.avatar.initials}
        </div>
        <div>
          <div class="bb-amber bb-bold" style="font-size:13px;">${dossier.ceo_name} &bull; ${dossier.company_name} (${dossier.ticker})</div>
          <div class="bb-cyan">${dossier.sector} &bull; ${p.index_membership || "US Public Equity"}</div>
          <div class="bb-muted">Tenure: ${dossier.start_date} &rarr; ${dossier.end_date} (${dossier.tenure_days.toLocaleString()} calendar days) &bull; Status: <strong>${dossier.status}</strong></div>
        </div>
      </div>
      <div class="bb-grid-2col">
        <div class="bb-box">
          <div class="bb-box__hdr"><span>COMPANY FUNDAMENTALS &amp; TENURE METRICS</span></div>
          <div class="bb-kv-list bb-kv-list--compact">
            <div class="bb-kv"><span>Market Cap / Scale:</span><strong>${p.market_cap || "Public Issuer"}</strong></div>
            <div class="bb-kv"><span>Revenue / Growth:</span><strong>${p.rev_growth || "N/A"}</strong></div>
            <div class="bb-kv"><span>ROE / Margin Profile:</span><strong>${p.roe_margin || "N/A"}</strong></div>
            <div class="bb-kv"><span>Tenure Start / Peak:</span><strong>$${dossier.start_price ?? "—"} &rarr; $${dossier.peak_price ?? "—"} (${dossier.peak_date || ""})</strong></div>
            <div class="bb-kv"><span>Cumulative Tenure Return:</span><strong class="${(ret || 0) >= 0 ? "bb-green" : "bb-red"}">${ret !== null ? `${ret >= 0 ? "+" : ""}${ret.toFixed(2)}%` : "—"}</strong></div>
          </div>
        </div>
        <div class="bb-box">
          <div class="bb-box__hdr"><span>COMPENSATION &amp; GOVERNANCE THESIS</span></div>
          <div class="bb-kv-list bb-kv-list--compact">
            <div class="bb-kv"><span>Executive Comp Mix:</span><strong>${p.comp_mix || "SEC DEF 14A"}</strong></div>
            <div style="margin-top:4px;color:#e2e8f0;">${p.governance_thesis || ""}</div>
          </div>
        </div>
      </div>
      <div class="bb-box">
        <div class="bb-box__hdr"><span>PRIMARY DATA &amp; SCREENING SOURCES (FUND SELECTION AUDIT)</span></div>
        <div class="bb-cyan">${p.sources || "SEC EDGAR Form 8-K, DEF 14A Proxy Filings, Yahoo Finance / CRSP."}</div>
      </div>
    `;
    modal.hidden = false;
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
        return `<tr class="bb-clickable-row" data-dossier-id="${d.id}">
          <td style="display:flex;align-items:center;gap:6px;">
            <span style="display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;font-size:9px;font-weight:700;color:${d.avatar.fg};background:${d.avatar.bg};border:1px solid #363c46;flex-shrink:0;overflow:hidden;">
              ${d.avatar.photo_url ? `<img src="${d.avatar.photo_url}" alt="${d.ceo_name}" style="width:100%;height:100%;object-fit:cover;" onerror="this.remove();" />` : d.avatar.initials}
            </span>
            <span><strong class="bb-amber">${d.ticker}</strong> &bull; <strong class="bb-cyan">${d.ceo_name}</strong> <span class="bb-muted">(${d.company_name})</span></span>
          </td>
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

    tbody.querySelectorAll("tr[data-dossier-id]").forEach((tr) => {
      tr.addEventListener("click", () => {
        const id = tr.getAttribute("data-dossier-id");
        const found = payload.ceo_dossiers.find((item) => item.id === id);
        if (found) openCeoModal(found);
      });
    });
  }

  function refreshCharts() {
    if (!state.payload || !window.BlackCeoCharts) return;
    const gpOpts = {
      strategy: state.strategy,
      scale: state.scale,
      showGross: state.showGross,
      cursorIdx: state.gpCursor,
    };
    window.BlackCeoCharts.renderEquityCurve("chart-equity-curve", state.payload, gpOpts);
    window.BlackCeoCharts.renderDrawdown("chart-drawdown", state.payload, gpOpts);
    window.BlackCeoCharts.renderRollingAlpha("chart-rolling-alpha", state.payload, gpOpts);
    window.BlackCeoCharts.renderConstituentBreadth("chart-constituent-count", state.payload);
    window.BlackCeoCharts.renderSectorBars("chart-sector-bars", state.payload);
    window.BlackCeoCharts.renderEcstRateChart("chart-ecst-rates", state.payload, state.showYieldOverlay);
    window.BlackCeoCharts.renderGarch3DSurface("chart-garch-3d", state.garchSpinAngle, state.garchShowPoints);
    window.BlackCeoCharts.renderVolOverlayChart("chart-vol-overlay", state.payload, state.volCursor);
    window.BlackCeoCharts.renderShockDecayChart("chart-shock-decay", state.shockSign);
    window.BlackCeoCharts.renderFF5Factors("chart-ff5-factors", state.payload, gpOpts);
    window.BlackCeoCharts.renderMonteCarlo("chart-mc-bootstrap", state.payload, gpOpts, state.mcResampleOffset);
    window.BlackCeoCharts.renderLjungBoxChart("chart-ljung-box");
  }

  async function pollLiveMarket() {
    try {
      const resp = await fetch("/api/live_market", { signal: AbortSignal.timeout(1500) });
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
    if (launchBtn) launchBtn.addEventListener("click", animateTypeAndLaunchDes);

    const strategySelect = document.getElementById("strategy-select");
    if (strategySelect) {
      strategySelect.addEventListener("change", (e) => {
        state.strategy = e.target.value;
        renderDynamicCommentary();
        renderDesMiniChart();
        renderMacroTable(state.payload);
        updateGpCursorTelemetry();
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

    const grossBtn = document.getElementById("gp-gross-toggle");
    if (grossBtn) {
      grossBtn.addEventListener("click", () => {
        state.showGross = !state.showGross;
        grossBtn.classList.toggle("bb-mini-btn--active", state.showGross);
        grossBtn.textContent = `GROSS-VS-NET: ${state.showGross ? "ON" : "OFF"}`;
        refreshCharts();
      });
    }

    const simBtn = document.getElementById("sim-ticks-btn");
    if (simBtn) {
      simBtn.addEventListener("click", () => {
        state.simulateTicks = !state.simulateTicks;
        simBtn.textContent = `[SIMULATE TICKS: ${state.simulateTicks ? "ON" : "OFF"}]`;
        renderMarketTables(state.marketData);
      });
    }

    // Page 1 Mini Transport
    const miniStart = document.getElementById("des-mini-start");
    const miniPlay = document.getElementById("des-mini-play");
    const miniEnd = document.getElementById("des-mini-end");
    if (miniStart) miniStart.addEventListener("click", () => { state.miniPlaying = false; state.miniCursor = 50; renderDesMiniChart(); });
    if (miniPlay) miniPlay.addEventListener("click", () => {
      state.miniPlaying = !state.miniPlaying;
      const maxLen = (state.payload?.series?.dates?.length || 6764) - 1;
      if (state.miniPlaying && state.miniCursor >= maxLen - 50) state.miniCursor = 50;
    });
    if (miniEnd) miniEnd.addEventListener("click", () => {
      state.miniPlaying = false;
      state.miniCursor = (state.payload?.series?.dates?.length || 6764) - 1;
      renderDesMiniChart();
    });

    // Page 3 GP Synchronized Transport
    const gpStart = document.getElementById("gp-start-btn");
    const gpPlay = document.getElementById("gp-play-btn");
    const gpEnd = document.getElementById("gp-end-btn");
    if (gpStart) gpStart.addEventListener("click", () => {
      state.gpPlaying = false;
      state.gpCursor = 0;
      updateGpCursorTelemetry();
      refreshCharts();
    });
    if (gpPlay) gpPlay.addEventListener("click", () => {
      state.gpPlaying = !state.gpPlaying;
      const maxLen = (state.payload?.series?.dates?.length || 6764) - 1;
      if (state.gpPlaying && state.gpCursor >= maxLen - 50) state.gpCursor = 0;
    });
    if (gpEnd) gpEnd.addEventListener("click", () => {
      state.gpPlaying = false;
      state.gpCursor = (state.payload?.series?.dates?.length || 6764) - 1;
      renderDynamicCommentary();
      updateGpCursorTelemetry();
      refreshCharts();
    });

    // Page 4 ECST Controls
    const yieldBtn = document.getElementById("ecst-yield-toggle");
    const regimeSel = document.getElementById("regime-select");
    if (yieldBtn) {
      yieldBtn.addEventListener("click", () => {
        state.showYieldOverlay = !state.showYieldOverlay;
        yieldBtn.classList.toggle("bb-mini-btn--active", state.showYieldOverlay);
        yieldBtn.textContent = `10Y YIELD OVERLAY: ${state.showYieldOverlay ? "ON" : "OFF"}`;
        refreshCharts();
      });
    }
    if (regimeSel) {
      regimeSel.addEventListener("change", (e) => {
        state.selectedRegime = e.target.value;
        renderMacroTable(state.payload);
      });
    }

    // Page 5 VOL Controls
    const spinBtn = document.getElementById("vol-spin-btn");
    const shockPos = document.getElementById("shock-pos-btn");
    const shockNeg = document.getElementById("shock-neg-btn");
    const volStart = document.getElementById("vol-start-btn");
    const volPlay = document.getElementById("vol-play-btn");
    const volEnd = document.getElementById("vol-end-btn");

    if (spinBtn) {
      spinBtn.addEventListener("click", () => {
        state.garchAutoSpin = !state.garchAutoSpin;
        spinBtn.classList.toggle("bb-mini-btn--active", state.garchAutoSpin);
        spinBtn.textContent = `3D AUTO-SPIN: ${state.garchAutoSpin ? "ON" : "OFF"}`;
      });
    }
    const ptsBtn = document.getElementById("vol-points-btn");
    if (ptsBtn) {
      ptsBtn.addEventListener("click", () => {
        state.garchShowPoints = !state.garchShowPoints;
        ptsBtn.classList.toggle("bb-mini-btn--active", state.garchShowPoints);
        ptsBtn.textContent = `[SHOW POINTS: ${state.garchShowPoints ? "ON" : "OFF"}]`;
        refreshCharts();
      });
    }
    if (shockPos && shockNeg) {
      shockPos.addEventListener("click", () => {
        state.shockSign = 1;
        shockPos.classList.add("bb-mini-btn--active");
        shockNeg.classList.remove("bb-mini-btn--active");
        refreshCharts();
      });
      shockNeg.addEventListener("click", () => {
        state.shockSign = -1;
        shockNeg.classList.add("bb-mini-btn--active");
        shockPos.classList.remove("bb-mini-btn--active");
        refreshCharts();
      });
    }
    if (volStart) volStart.addEventListener("click", () => { state.volPlaying = false; state.volCursor = 0; updateVolCursorReadout(); refreshCharts(); });
    if (volPlay) volPlay.addEventListener("click", () => {
      state.volPlaying = !state.volPlaying;
      const maxLen = (state.payload?.series?.dates?.length || 6764) - 1;
      if (state.volPlaying && state.volCursor >= maxLen - 50) state.volCursor = 0;
    });
    if (volEnd) volEnd.addEventListener("click", () => {
      state.volPlaying = false;
      state.volCursor = (state.payload?.series?.dates?.length || 6764) - 1;
      updateVolCursorReadout();
      refreshCharts();
    });

    // Page 6 Resample & Export Audit Controls
    const resampleBtn = document.getElementById("resample-mc-btn");
    const exportBtn = document.getElementById("export-audit-btn");
    if (resampleBtn) {
      resampleBtn.addEventListener("click", () => {
        let frames = 0;
        const anim = setInterval(() => {
          frames += 1;
          state.mcResampleOffset = (Math.random() - 0.5) * 0.45;
          refreshCharts();
          if (frames >= 10) {
            clearInterval(anim);
            state.mcResampleOffset = 0;
            refreshCharts();
          }
        }, 65);
      });
    }
    if (exportBtn) {
      exportBtn.addEventListener("click", () => {
        if (!state.payload) return;
        const audit = {
          exported_at: new Date().toISOString(),
          metadata: state.payload.metadata,
          summary_metrics: state.payload.summary_metrics,
          monte_carlo: state.payload.monte_carlo,
        };
        const blob = new Blob([JSON.stringify(audit, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `BCEOS_IC_Audit_${state.payload.metadata.horizon_end}.json`;
        a.click();
        URL.revokeObjectURL(url);
      });
    }

    // Roster search, filter & modal close
    const rosterSearch = document.getElementById("roster-search");
    const statusFilter = document.getElementById("roster-status-filter");
    const modal = document.getElementById("ceo-modal");
    const modalClose = document.getElementById("ceo-modal-close");

    if (rosterSearch) rosterSearch.addEventListener("input", (e) => { state.search = e.target.value; renderRosterTable(state.payload); });
    if (statusFilter) statusFilter.addEventListener("change", (e) => { state.statusFilter = e.target.value; renderRosterTable(state.payload); });
    if (modalClose && modal) modalClose.addEventListener("click", () => { modal.hidden = true; });
    if (modal) modal.addEventListener("click", (e) => { if (e.target === modal) modal.hidden = true; });

    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && modal && !modal.hidden) {
        modal.hidden = true;
        return;
      }
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

    // Unified playback & 3D surface loop
    setInterval(() => {
      if (!state.payload) return;
      const maxLen = state.payload.series.dates.length - 1;

      if (state.miniPlaying) {
        const spd = Number(document.getElementById("des-mini-speed")?.value || 5);
        state.miniCursor = Math.min(maxLen, state.miniCursor + Math.max(1, Math.round(spd * 12)));
        if (state.miniCursor >= maxLen) state.miniPlaying = false;
        renderDesMiniChart();
      }
      if (state.gpPlaying && state.activeScreen === "3") {
        const spd = Number(document.getElementById("gp-speed-select")?.value || 10);
        state.gpCursor = Math.min(maxLen, state.gpCursor + Math.max(1, Math.round(spd * 12)));
        if (state.gpCursor >= maxLen) state.gpPlaying = false;
        updateGpCursorTelemetry();
        refreshCharts();
      }
      if (state.volPlaying && state.activeScreen === "5") {
        const spd = Number(document.getElementById("vol-speed-select")?.value || 10);
        state.volCursor = Math.min(maxLen, state.volCursor + Math.max(1, Math.round(spd * 12)));
        if (state.volCursor >= maxLen) state.volPlaying = false;
        updateVolCursorReadout();
        refreshCharts();
      }
      if (state.garchAutoSpin && state.activeScreen === "5") {
        state.garchSpinAngle += 0.04;
        window.BlackCeoCharts?.renderGarch3DSurface("chart-garch-3d", state.garchSpinAngle, state.garchShowPoints);
      }
    }, 55);

    setInterval(() => {
      if (state.simulateTicks && state.marketData?.equities) {
        state.marketData.equities.forEach((eq) => {
          const delta = Number(((Math.random() - 0.48) * (eq.last * 0.0012)).toFixed(2));
          eq.last = Number((eq.last + delta).toFixed(2));
          eq.chg = Number((eq.chg + delta).toFixed(2));
        });
        if (state.marketData.intraday?.spx) {
          state.marketData.intraday.spx.push(state.marketData.equities[0].last);
          state.marketData.intraday.spx.shift();
        }
        renderMarketTables(state.marketData);
      }
    }, 1800);

    await pollLiveMarket();
    setInterval(pollLiveMarket, 20000);

    try {
      const resp = await fetch(`data/dashboard_payload.json?t=${Date.now()}`);
      if (!resp.ok) throw new Error("Payload HTTP error");
      state.payload = await resp.json();
      const maxLen = state.payload.series.dates.length - 1;
      state.miniCursor = maxLen;
      state.gpCursor = maxLen;
      state.volCursor = maxLen;

      syncMetadataDates();
      renderDynamicCommentary();
      renderDesMiniChart();
      updateGpCursorTelemetry();
      updateVolCursorReadout();
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
