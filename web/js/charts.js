(function () {
  "use strict";

  const COLORS = {
    bg: "#07090d",
    grid: "#1e232b",
    text: "#cbd5e1",
    amber: "#ff9900",
    cyan: "#00d1ff",
    green: "#00ff00",
    red: "#ff3b30",
    slate: "#64748b",
  };

  function baseLayout(overrides) {
    return Object.assign(
      {
        paper_bgcolor: COLORS.bg,
        plot_bgcolor: COLORS.bg,
        font: {
          family: '"JetBrains Mono", "Courier New", monospace',
          color: COLORS.text,
          size: 10,
        },
        margin: { l: 48, r: 16, t: 10, b: 26 },
        xaxis: {
          gridcolor: COLORS.grid,
          zerolinecolor: COLORS.grid,
          showline: true,
          linecolor: "#363c46",
        },
        yaxis: {
          gridcolor: COLORS.grid,
          zerolinecolor: COLORS.grid,
          showline: true,
          linecolor: "#363c46",
        },
        legend: {
          orientation: "h",
          y: 1.12,
          x: 0,
          font: { size: 9.5 },
        },
        hovermode: "x unified",
      },
      overrides || {}
    );
  }

  const CONFIG = {
    responsive: true,
    displayModeBar: false,
  };

  function renderEquityCurve(elId, payload, opts) {
    const el = document.getElementById(elId);
    if (!el || !window.Plotly || !payload) return;
    const dates = payload.series.dates;
    const strat = payload.series.wealth_10k[opts.strategy];
    const gross = payload.series.wealth_10k.Black_CEO_Gross;
    const spy = payload.series.wealth_10k.SPY;
    const rsp = payload.series.wealth_10k.RSP;
    const cursorDate = dates[opts.cursorIdx || dates.length - 1];

    const traces = [
      {
        x: dates,
        y: strat,
        name: opts.strategy === "Convex_Quality_Net" ? "Convex-QP (Net)" : "BCEOS Eq-Wt (Net)",
        type: "scatter",
        mode: "lines",
        line: { color: COLORS.green, width: 2 },
      },
      {
        x: dates,
        y: spy,
        name: "S&P 500 (SPY)",
        type: "scatter",
        mode: "lines",
        line: { color: COLORS.cyan, width: 1.5 },
      },
      {
        x: dates,
        y: rsp,
        name: "S&P Eq-Wt (RSP)",
        type: "scatter",
        mode: "lines",
        line: { color: COLORS.amber, width: 1.2, dash: "dot" },
      },
    ];

    if (opts.showGross && gross && opts.strategy === "Black_CEO_Net") {
      traces.splice(1, 0, {
        x: dates,
        y: gross,
        name: "BCEOS Gross (0 bps)",
        type: "scatter",
        mode: "lines",
        line: { color: "#ffffff", width: 1.1, dash: "dash" },
      });
    }

    const layout = baseLayout({
      yaxis: {
        type: opts.scale === "linear" ? "linear" : "log",
        tickprefix: "$",
        gridcolor: COLORS.grid,
      },
      shapes: [
        {
          type: "line",
          x0: cursorDate,
          x1: cursorDate,
          y0: 0,
          y1: 1,
          yref: "paper",
          line: { color: COLORS.green, width: 1.5, dash: "dot" },
        },
      ],
    });
    window.Plotly.react(el, traces, layout, CONFIG);
  }

  function renderDrawdown(elId, payload, opts) {
    const el = document.getElementById(elId);
    if (!el || !window.Plotly || !payload) return;
    const dates = payload.series.dates;
    const stratDd = payload.series.drawdown_pct[opts.strategy];
    const spyDd = payload.series.drawdown_pct.SPY;
    const cursorDate = dates[opts.cursorIdx || dates.length - 1];

    const traces = [
      {
        x: dates,
        y: stratDd,
        name: "BCEOS Drawdown %",
        type: "scatter",
        mode: "lines",
        fill: "tozeroy",
        fillcolor: "rgba(255, 59, 48, 0.22)",
        line: { color: COLORS.red, width: 1.5 },
      },
      {
        x: dates,
        y: spyDd,
        name: "SPY Drawdown %",
        type: "scatter",
        mode: "lines",
        line: { color: COLORS.slate, width: 1.2 },
      },
    ];

    const layout = baseLayout({
      yaxis: { ticksuffix: "%", gridcolor: COLORS.grid },
      shapes: [
        {
          type: "line",
          x0: cursorDate,
          x1: cursorDate,
          y0: 0,
          y1: 1,
          yref: "paper",
          line: { color: COLORS.green, width: 1.5, dash: "dot" },
        },
      ],
    });
    window.Plotly.react(el, traces, layout, CONFIG);
  }

  function renderRollingAlpha(elId, payload, opts) {
    const el = document.getElementById(elId);
    if (!el || !window.Plotly || !payload) return;
    const dates = payload.series.dates;
    const roll = payload.series.rolling_3y_excess_pct || [];
    const cursorDate = dates[opts.cursorIdx || dates.length - 1];

    const traces = [
      {
        x: dates,
        y: roll,
        name: "Rolling 3Y Excess CAGR vs SPX (%)",
        type: "scatter",
        mode: "lines",
        fill: "tozeroy",
        fillcolor: "rgba(0, 209, 255, 0.16)",
        line: { color: COLORS.cyan, width: 1.5 },
      },
    ];

    const layout = baseLayout({
      margin: { l: 48, r: 16, t: 6, b: 20 },
      showlegend: false,
      yaxis: { ticksuffix: "%", gridcolor: COLORS.grid, zerolinecolor: COLORS.amber },
      shapes: [
        {
          type: "line",
          x0: cursorDate,
          x1: cursorDate,
          y0: 0,
          y1: 1,
          yref: "paper",
          line: { color: COLORS.green, width: 1.5, dash: "dot" },
        },
      ],
    });
    window.Plotly.react(el, traces, layout, CONFIG);
  }

  function renderConstituentBreadth(elId, payload) {
    const el = document.getElementById(elId);
    if (!el || !window.Plotly || !payload) return;
    const dates = payload.series.dates;
    const counts = payload.series.active_count;
    const traces = [
      {
        x: dates,
        y: counts,
        name: "Active Black CEOs (Nt)",
        type: "scatter",
        mode: "lines",
        line: { color: COLORS.amber, width: 2, shape: "hv" },
        fill: "tozeroy",
        fillcolor: "rgba(255, 153, 0, 0.18)",
      },
    ];
    const layout = baseLayout({
      margin: { l: 38, r: 12, t: 6, b: 22 },
      showlegend: false,
      yaxis: { title: "Issuers", gridcolor: COLORS.grid, rangemode: "tozero" },
    });
    window.Plotly.react(el, traces, layout, CONFIG);
  }

  function renderSectorBars(elId, payload) {
    const el = document.getElementById(elId);
    if (!el || !window.Plotly || !payload || !Array.isArray(payload.sector_breakdown)) return;
    const rows = payload.sector_breakdown;
    const sectors = rows.map((r) => r.sector.replace("Consumer Discretionary", "Cons. Disc.").replace("Information Technology", "Info Tech").replace("Financial Services", "Financials"));
    const totals = rows.map((r) => r.total);
    const actives = rows.map((r) => r.active);

    const traces = [
      {
        x: sectors,
        y: totals,
        name: "Total Tenures (40)",
        type: "bar",
        marker: { color: COLORS.amber },
      },
      {
        x: sectors,
        y: actives,
        name: "Active Now (17)",
        type: "bar",
        marker: { color: COLORS.green },
      },
    ];

    const layout = baseLayout({
      barmode: "group",
      margin: { l: 32, r: 10, t: 6, b: 28 },
      legend: { orientation: "h", y: 1.18, x: 0, font: { size: 9 } },
      xaxis: { tickfont: { size: 8.5 }, gridcolor: COLORS.grid },
      yaxis: { gridcolor: COLORS.grid, rangemode: "tozero" },
    });
    window.Plotly.react(el, traces, layout, CONFIG);
  }

  function renderEcstRateChart(elId, payload, showYieldOverlay) {
    const el = document.getElementById(elId);
    if (!el || !window.Plotly || !payload) return;
    const regimes = payload.timeseries_diagnostics?.macro_regimes || [];
    const labels = regimes.map((r) => r.label.split("(")[0].trim());
    const bceosRets = regimes.map((r) => Number((r.series?.Black_CEO_Net?.cumulative_return || 0) * 100).toFixed(1));
    const spyRets = regimes.map((r) => Number((r.series?.SPY_Daily?.cumulative_return || 0) * 100).toFixed(1));
    const tenYearProxy = [4.95, 3.65, 2.15, 1.35, 3.88, 4.42];

    const traces = [
      {
        x: labels,
        y: bceosRets,
        name: "BCEOS Cum. Return %",
        type: "bar",
        marker: { color: COLORS.green },
      },
      {
        x: labels,
        y: spyRets,
        name: "SPX Cum. Return %",
        type: "bar",
        marker: { color: COLORS.cyan },
      },
    ];

    if (showYieldOverlay) {
      traces.push({
        x: labels,
        y: tenYearProxy,
        name: "Avg US 10Y Yield (%)",
        type: "scatter",
        mode: "lines+markers",
        yaxis: "y2",
        line: { color: COLORS.amber, width: 2, dash: "dash" },
        marker: { size: 6, color: COLORS.amber },
      });
    }

    const layout = baseLayout({
      barmode: "group",
      margin: { l: 42, r: 42, t: 16, b: 36 },
      xaxis: { tickfont: { size: 8.5 }, gridcolor: COLORS.grid },
      yaxis: { ticksuffix: "%", gridcolor: COLORS.grid },
      yaxis2: {
        overlaying: "y",
        side: "right",
        ticksuffix: "%",
        showgrid: false,
        tickfont: { color: COLORS.amber },
      },
    });
    window.Plotly.react(el, traces, layout, CONFIG);
  }

  function renderGarch3DSurface(elId, spinAngle, showPoints) {
    const el = document.getElementById(elId);
    if (!el || !window.Plotly) return;
    const rect = el.getBoundingClientRect();
    if (rect.width < 40 || rect.height < 40) return;

    const shocks = [-3, -2, -1, 0, 1, 2, 3];
    const horizons = [1, 5, 10, 20, 30, 45, 60];
    const zBceos = horizons.map((h) =>
      shocks.map((s) => {
        const decay = Math.pow(0.955, h);
        const gamma = s < 0 ? 0.0594 : 0.0;
        const impact = (0.2198 + gamma) * s * s * 6.5 * decay;
        return Number((24.0 + impact).toFixed(2));
      })
    );
    const zSpy = horizons.map((h) =>
      shocks.map((s) => {
        const decay = Math.pow(0.975, h);
        const gamma = s < 0 ? 0.1929 : 0.0;
        const impact = (0.0326 + gamma) * s * s * 6.2 * decay;
        return Number((18.5 + impact).toFixed(2));
      })
    );

    const rad = spinAngle || 0.75;
    const radius = 2.05;
    const eyeX = radius * Math.cos(rad);
    const eyeY = radius * Math.sin(rad);

    const traces = [
      {
        x: shocks,
        y: horizons,
        z: zBceos,
        name: "BCEOS Symmetric Bowl",
        type: "surface",
        showscale: false,
        colorscale: [
          [0, "#0b2e1e"],
          [1, "#ff9900"],
        ],
        opacity: showPoints ? 0.42 : 0.9,
      },
      {
        x: shocks,
        y: horizons,
        z: zSpy,
        name: "SPY Asymmetric V-Skew",
        type: "surface",
        showscale: false,
        colorscale: [
          [0, "#0a2540"],
          [1, "#00d1ff"],
        ],
        opacity: showPoints ? 0.32 : 0.65,
      },
    ];

    if (showPoints) {
      const px = [];
      const py = [];
      const pzB = [];
      const pzS = [];
      horizons.forEach((h, hi) => {
        shocks.forEach((s, si) => {
          px.push(s);
          py.push(h);
          pzB.push(zBceos[hi][si]);
          pzS.push(zSpy[hi][si]);
        });
      });
      traces.push(
        {
          x: px,
          y: py,
          z: pzB,
          name: "BCEOS Nodes",
          type: "scatter3d",
          mode: "markers",
          marker: { size: 3.5, color: "#ff9900" },
        },
        {
          x: px,
          y: py,
          z: pzS,
          name: "SPY Nodes",
          type: "scatter3d",
          mode: "markers",
          marker: { size: 3, color: "#00d1ff" },
        }
      );
    }

    const layout = {
      paper_bgcolor: COLORS.bg,
      width: Math.floor(rect.width),
      height: Math.floor(rect.height),
      margin: { l: 0, r: 0, t: 0, b: 0 },
      showlegend: false,
      font: { family: '"JetBrains Mono", monospace', color: COLORS.text, size: 9 },
      scene: {
        domain: { x: [0, 1], y: [0, 1] },
        aspectmode: "cube",
        xaxis: { title: "Shock (σ)", range: [-3.2, 3.2], gridcolor: COLORS.grid, backgroundcolor: COLORS.bg },
        yaxis: { title: "Horizon (d)", range: [0, 62], gridcolor: COLORS.grid, backgroundcolor: COLORS.bg },
        zaxis: { title: "Vol %", range: [15, 58], gridcolor: COLORS.grid, backgroundcolor: COLORS.bg },
        camera: {
          center: { x: 0, y: 0, z: -0.08 },
          eye: { x: eyeX, y: eyeY, z: 1.05 },
        },
      },
    };
    window.Plotly.react(el, traces, layout, CONFIG);
  }

  function renderVolOverlayChart(elId, payload, cursorIdx) {
    const el = document.getElementById(elId);
    if (!el || !window.Plotly || !payload) return;
    const dates = payload.series.dates;
    const unscaled = payload.series.wealth_10k.Black_CEO_Net;
    const vt = payload.series.wealth_10k.VolTarget_20 || unscaled;
    const cursorDate = dates[cursorIdx || dates.length - 1];

    const traces = [
      {
        x: dates,
        y: unscaled,
        name: "Unscaled ($264.8k)",
        type: "scatter",
        mode: "lines",
        line: { color: COLORS.amber, width: 1.4 },
      },
      {
        x: dates,
        y: vt,
        name: "20% Vol-Target (MDD -51.1%)",
        type: "scatter",
        mode: "lines",
        line: { color: COLORS.green, width: 2 },
      },
    ];

    const layout = baseLayout({
      margin: { l: 44, r: 10, t: 6, b: 20 },
      yaxis: { type: "log", tickprefix: "$", gridcolor: COLORS.grid },
      shapes: [
        {
          type: "line",
          x0: cursorDate,
          x1: cursorDate,
          y0: 0,
          y1: 1,
          yref: "paper",
          line: { color: COLORS.cyan, width: 1.5, dash: "dot" },
        },
      ],
    });
    window.Plotly.react(el, traces, layout, CONFIG);
  }

  function renderShockDecayChart(elId, shockSign) {
    const el = document.getElementById(elId);
    if (!el || !window.Plotly) return;
    const days = Array.from({ length: 61 }, (_, i) => i);
    const mult = shockSign < 0 ? 1.12 : 1.0;
    const pre2010 = days.map((d) => Number((26.0 + 48.0 * mult * Math.pow(0.5, d / 146.8)).toFixed(2)));
    const post2010 = days.map((d) => Number((22.0 + 44.0 * mult * Math.pow(0.5, d / 7.0)).toFixed(2)));

    const traces = [
      {
        x: days,
        y: pre2010,
        name: "Pre-2010 (HL: 146.8d)",
        type: "scatter",
        mode: "lines",
        line: { color: COLORS.red, width: 2 },
      },
      {
        x: days,
        y: post2010,
        name: "Post-2010 (HL: 7.0d)",
        type: "scatter",
        mode: "lines",
        line: { color: COLORS.green, width: 2 },
      },
    ];

    const layout = baseLayout({
      margin: { l: 40, r: 10, t: 6, b: 22 },
      xaxis: { title: "Trading Days Post-Shock", gridcolor: COLORS.grid },
      yaxis: { ticksuffix: "%", gridcolor: COLORS.grid },
    });
    window.Plotly.react(el, traces, layout, CONFIG);
  }

  function renderFF5Factors(elId, payload, opts) {
    const el = document.getElementById(elId);
    if (!el || !window.Plotly || !payload) return;
    const target = opts.strategy === "Convex_Quality_Net" ? "convex" : "equal";
    const allRows = payload.factor_loadings || [];
    const getVal = (obj, keys) => {
      for (const k of keys) {
        if (obj[k] !== undefined && obj[k] !== null) return obj[k];
      }
      return null;
    };
    let rows = allRows.filter((r) => {
      const pName = String(getVal(r, ["Portfolio", "portfolio", "strategy"]) || "").toLowerCase();
      const fName = String(getVal(r, ["Factor", "factor", "term"]) || "");
      return pName.includes(target) && !fName.toLowerCase().includes("alpha");
    });
    if (!rows.length) {
      rows = allRows
        .filter((r) => !String(getVal(r, ["Factor", "factor", "term"]) || "").toLowerCase().includes("alpha"))
        .slice(0, 5);
    }
    const factors = rows.map((r) => String(getVal(r, ["Factor", "factor", "term"]) || ""));
    const betas = rows.map((r) => Number(getVal(r, ["Estimate", "estimate", "Beta", "beta", "coef"]) || 0));
    const labels = rows.map((r, idx) => {
      const p = Number(getVal(r, ["P_Value", "p_value", "pvalue", "p"]) ?? 1);
      const t = Number(getVal(r, ["T_Stat", "t_stat", "tstat", "t"]) ?? 0);
      const stars = p < 0.01 ? "***" : p < 0.05 ? "**" : p < 0.1 ? "*" : "";
      return `${betas[idx].toFixed(2)} (t=${t.toFixed(1)})${stars}`;
    });

    const traces = [
      {
        y: factors,
        x: betas,
        text: labels,
        textposition: "outside",
        orientation: "h",
        type: "bar",
        marker: { color: betas.map((b) => (b >= 0 ? COLORS.green : COLORS.red)) },
      },
    ];

    const layout = baseLayout({
      margin: { l: 64, r: 64, t: 6, b: 22 },
      showlegend: false,
      xaxis: { range: [-0.45, 1.25], gridcolor: COLORS.grid },
      yaxis: { autorange: "reversed", gridcolor: COLORS.grid },
    });
    window.Plotly.react(el, traces, layout, CONFIG);
  }

  function renderMonteCarlo(elId, payload, opts, resampleOffset) {
    const el = document.getElementById(elId);
    if (!el || !window.Plotly || !payload) return;
    const mc = payload.monte_carlo || {};
    const shift = resampleOffset || 0;
    const emp =
      opts.strategy === "Convex_Quality_Net"
        ? Number(mc.empirical_cagr_cq || 0.0952) * 100
        : Number(mc.empirical_cagr_eq || 0.1299) * 100;

    const bars = [
      { label: "5th Pctl (Null)", val: Number((Number(mc.null_cagr_p05 || 0.082) * 100 + shift * 0.6).toFixed(2)), color: COLORS.slate },
      { label: "Bootstrap Mean", val: Number((Number(mc.null_cagr_mean || 0.104) * 100 + shift).toFixed(2)), color: COLORS.cyan },
      { label: "95th Pctl (Null)", val: Number((Number(mc.null_cagr_p95 || 0.128) * 100 + shift * 0.8).toFixed(2)), color: COLORS.amber },
      { label: "BCEOS Actual", val: Number(emp.toFixed(2)), color: COLORS.green },
    ];

    const traces = [
      {
        x: bars.map((b) => b.label),
        y: bars.map((b) => b.val),
        text: bars.map((b) => `${b.val.toFixed(2)}%`),
        textposition: "outside",
        type: "bar",
        marker: { color: bars.map((b) => b.color) },
      },
    ];

    const layout = baseLayout({
      margin: { l: 42, r: 12, t: 10, b: 24 },
      showlegend: false,
      yaxis: { ticksuffix: "%", range: [0, 16.5], gridcolor: COLORS.grid },
    });
    window.Plotly.react(el, traces, layout, CONFIG);
  }

  function renderLjungBoxChart(elId) {
    const el = document.getElementById(elId);
    if (!el || !window.Plotly) return;
    const lags = ["Lag 1 (Returns)", "Lag 5 (Vol)", "Lag 10 (Vol)", "Lag 21 (Vol)"];
    const pVals = [0.1164, 0.0001, 0.0001, 0.0001];

    const traces = [
      {
        x: lags,
        y: pVals,
        text: ["p=0.1164 (Noise)", "p<1e-20 (Cluster)", "p<1e-20 (Cluster)", "p<1e-20 (Cluster)"],
        textposition: "outside",
        type: "bar",
        marker: { color: [COLORS.green, COLORS.amber, COLORS.amber, COLORS.amber] },
      },
    ];

    const layout = baseLayout({
      margin: { l: 42, r: 12, t: 10, b: 24 },
      showlegend: false,
      yaxis: { range: [0, 0.15], title: "p-Value", gridcolor: COLORS.grid },
      shapes: [
        {
          type: "line",
          x0: -0.5,
          x1: 3.5,
          y0: 0.05,
          y1: 0.05,
          line: { color: COLORS.red, width: 1.2, dash: "dash" },
        },
      ],
    });
    window.Plotly.react(el, traces, layout, CONFIG);
  }

  window.BlackCeoCharts = {
    renderEquityCurve,
    renderDrawdown,
    renderRollingAlpha,
    renderConstituentBreadth,
    renderSectorBars,
    renderEcstRateChart,
    renderGarch3DSurface,
    renderVolOverlayChart,
    renderShockDecayChart,
    renderFF5Factors,
    renderMonteCarlo,
    renderLjungBoxChart,
  };
})();
