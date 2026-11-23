/* global Plotly */
window.BlackCeoCharts = (function () {
  "use strict";

  const BASE_LAYOUT = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(6,9,15,0.55)",
    font: {
      family: "JetBrains Mono, SFMono-Regular, Menlo, monospace",
      color: "#cbd5e1",
      size: 11,
    },
    margin: { t: 24, r: 20, b: 40, l: 56 },
    xaxis: {
      gridcolor: "rgba(30,41,59,0.65)",
      zerolinecolor: "rgba(51,65,85,0.8)",
    },
    yaxis: {
      gridcolor: "rgba(30,41,59,0.65)",
      zerolinecolor: "rgba(51,65,85,0.8)",
    },
    legend: {
      orientation: "h",
      y: 1.08,
      x: 0,
      bgcolor: "rgba(0,0,0,0)",
    },
    hovermode: "x unified",
  };

  const PLOT_CONFIG = {
    responsive: true,
    displayModeBar: false,
  };

  function filterWindow(dates, values, range) {
    if (!range) {
      return { x: dates, y: values };
    }
    const x = [];
    const y = [];
    for (let i = 0; i < dates.length; i += 1) {
      if (dates[i] >= range.start && dates[i] <= range.end) {
        x.push(dates[i]);
        y.push(values[i]);
      }
    }
    return { x, y };
  }

  function rebaseSeries(values) {
    const first = values.find((v) => v !== null && v !== undefined && v > 0);
    if (!first) {
      return values;
    }
    return values.map((v) => (v === null || v === undefined ? null : Number(((v / first) * 10000).toFixed(2))));
  }

  function renderEquityCurve(elId, payload, opts) {
    const el = document.getElementById(elId);
    if (!el || typeof Plotly === "undefined") {
      return;
    }
    const dates = payload.series.dates;
    const stratKey = opts.strategy || "Black_CEO_Net";
    const stratLabel =
      stratKey === "Convex_Quality_Net" ? "Black CEO Convex-Quality QP" : "Black CEO Equal-Weight (Net)";

    const stratWin = filterWindow(dates, payload.series.wealth_10k[stratKey], opts.range);
    const spyWin = filterWindow(dates, payload.series.wealth_10k.SPY, opts.range);
    const rspWin = filterWindow(dates, payload.series.wealth_10k.RSP, opts.range);

    const traces = [
      {
        x: stratWin.x,
        y: opts.range ? rebaseSeries(stratWin.y) : stratWin.y,
        name: stratLabel,
        type: "scatter",
        mode: "lines",
        line: { color: "#00e676", width: 2.3 },
      },
      {
        x: spyWin.x,
        y: opts.range ? rebaseSeries(spyWin.y) : spyWin.y,
        name: "SPY (S&P 500 Cap-Wtd)",
        type: "scatter",
        mode: "lines",
        line: { color: "#00d4ff", width: 1.8 },
      },
      {
        x: rspWin.x,
        y: opts.range ? rebaseSeries(rspWin.y) : rspWin.y,
        name: "RSP (S&P 500 Eq-Wtd)",
        type: "scatter",
        mode: "lines",
        line: { color: "#ff9f1c", width: 1.5, dash: "dot" },
      },
    ];

    const layout = {
      ...BASE_LAYOUT,
      yaxis: {
        ...BASE_LAYOUT.yaxis,
        type: opts.scale === "linear" ? "linear" : "log",
        tickprefix: "$",
      },
    };

    Plotly.react(el, traces, layout, PLOT_CONFIG);
  }

  function renderDrawdown(elId, payload, opts) {
    const el = document.getElementById(elId);
    if (!el || typeof Plotly === "undefined") {
      return;
    }
    const dates = payload.series.dates;
    const stratKey = opts.strategy || "Black_CEO_Net";
    const stratWin = filterWindow(dates, payload.series.drawdown_pct[stratKey], opts.range);
    const spyWin = filterWindow(dates, payload.series.drawdown_pct.SPY, opts.range);

    const traces = [
      {
        x: stratWin.x,
        y: stratWin.y,
        name: "Black CEO Strategy DD %",
        type: "scatter",
        mode: "lines",
        fill: "tozeroy",
        fillcolor: "rgba(255, 82, 82, 0.18)",
        line: { color: "#ff5252", width: 1.6 },
      },
      {
        x: spyWin.x,
        y: spyWin.y,
        name: "SPY DD %",
        type: "scatter",
        mode: "lines",
        line: { color: "#64748b", width: 1.3 },
      },
    ];

    const layout = {
      ...BASE_LAYOUT,
      yaxis: {
        ...BASE_LAYOUT.yaxis,
        ticksuffix: "%",
      },
    };

    Plotly.react(el, traces, layout, PLOT_CONFIG);
  }

  function renderConstituentBreadth(elId, payload, opts) {
    const el = document.getElementById(elId);
    if (!el || typeof Plotly === "undefined") {
      return;
    }
    const win = filterWindow(payload.series.dates, payload.series.active_count, opts.range);
    const traces = [
      {
        x: win.x,
        y: win.y,
        name: "Active Black CEOs (Nt)",
        type: "scatter",
        mode: "lines",
        fill: "tozeroy",
        fillcolor: "rgba(0, 212, 255, 0.18)",
        line: { color: "#00d4ff", width: 1.8, shape: "hv" },
      },
    ];

    const layout = {
      ...BASE_LAYOUT,
      yaxis: {
        ...BASE_LAYOUT.yaxis,
        title: "Active Issuers",
        rangemode: "tozero",
      },
    };

    Plotly.react(el, traces, layout, PLOT_CONFIG);
  }

  function renderFF5Factors(elId, payload) {
    const el = document.getElementById(elId);
    if (!el || typeof Plotly === "undefined" || !Array.isArray(payload.factor_loadings)) {
      return;
    }
    const rows = payload.factor_loadings.filter((r) => r.Factor !== "Alpha (Daily)");
    const factors = rows.map((r) => r.Factor);
    const betas = rows.map((r) => Number(r.Beta || r.Coefficient || 0));
    const colors = betas.map((b) => (b >= 0 ? "#00d4ff" : "#ff9f1c"));

    const traces = [
      {
        x: factors,
        y: betas,
        type: "bar",
        marker: { color: colors },
        text: betas.map((b) => b.toFixed(3)),
        textposition: "outside",
      },
    ];

    const layout = {
      ...BASE_LAYOUT,
      hovermode: "closest",
      yaxis: {
        ...BASE_LAYOUT.yaxis,
        title: "Factor Beta",
      },
    };

    Plotly.react(el, traces, layout, PLOT_CONFIG);
  }

  function renderMonteCarlo(elId, payload) {
    const el = document.getElementById(elId);
    if (!el || typeof Plotly === "undefined" || !payload.monte_carlo) {
      return;
    }
    const mc = payload.monte_carlo;
    const dist = mc.simulated_cagrs || mc.cagr_distribution || [];
    const empCagr = Number(mc.empirical_cagr || 0.1299) * (dist[0] > 1 ? 100 : 1);

    const traces = [
      {
        x: dist.length ? dist : [0.07, 0.08, 0.09, 0.095, 0.1, 0.105, 0.11, 0.12],
        type: "histogram",
        name: "10,000 Random S&P Cohorts",
        marker: { color: "rgba(100, 116, 139, 0.65)", line: { color: "#94a3b8", width: 1 } },
      },
    ];

    const layout = {
      ...BASE_LAYOUT,
      hovermode: "closest",
      shapes: [
        {
          type: "line",
          x0: empCagr,
          x1: empCagr,
          y0: 0,
          y1: 1,
          yref: "paper",
          line: { color: "#00e676", width: 2.5, dash: "dash" },
        },
      ],
      annotations: [
        {
          x: empCagr,
          y: 0.95,
          yref: "paper",
          text: "Black CEO Empirical",
          showarrow: true,
          arrowhead: 2,
          font: { color: "#00e676", size: 11 },
        },
      ],
    };

    Plotly.react(el, traces, layout, PLOT_CONFIG);
  }

  return {
    renderEquityCurve,
    renderDrawdown,
    renderConstituentBreadth,
    renderFF5Factors,
    renderMonteCarlo,
  };
})();
