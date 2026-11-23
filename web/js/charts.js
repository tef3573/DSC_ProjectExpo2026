/* global Plotly */
window.BlackCeoCharts = (function () {
  "use strict";

  const BASE_LAYOUT = {
    paper_bgcolor: "#000000",
    plot_bgcolor: "#05070a",
    font: {
      family: "JetBrains Mono, Courier New, Menlo, monospace",
      color: "#cbd5e1",
      size: 10.5,
    },
    margin: { t: 22, r: 16, b: 36, l: 52 },
    xaxis: {
      gridcolor: "#1e2530",
      zerolinecolor: "#383e47",
    },
    yaxis: {
      gridcolor: "#1e2530",
      zerolinecolor: "#383e47",
    },
    legend: {
      orientation: "h",
      y: 1.1,
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
        line: { color: "#00e676", width: 2 },
      },
      {
        x: spyWin.x,
        y: opts.range ? rebaseSeries(spyWin.y) : spyWin.y,
        name: "SPY (Cap-Wtd)",
        type: "scatter",
        mode: "lines",
        line: { color: "#00d4ff", width: 1.6 },
      },
      {
        x: rspWin.x,
        y: opts.range ? rebaseSeries(rspWin.y) : rspWin.y,
        name: "RSP (Eq-Wtd)",
        type: "scatter",
        mode: "lines",
        line: { color: "#ff9f1c", width: 1.4, dash: "dot" },
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
        name: "Black CEO DD %",
        type: "scatter",
        mode: "lines",
        fill: "tozeroy",
        fillcolor: "rgba(255, 82, 82, 0.2)",
        line: { color: "#ff5252", width: 1.5 },
      },
      {
        x: spyWin.x,
        y: spyWin.y,
        name: "SPY DD %",
        type: "scatter",
        mode: "lines",
        line: { color: "#94a3b8", width: 1.2 },
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
        fillcolor: "rgba(255, 159, 28, 0.2)",
        line: { color: "#ff9f1c", width: 1.7, shape: "hv" },
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

  function renderFF5Factors(elId, payload, opts) {
    const el = document.getElementById(elId);
    if (!el || typeof Plotly === "undefined" || !Array.isArray(payload.factor_loadings)) {
      return;
    }
    const targetScheme =
      opts.strategy === "Convex_Quality_Net" ? "Convex-Quality (Net)" : "Equal-Weight (Net)";
    const rows = payload.factor_loadings.filter(
      (r) => r.scheme === targetScheme && !String(r.parameter).startsWith("Alpha")
    );
    const factors = rows.map((r) => r.parameter);
    const betas = rows.map((r) => Number(r.estimate || 0));
    const tStats = rows.map((r) => `t=${Number(r.t_stat || 0).toFixed(2)}`);
    const colors = betas.map((b) => (b >= 0 ? "#00e676" : "#ff5252"));

    const traces = [
      {
        x: factors,
        y: betas,
        type: "bar",
        marker: { color: colors },
        text: betas.map((b, i) => `${b.toFixed(3)} (${tStats[i]})`),
        textposition: "outside",
      },
    ];

    const layout = {
      ...BASE_LAYOUT,
      hovermode: "closest",
      yaxis: {
        ...BASE_LAYOUT.yaxis,
        title: "Factor Loading (Beta)",
        range: [-0.45, 1.15],
      },
    };

    Plotly.react(el, traces, layout, PLOT_CONFIG);
  }

  function renderMonteCarlo(elId, payload, opts) {
    const el = document.getElementById(elId);
    if (!el || typeof Plotly === "undefined" || !payload.monte_carlo?.strategies) {
      return;
    }
    const targetScheme =
      opts.strategy === "Convex_Quality_Net" ? "Convex-Quality (Net)" : "Equal-Weight (Net)";
    const cagrBlock = payload.monte_carlo.strategies[targetScheme]?.cagr || {};
    const empPct = Number(cagrBlock.empirical_value || 0.1299) * 100;
    const p5 = Number(cagrBlock.ci_lower_95 || 0.082) * 100;
    const p50 = Number(cagrBlock.bootstrap_mean || 0.104) * 100;
    const p95 = Number(cagrBlock.ci_upper_95 || 0.128) * 100;

    const traces = [
      {
        x: ["5th Pct (Null)", "Bootstrap Mean", "95th Pct (Null)", "Black CEO Empirical"],
        y: [p5, p50, p95, empPct],
        type: "bar",
        marker: {
          color: ["#475569", "#64748b", "#ff9f1c", "#00e676"],
        },
        text: [p5, p50, p95, empPct].map((v) => `${v.toFixed(2)}%`),
        textposition: "outside",
      },
    ];

    const layout = {
      ...BASE_LAYOUT,
      hovermode: "closest",
      yaxis: {
        ...BASE_LAYOUT.yaxis,
        title: "Annualized CAGR (%)",
        ticksuffix: "%",
        range: [0, Math.max(empPct, p95) * 1.25],
      },
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
