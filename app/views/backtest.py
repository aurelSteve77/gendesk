"""Out-of-sample results."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.loaders import load_report, load_returns, missing_artifact
from app.theme import Palette, stat_tiles, strategy_color

HEADLINE = ("gendesk_rl", "gendesk_wbc", "pipeline_multistage", "teacher_book", "benchmark_spy")

DISPLAY = {
    "gendesk_rl": "GenDesk (RL)",
    "gendesk_wbc": "GenDesk (WBC)",
    "gendesk_pretrain": "GenDesk (pretrained)",
    "pipeline_multistage": "Multi-stage pipeline",
    "teacher_book": "Teacher screen",
    "benchmark_spy": "S&P 500 ETF",
    "equal_weight": "Equal weight",
    "momentum_12_1": "12-1 momentum",
    "low_volatility": "Low volatility",
    "risk_parity": "Risk parity",
}


def _label(name: str) -> str:
    return DISPLAY.get(name, name.replace("_", " "))


def _stat(summary: pd.DataFrame, strategy: str, column: str) -> float:
    """Read one statistic, tolerating a report written by an older pipeline version."""
    if column not in summary.columns or strategy not in summary.index:
        return float("nan")
    return float(summary.loc[strategy, column])


def _fmt(value: float, spec: str) -> str:
    return "n/a" if value != value else format(value, spec)


def _equity_chart(returns: pd.DataFrame, selected: list[str], palette: Palette) -> go.Figure:
    figure = go.Figure()
    for name in selected:
        equity = (1.0 + returns[name].fillna(0.0)).cumprod()
        figure.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity,
                name=_label(name),
                mode="lines",
                line={"width": 2, "color": strategy_color(name, palette)},
                hovertemplate=f"<b>{_label(name)}</b> %{{y:.2f}}x<extra></extra>",
            )
        )
    figure.update_layout(
        height=420,
        yaxis={"title": "growth of 1", "type": "log", "tickformat": ".2f"},
        xaxis={"title": None},
    )
    return figure


def _drawdown_chart(returns: pd.DataFrame, selected: list[str], palette: Palette) -> go.Figure:
    figure = go.Figure()
    for name in selected:
        equity = (1.0 + returns[name].fillna(0.0)).cumprod()
        drawdown = (equity / equity.cummax() - 1.0) * 100
        figure.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown,
                name=_label(name),
                mode="lines",
                line={"width": 2, "color": strategy_color(name, palette)},
                hovertemplate=f"<b>{_label(name)}</b> %{{y:.1f}}%<extra></extra>",
            )
        )
    figure.update_layout(
        height=280, yaxis={"title": "drawdown", "ticksuffix": "%"}, xaxis={"title": None}
    )
    return figure


def _sharpe_chart(inference: dict, selected: list[str], palette: Palette) -> go.Figure:
    rows = [(n, inference[n]["sharpe_test"]) for n in selected if n in inference]
    rows.sort(key=lambda kv: kv[1]["estimate"])

    figure = go.Figure()
    for name, test in rows:
        color = strategy_color(name, palette)
        figure.add_trace(
            go.Scatter(
                x=[test["ci_low"], test["ci_high"]],
                y=[_label(name), _label(name)],
                mode="lines",
                line={"width": 2, "color": color},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[test["estimate"]],
                y=[_label(name)],
                mode="markers",
                marker={"size": 11, "color": color, "line": {"width": 2, "color": palette.surface}},
                name=_label(name),
                showlegend=False,
                hovertemplate=(
                    f"<b>{_label(name)}</b><br>Sharpe %{{x:.2f}}<br>"
                    f"95% CI [{test['ci_low']:.2f}, {test['ci_high']:.2f}]<extra></extra>"
                ),
            )
        )
    figure.add_vline(x=0, line={"color": palette.grid, "width": 1, "dash": "dot"})
    figure.update_layout(
        height=60 + 34 * max(len(rows), 1),
        xaxis={"title": "annualised Sharpe (95% block-bootstrap interval)"},
        yaxis={"title": None},
        hovermode="closest",
        showlegend=False,
    )
    return figure


def render(palette: Palette) -> None:
    window = st.radio(
        "Window",
        ["test", "valid"],
        horizontal=True,
        format_func=lambda w: "Out-of-sample" if w == "test" else "Validation",
    )
    report = load_report(f"backtest_{window}.json")
    returns = load_returns(window)
    if report is None or returns is None:
        missing_artifact("The backtest", f"gendesk eval backtest --window {window}")
        return

    summary = pd.DataFrame(report["summary"]).set_index("strategy")
    st.markdown(
        f"Walk-forward from **{report['start']}** to **{report['end']}**. Returns are the "
        "equal-weighted average across the six mandates, net of "
        f"{report['config']['backtest']['cost_bps']:.0f} bp one-way costs, rebalanced every "
        f"{report['config']['backtest']['rebalance_days']} sessions."
    )

    headline = "gendesk_rl" if "gendesk_rl" in summary.index else summary.index[0]
    reference = (
        "pipeline_multistage" if "pipeline_multistage" in summary.index else summary.index[-1]
    )
    stat_tiles(
        [
            ("Sharpe", _fmt(_stat(summary, headline, "sharpe"), ".2f"), _label(headline)),
            (
                "vs pipeline",
                _fmt(
                    _stat(summary, headline, "sharpe") - _stat(summary, reference, "sharpe"),
                    "+.2f",
                ),
                "Sharpe difference",
            ),
            ("CAGR", _fmt(_stat(summary, headline, "cagr"), ".1%"), _label(headline)),
            (
                "Max drawdown",
                _fmt(_stat(summary, headline, "max_drawdown"), ".1%"),
                "peak to trough",
            ),
            (
                "Turnover",
                _fmt(_stat(summary, headline, "annual_turnover"), ".1f") + "x",
                "annualised, one-way",
            ),
            (
                "PBO",
                _fmt(report.get("pbo", {}).get("pbo", float("nan")), ".0%"),
                "prob. of backtest overfitting",
            ),
        ]
    )

    default = [n for n in HEADLINE if n in returns.columns]
    selected = st.multiselect(
        "Strategies", list(returns.columns), default=default, format_func=_label
    )
    if len(selected) > 8:
        st.warning("Showing the first eight strategies; colours are never cycled.")
        selected = selected[:8]
    if not selected:
        st.stop()

    st.plotly_chart(_equity_chart(returns, selected, palette), width="stretch")
    st.plotly_chart(_drawdown_chart(returns, selected, palette), width="stretch")

    st.markdown("#### Sharpe ratios with bootstrap intervals")
    st.plotly_chart(_sharpe_chart(report["inference"], selected, palette), width="stretch")
    st.caption(
        "Stationary block bootstrap, 21-session blocks, 2000 resamples. An interval that "
        "straddles zero is a strategy that has not demonstrated an edge over this window."
    )

    comparisons = report.get("comparisons", {})
    if comparisons:
        st.markdown("#### Paired comparisons")
        frame = pd.DataFrame(
            [
                {
                    "Comparison": key.replace("_vs_", " vs ").replace("_", " "),
                    "Sharpe difference": payload["estimate"],
                    "95% CI": f"[{payload['ci_low']:.2f}, {payload['ci_high']:.2f}]",
                    "p-value": payload["p_value"],
                }
                for key, payload in comparisons.items()
            ]
        )
        st.dataframe(frame, hide_index=True, width="stretch")
        st.caption(
            "Both legs are resampled with the same block indices, so the difference test "
            "keeps their contemporaneous correlation instead of throwing it away."
        )

    st.markdown("#### Full table")
    table = summary.copy()
    table.index = [_label(i) for i in table.index]
    st.dataframe(
        table[
            [
                c
                for c in [
                    "cagr",
                    "vol",
                    "sharpe",
                    "sortino",
                    "max_drawdown",
                    "calmar",
                    "hit_rate",
                    "annual_turnover",
                    "cost_drag",
                ]
                if c in table.columns
            ]
        ].style.format(
            {
                "cagr": "{:.2%}",
                "vol": "{:.2%}",
                "sharpe": "{:.2f}",
                "sortino": "{:.2f}",
                "max_drawdown": "{:.2%}",
                "calmar": "{:.2f}",
                "hit_rate": "{:.1%}",
                "annual_turnover": "{:.2f}",
                "cost_drag": "{:.2%}",
            }
        ),
        width="stretch",
    )

    per_persona = report.get("per_persona")
    if per_persona:
        st.markdown("#### By mandate")
        frame = pd.DataFrame(per_persona)
        frame = frame[frame["persona"] != "-"]
        pivot = frame.pivot_table(index="persona", columns="strategy", values="sharpe")
        st.dataframe(
            pivot.rename(columns=_label)
            .style.format("{:.2f}")
            .background_gradient(cmap="Blues", axis=None),
            width="stretch",
        )
        st.caption("Annualised Sharpe by mandate. Every mandate faces the same market.")

    with st.expander("Return correlations"):
        corr = returns[selected].corr()
        corr.index = [_label(i) for i in corr.index]
        corr.columns = [_label(c) for c in corr.columns]
        st.dataframe(corr.style.format("{:.2f}"), width="stretch")

    with st.expander("Deflated Sharpe ratio"):
        rows = [
            {
                "Strategy": _label(name),
                "Sharpe": payload["deflated"]["sharpe"],
                "Null expected max": payload["deflated"]["sharpe0"],
                "P(true Sharpe > 0)": payload["deflated"]["dsr"],
            }
            for name, payload in report["inference"].items()
            if name in selected
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            f"Deflated against {report['config']['backtest']['n_trials']} configurations "
            "tried, and corrected for the skew and kurtosis of the realised returns."
        )
