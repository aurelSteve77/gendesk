"""Reinforcement learning and the diversification that shows up uninvited."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.loaders import get_config, load_rl_trace, missing_artifact
from app.theme import Palette, stat_tiles

DIVERSITY_SERIES = {
    "div_mean_correlation": ("Mean pairwise correlation", 0, ""),
    "div_diversification_ratio": ("Diversification ratio", 2, ""),
    "div_n_sectors": ("Distinct sectors per page", 3, ""),
    "div_effective_bets": ("Effective number of bets", 4, ""),
}


def _smooth(series: pd.Series, window: int = 15) -> pd.Series:
    return series.rolling(window, min_periods=max(2, window // 3)).mean()


def _line(x, y, name: str, color: str, dash: str | None = None, width: int = 2) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=y,
        name=name,
        mode="lines",
        line={"width": width, "color": color, **({"dash": dash} if dash else {})},
        hovertemplate=f"<b>{name}</b> step %{{x}}: %{{y:.3f}}<extra></extra>",
    )


def render(palette: Palette) -> None:
    config = get_config()
    trace = load_rl_trace(config.run_name)
    if trace is None:
        missing_artifact("The RL trace", "gendesk train rl")
        return

    reward = trace[["step", "mean_reward"]].dropna()
    stat_tiles(
        [
            ("RL steps", f"{int(trace['step'].max()) + 1}", "Dr. GRPO updates"),
            ("Group size", f"{config.training.rl.group_size}", "pages sampled per prompt"),
            (
                "Mean reward",
                f"{_smooth(reward['mean_reward']).iloc[-1]:.3f}",
                "15-step moving average",
            ),
            (
                "Reward change",
                f"{_smooth(reward['mean_reward']).iloc[-1] - _smooth(reward['mean_reward']).dropna().iloc[0]:+.3f}",
                "from start of training",
            ),
            ("KL to reference", f"{trace['kl'].iloc[-1]:.4f}", "k3 estimator"),
        ]
    )

    st.markdown("#### Page reward")
    figure = go.Figure()
    figure.add_trace(
        _line(
            reward["step"], reward["mean_reward"], "per step", palette.text_muted, palette, width=1
        )
    )
    figure.add_trace(
        _line(
            reward["step"],
            _smooth(reward["mean_reward"]),
            "15-step average",
            palette.color(0),
            palette,
        )
    )
    figure.update_layout(
        height=320,
        xaxis={"title": "optimisation step"},
        yaxis={"title": "risk-adjusted active reward"},
    )
    st.plotly_chart(figure, use_container_width=True)
    st.caption(
        "The reward is the page's benchmark-relative return over the mandate's horizon, "
        "scaled by the volatility budget, minus drawdown and turnover penalties. Nothing "
        "in it mentions diversification."
    )

    st.divider()
    st.markdown("#### Diversification, which nobody asked for")
    st.markdown(
        "Netflix reports that page diversity rises during RL despite not being optimised, "
        "which they read as evidence that the model is optimising the page as a whole. A "
        "portfolio has a sharper version of that test, because diversification has a "
        "canonical definition. These are the measurements."
    )

    available = [c for c in DIVERSITY_SERIES if c in trace.columns and trace[c].notna().any()]
    if not available:
        st.info("Diversity was not sampled during this run.")
        return

    columns = st.columns(2)
    for i, column in enumerate(available):
        label, slot, _ = DIVERSITY_SERIES[column]
        series = trace[["step", column]].dropna()
        figure = go.Figure()
        figure.add_trace(_line(series["step"], series[column], label, palette.color(slot)))
        # Least-squares trend: the claim is about direction, so show the direction.
        if len(series) >= 3:
            slope, intercept = np.polyfit(series["step"], series[column], 1)
            figure.add_trace(
                go.Scatter(
                    x=series["step"],
                    y=slope * series["step"] + intercept,
                    name="trend",
                    mode="lines",
                    line={"width": 1, "color": palette.text_muted, "dash": "dash"},
                    hovertemplate=f"trend: {slope:+.2e} per step<extra></extra>",
                )
            )
        figure.update_layout(
            title=label,
            height=280,
            xaxis={"title": "optimisation step"},
            yaxis={"title": None},
            showlegend=False,
        )
        columns[i % 2].plotly_chart(figure, use_container_width=True)

    rows = []
    for column in available:
        series = trace[["step", column]].dropna()
        if len(series) < 3:
            continue
        slope = float(np.polyfit(series["step"], series[column], 1)[0])
        rows.append(
            {
                "Measure": DIVERSITY_SERIES[column][0],
                "Start": float(series[column].iloc[0]),
                "End": float(series[column].iloc[-1]),
                "Change": float(series[column].iloc[-1] - series[column].iloc[0]),
                "Slope per step": slope,
            }
        )
    st.dataframe(
        pd.DataFrame(rows).style.format(
            {"Start": "{:.3f}", "End": "{:.3f}", "Change": "{:+.3f}", "Slope per step": "{:+.2e}"}
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Lower mean pairwise correlation and a higher diversification ratio both mean the "
        "book's risk decomposes into more independent pieces."
    )

    with st.expander("Policy diagnostics"):
        figure = go.Figure()
        figure.add_trace(_line(trace["step"], trace["kl"], "KL to reference", palette.color(5)))
        figure.update_layout(height=240, yaxis={"title": "KL"}, xaxis={"title": "step"})
        st.plotly_chart(figure, use_container_width=True)

        figure = go.Figure()
        figure.add_trace(_line(trace["step"], trace["entropy"], "policy entropy", palette.color(6)))
        figure.update_layout(height=240, yaxis={"title": "nats"}, xaxis={"title": "step"})
        st.plotly_chart(figure, use_container_width=True)
        st.caption(
            "A collapsing entropy with a rising reward is the failure mode to watch for: "
            "the policy has found one page it likes and stopped exploring."
        )
