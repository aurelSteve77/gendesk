"""Ablations: where the information actually lives."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.loaders import load_ablations, missing_artifact
from app.theme import Palette, stat_tiles
from gendesk.evaluation.ablations import summarise_headline

CELL_LABELS = {
    "ctx0_rows_only": "Rows only",
    "ctx1_mandate": "+ mandate",
    "ctx2_mandate_regime": "+ regime",
    "ctx3_full": "+ history (full)",
    "cap0_tiny": "Tiny",
    "cap1_small": "Small",
    "cap2_base": "Base",
    "cap3_large": "Large",
    "design_no_semantic_fusion": "No feature fusion",
    "design_no_row_tokens": "No row tokens",
    "design_no_outcome_filter": "No outcome filter",
}


def _ladder_chart(
    frame: pd.DataFrame, value: str, title: str, suffix: str, palette: Palette, slot: int
) -> go.Figure:
    labels = [CELL_LABELS.get(c, c) for c in frame["cell"]]
    values = frame[value].to_numpy() * 100
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker={"color": palette.color(slot), "cornerradius": 4},
            text=[f"{v:+.1f}{suffix}" for v in values],
            textposition="outside",
            textfont={"color": palette.text_secondary, "size": 12},
            hovertemplate="%{x}: %{y:.2f}" + suffix + "<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        height=320,
        yaxis={"title": f"vs first rung ({suffix})", "ticksuffix": suffix},
        xaxis={"title": None},
        hovermode="closest",
        showlegend=False,
        bargap=0.35,
    )
    figure.add_hline(y=0, line={"color": palette.grid, "width": 1})
    return figure


def render(palette: Palette) -> None:
    frame = load_ablations()
    if frame is None:
        missing_artifact("The ablation grid", "gendesk eval ablations")
        return

    headline = summarise_headline(frame)
    if headline:
        st.markdown(
            "GenPage reports that enriching the prompt beats scaling the model. That is a "
            "testable claim, and this is the test: identical corpus, identical validation "
            "split, identical epochs, one variable at a time."
        )
        stat_tiles(
            [
                (
                    "Context enrichment",
                    f"{headline['context_loss_reduction']:+.1%}",
                    "validation loss reduction",
                ),
                (
                    "Capacity scaling",
                    f"{headline['capacity_loss_reduction']:+.1%}",
                    f"at {headline['capacity_multiple']:.0f}x parameters",
                ),
                ("Context MRR", f"{headline['context_mrr_gain']:+.1%}", "slot retrieval"),
                ("Capacity MRR", f"{headline['capacity_mrr_gain']:+.1%}", "slot retrieval"),
            ]
        )

    left, right = st.columns(2)
    context = frame[frame.family == "context"]
    capacity = frame[frame.family == "capacity"]
    with left:
        st.plotly_chart(
            _ladder_chart(
                context, "loss_vs_family_base", "Prompt content (fixed capacity)", "%", palette, 0
            ),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            _ladder_chart(
                capacity, "loss_vs_family_base", "Backbone capacity (fixed prompt)", "%", palette, 1
            ),
            width="stretch",
        )

    st.caption(
        "Bars are validation-loss reduction relative to the first rung of each ladder. "
        "The capacity ladder spans "
        f"{capacity['params_m'].min():.1f}M to {capacity['params_m'].max():.1f}M parameters."
    )

    design = frame[frame.family == "design"]
    if not design.empty:
        st.markdown("#### Design choices")
        base = float(frame[frame.cell == "ctx3_full"]["loss"].iloc[0])
        rows = [
            {
                "Ablation": CELL_LABELS.get(row.cell, row.cell),
                "Validation loss": row.loss,
                "vs full model": row.loss - base,
                "MRR": row.mrr,
                "hit@5": row["hit@5"],
            }
            for row in design.itertuples()
        ]
        st.dataframe(
            pd.DataFrame(rows).style.format(
                {
                    "Validation loss": "{:.4f}",
                    "vs full model": "{:+.4f}",
                    "MRR": "{:.4f}",
                    "hit@5": "{:.3f}",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "A positive 'vs full model' means removing the component made the model worse. "
            "'No outcome filter' pretrains on every teacher candidate instead of only the "
            "ones that earned their keep."
        )

    with st.expander("Full grid"):
        display = frame.copy()
        display["cell"] = display["cell"].map(lambda c: CELL_LABELS.get(c, c))
        st.dataframe(display, hide_index=True, width="stretch")
