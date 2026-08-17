"""Under the hood: the token stream, the vocabulary, the training curves."""

from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.loaders import (
    CHECKPOINT_LABELS,
    get_config,
    get_model,
    get_store,
    get_vocab,
    load_report,
    load_training_trace,
    missing_artifact,
)
from app.theme import Palette, stat_tiles
from gendesk.evaluation.strategies import GenDeskStrategy
from gendesk.features.regimes import REGIME_AXES
from gendesk.tokenization.page import PageContext, PageSequence
from gendesk.tokenization.vocab import ROW_ARCHETYPES, SPECIALS
from gendesk.training.checkpoint import checkpoint_exists

TOKEN_KINDS = {
    "special": 7,
    "persona": 0,
    "risk": 0,
    "horizon": 0,
    "regime": 2,
    "row": 1,
    "entity": 5,
}


def _kind(token: str) -> str:
    if token in SPECIALS:
        return "special"
    for prefix in ("persona", "risk", "horizon", "regime", "row"):
        if token.startswith(f"<{prefix}:"):
            return prefix
    return "entity"


def _chips(tokens: list[str], palette: Palette) -> str:
    parts = ['<div class="gd-chips">']
    for token in tokens:
        kind = _kind(token)
        color = palette.color(TOKEN_KINDS.get(kind, 8))
        parts.append(
            f'<span class="gd-chip" style="border-color:{color};color:{color}" '
            f'title="{kind}">{html.escape(token)}</span>'
        )
    parts.append("</div>")
    return "".join(parts)


def render(palette: Palette) -> None:
    config = get_config()
    store = get_store()
    vocab = get_vocab()

    st.markdown(
        "A general-purpose tokenizer spends four subword pieces on `NVDA`. This one spends "
        "one token, and the same is true of a mandate, a regime bucket and a row archetype. "
        "That is the whole efficiency argument in GenPage, and it is why a page fits in "
        "roughly 50 tokens."
    )

    n_entities = vocab.n_instruments
    n_context = vocab.entity_offset - len(SPECIALS) - len(ROW_ARCHETYPES)
    stat_tiles(
        [
            ("Vocabulary", f"{vocab.size}", "total tokens"),
            ("Instruments", f"{n_entities}", "one token each"),
            ("Context tokens", f"{n_context}", "mandate + regime buckets"),
            ("Row archetypes", f"{len(ROW_ARCHETYPES)}", "page section types"),
            ("Fingerprint", vocab.fingerprint()[:8], "checkpoint compatibility key"),
        ]
    )

    st.divider()
    st.markdown("#### A page as the model sees it")

    checkpoints = [n for n in CHECKPOINT_LABELS if checkpoint_exists(n)]
    if not checkpoints:
        missing_artifact("A trained checkpoint", "gendesk train pretrain")
        return

    left, right = st.columns([2, 1])
    with left:
        persona_name = st.selectbox(
            "Mandate",
            [p.name for p in config.personas],
            index=1,
            format_func=lambda n: n.replace("_", " ").title(),
        )
    with right:
        checkpoint = st.selectbox("Stage", checkpoints, format_func=lambda n: CHECKPOINT_LABELS[n])

    persona = next(p for p in config.personas if p.name == persona_name)
    model, _ = get_model(checkpoint)
    position = len(store.dates) - 1

    strategy = GenDeskStrategy(
        model=model,
        vocab=vocab,
        store=store,
        config=config,
        persona=persona,
        head="value" if checkpoint == "wbc" else "lm",
        temperature=0.0,
    )
    strategy(position)
    page = next(iter(strategy.pages.values()))

    context = PageContext(
        persona=persona.name,
        risk_budget=persona.risk_budget,
        horizon_days=persona.horizon_days,
        regimes={axis: int(store.regimes[axis].iloc[position]) for axis in REGIME_AXES},
        history=(),
    )
    sequence = PageSequence(vocab)
    encoded = sequence.encode(page, context)
    tokens = [vocab.tokens[int(t)] for t in encoded.tokens]

    st.markdown("**Prompt** - context the page is conditioned on")
    st.markdown(_chips(tokens[: encoded.prompt_len], palette), unsafe_allow_html=True)
    st.markdown("**Generated page** - rows and their instruments, in layout order")
    st.markdown(_chips(tokens[encoded.prompt_len :], palette), unsafe_allow_html=True)
    st.caption(
        f"{len(tokens)} tokens total: {encoded.prompt_len} of prompt and "
        f"{len(tokens) - encoded.prompt_len} of page. Colour indicates token kind."
    )

    st.divider()
    st.markdown("#### Row archetypes")
    from gendesk.corpus.rows import ARCHETYPES

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Archetype": arch.title,
                    "Thesis": arch.thesis,
                    "On this page": "yes" if name in page.archetypes else "",
                }
                for name, arch in ARCHETYPES.items()
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.divider()
    st.markdown("#### Training curves")
    for stage in ("pretrain", "wbc"):
        trace = load_training_trace(config.run_name, stage)
        if trace is None or ("loss" not in trace.columns and "wbc_loss" not in trace.columns):
            continue
        column = "loss" if "loss" in trace.columns else "wbc_loss"
        steps = trace.dropna(subset=[column])
        if steps.empty:
            continue
        figure = go.Figure(
            go.Scatter(
                x=steps["step"],
                y=steps[column],
                mode="lines",
                name=stage,
                line={"width": 2, "color": palette.color(0 if stage == "pretrain" else 2)},
                hovertemplate="step %{x}: %{y:.4f}<extra></extra>",
            )
        )
        figure.update_layout(
            title=f"Stage {'1 pretraining' if stage == 'pretrain' else '2 post-training'}",
            height=260,
            xaxis={"title": "step"},
            yaxis={"title": "training loss"},
            showlegend=False,
        )
        st.plotly_chart(figure, use_container_width=True)

    latency = load_report("latency.json")
    if latency:
        st.divider()
        st.markdown("#### Serving latency")
        frame = pd.DataFrame(latency)
        modes = frame[frame["mode"] != "reduction"]
        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=modes["mode"],
                y=modes["median_ms"],
                marker={"color": palette.color(0), "cornerradius": 4},
                text=[f"{v:.0f} ms" for v in modes["median_ms"]],
                textposition="outside",
                textfont={"color": palette.text_secondary},
                hovertemplate="%{x}: %{y:.1f} ms<extra></extra>",
            )
        )
        figure.update_layout(
            height=280,
            yaxis={"title": "median latency (ms)"},
            xaxis={"title": None},
            showlegend=False,
            hovermode="closest",
            bargap=0.45,
        )
        st.plotly_chart(figure, use_container_width=True)

        reduction = frame[frame["mode"] == "reduction"]
        if not reduction.empty:
            row = reduction.iloc[0]
            stat_tiles(
                [
                    ("Latency reduction", f"{row['median_ms']:.0%}", "median wall clock"),
                    ("Sequential calls", f"{row['sequential_model_calls']:.0%}", "fewer per page"),
                    (
                        "Autoregressive",
                        f"{modes.iloc[0]['sequential_model_calls']:.0f}",
                        "model invocations",
                    ),
                    (
                        "Hybrid",
                        f"{modes.iloc[1]['sequential_model_calls']:.0f}",
                        "model invocations",
                    ),
                ]
            )
        st.caption(
            "Wall clock is hardware-specific; the count of sequential model invocations is "
            "not, and it is the quantity hybrid decoding actually removes."
        )
