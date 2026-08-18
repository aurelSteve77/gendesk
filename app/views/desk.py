"""The desk: generate a page and inspect it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.loaders import CHECKPOINT_LABELS, get_config, get_model, get_store, get_vocab
from app.theme import Palette, archetype_color, stat_tiles
from gendesk.corpus.rows import ARCHETYPES
from gendesk.data.universe import FUND_SECTOR
from gendesk.evaluation.diversity import page_diversity
from gendesk.evaluation.strategies import GenDeskStrategy
from gendesk.features.regimes import BUCKET_LABELS, REGIME_AXES, describe_regime
from gendesk.steering import apply_instruction, parse_instruction
from gendesk.tokenization.page import Page

REGIME_TONE = {
    "stressed": "critical",
    "vol_spiking": "warning",
    "downtrend": "warning",
    "narrow": "warning",
    "inverted": "warning",
    "high_corr": "warning",
}


def _regime_tiles(store, position: int) -> None:
    row = store.regimes.iloc[position]
    labels = describe_regime(row)
    stat_tiles(
        [
            (
                axis.replace("_", " "),
                labels[axis].replace("_", " "),
                f"tercile {int(row[axis]) + 1}/3",
            )
            for axis in REGIME_AXES
        ]
    )


def _row_chart(row_symbols: tuple[str, ...], store, position: int, palette: Palette) -> go.Figure:
    """Normalised 63-session price paths for one row's instruments."""
    start = max(0, position - 62)
    prices = store.returns.iloc[start : position + 1][list(row_symbols)]
    paths = (1.0 + prices).cumprod()
    paths = paths / paths.iloc[0] - 1.0

    figure = go.Figure()
    for i, symbol in enumerate(row_symbols):
        figure.add_trace(
            go.Scatter(
                x=paths.index,
                y=paths[symbol] * 100,
                name=symbol,
                mode="lines",
                line={"width": 2, "color": palette.color(i)},
                hovertemplate=f"<b>{symbol}</b> %{{y:.1f}}%<extra></extra>",
            )
        )
        figure.add_annotation(
            x=paths.index[-1],
            y=float(paths[symbol].iloc[-1] * 100),
            text=f" {symbol}",
            showarrow=False,
            xanchor="left",
            font={"size": 11, "color": palette.text_secondary},
        )

    figure.update_layout(
        height=210,
        margin={"l": 44, "r": 68, "t": 8, "b": 28},
        showlegend=False,
        yaxis={"title": None, "ticksuffix": "%"},
        xaxis={"title": None},
    )
    figure.add_hline(y=0, line={"color": palette.grid, "width": 1})
    return figure


def _row_table(
    row_symbols: tuple[str, ...], weights: pd.Series, store, position: int
) -> pd.DataFrame:
    by_symbol = store.catalog.by_symbol
    index = store.symbol_index
    mom = store.feature("mom_12_1")[position]
    rev = store.feature("rev_5d")[position]
    idio = store.feature("idio_vol")[position]

    return pd.DataFrame(
        [
            {
                "Instrument": symbol,
                "Sector": by_symbol[symbol].sector,
                "Weight": float(weights.get(symbol, 0.0)),
                "Vol (ann.)": float(store.vol.iloc[position].get(symbol, np.nan)),
                "Beta": float(store.beta.iloc[position].get(symbol, np.nan)),
                "12-1 mom (z)": float(mom[index[symbol]]),
                "5d reversal (z)": float(rev[index[symbol]]),
                "Idio vol (z)": float(idio[index[symbol]]),
            }
            for symbol in row_symbols
            if symbol in index
        ]
    )


def _sector_chart(weights: pd.Series, store, palette: Palette) -> go.Figure:
    by_symbol = store.catalog.by_symbol
    exposure: dict[str, float] = {}
    for symbol, weight in weights.items():
        sector = by_symbol[str(symbol)].sector
        exposure[sector] = exposure.get(sector, 0.0) + float(weight)

    series = pd.Series(exposure).sort_values()
    figure = go.Figure(
        go.Bar(
            x=series.to_numpy() * 100,
            y=series.index,
            orientation="h",
            marker={"color": palette.color(0), "cornerradius": 4},
            text=[f"{v:.1%}" for v in series],
            textposition="outside",
            textfont={"color": palette.text_secondary, "size": 11},
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    figure.update_layout(
        height=max(200, 30 * len(series)),
        margin={"l": 8, "r": 48, "t": 8, "b": 28},
        xaxis={"title": "weight", "ticksuffix": "%"},
        yaxis={"title": None},
        hovermode="closest",
        showlegend=False,
    )
    return figure


def render(palette: Palette) -> None:
    config = get_config()
    store = get_store()
    get_vocab()

    with st.sidebar:
        st.subheader("Desk controls")
        checkpoints = [name for name in CHECKPOINT_LABELS if _has(name)]
        if not checkpoints:
            st.error("No checkpoint found. Run `make train` first.")
            st.stop()
        checkpoint = st.selectbox(
            "Model stage",
            checkpoints,
            format_func=lambda n: CHECKPOINT_LABELS[n],
        )
        persona_name = st.selectbox(
            "Mandate", [p.name for p in config.personas], index=1, format_func=_pretty
        )
        max_pos = len(store.dates) - 1
        as_of = st.slider(
            "As of",
            min_value=int(store.date_position(config.backtest.valid_end)),
            max_value=int(max_pos),
            value=int(max_pos),
            format="%d",
            help="Position in the trading calendar. The right edge is the latest session.",
        )
        st.caption(f"**{store.dates[as_of].date()}**")

        hybrid = st.toggle(
            "Hybrid row decoding",
            value=True,
            help=(
                "Autoregress the first two slots of each row, then fill the rest from a "
                "single forward pass."
            ),
        )
        temperature = st.slider("Sampling temperature", 0.0, 1.2, 0.0, 0.1)
        instruction = st.text_input(
            "Steering instruction",
            placeholder="e.g. add duration hedges and cut energy exposure",
        )

    persona = next(p for p in config.personas if p.name == persona_name)
    active_config = config
    if instruction:
        parsed = parse_instruction(instruction, store.catalog.symbols, store.catalog.sectors)
        persona, active_config = apply_instruction(
            instruction, persona, config, parsed, catalog=store.catalog
        )
        if parsed.is_empty:
            st.warning(
                "No rule matched that instruction, so the page is unchanged. "
                "The parser is deliberately literal: a mandate change that cannot be "
                "stated as a constraint is not applied silently."
            )
        else:
            st.success("Steering applied - " + "; ".join(parsed.describe()))

    model, _ = get_model(checkpoint)
    head = "value" if checkpoint == "wbc" else "lm"

    strategy = GenDeskStrategy(
        model=model,
        vocab=get_vocab(),
        store=store,
        config=active_config,
        persona=persona,
        head=head,
        temperature=temperature,
        hybrid=hybrid,
    )
    weights = strategy(int(as_of))
    page: Page = next(iter(strategy.pages.values()))
    diagnostics = strategy.diagnostics[-1]

    st.markdown(f"### {_pretty(persona.name)} - {store.dates[as_of].date()}")
    st.markdown(
        f'<span class="gd-note">{persona.risk_budget} risk budget, '
        f"{persona.horizon_days}-session horizon, generated by "
        f"{CHECKPOINT_LABELS[checkpoint].lower()}</span>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Market regime")
    _regime_tiles(store, int(as_of))

    metrics = page_diversity(page, store, int(as_of), persona)
    stat_tiles(
        [
            ("Instruments", f"{metrics.n_names}", f"{metrics.n_sectors} sectors"),
            ("Effective bets", f"{metrics.effective_bets:.1f}", "1 / sum of squared weights"),
            ("Diversification", f"{metrics.diversification_ratio:.2f}", "weighted vol / book vol"),
            ("Mean pair corr.", f"{metrics.mean_correlation:.2f}", "trailing 126 sessions"),
            ("Sequential calls", f"{diagnostics['model_calls']}", "model invocations"),
            ("Latency", f"{diagnostics['latency_ms']:.0f} ms", "wall clock"),
        ]
    )

    st.divider()

    for row_index, row in enumerate(page.rows):
        archetype = ARCHETYPES.get(row.archetype)
        weight = sum(float(weights.get(s, 0.0)) for s in row.symbols)
        color = archetype_color(row.archetype, palette)
        st.markdown(
            f'<div class="gd-row-head">'
            f'<span class="gd-swatch" style="background:{color}"></span>'
            f'<span class="gd-row-title">{archetype.title if archetype else row.archetype}</span>'
            f'<span class="gd-row-weight">{weight:.1%} of book</span></div>'
            f'<div class="gd-row-thesis">{archetype.thesis if archetype else ""}</div>',
            unsafe_allow_html=True,
        )
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(
                _row_chart(row.symbols, store, int(as_of), palette),
                width="stretch",
                # A mandate with few allowed rows can repeat an archetype, so the key is
                # the slot, not the name.
                key=f"row_chart_{row_index}",
            )
        with right:
            table = _row_table(row.symbols, weights, store, int(as_of))
            st.dataframe(
                table,
                hide_index=True,
                width="stretch",
                column_config={
                    "Weight": st.column_config.NumberColumn(format="%.2f%%"),
                    "Vol (ann.)": st.column_config.NumberColumn(format="%.1f%%"),
                    "Beta": st.column_config.NumberColumn(format="%.2f"),
                    "12-1 mom (z)": st.column_config.NumberColumn(format="%.2f"),
                    "5d reversal (z)": st.column_config.NumberColumn(format="%.2f"),
                    "Idio vol (z)": st.column_config.NumberColumn(format="%.2f"),
                },
            )

    st.divider()
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Sector exposure")
        st.plotly_chart(_sector_chart(weights, store, palette), width="stretch")
        by_symbol = store.catalog.by_symbol
        equity_names = {s: w for s, w in weights.items() if by_symbol[str(s)].sector != FUND_SECTOR}
        st.caption(
            f"{len(equity_names)} single names, "
            f"{len(weights) - len(equity_names)} funds. Cap: "
            f"{active_config.decode.max_names_per_sector} names per sector, enforced as a "
            "token mask during generation rather than as a post-hoc filter."
        )
    with right:
        st.markdown("#### Constraint audit")
        report = {k: v for k, v in diagnostics.items() if k not in ("date",)}
        st.dataframe(
            pd.DataFrame([{"rule": k.replace("_", " "), "count": v} for k, v in report.items()]),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Counts are candidate tokens the decoder masked out. A page is compliant "
            "by construction: an illegal instrument is never sampled, so nothing has to "
            "be rejected and retried."
        )


def _pretty(name: str) -> str:
    return name.replace("_", " ").title()


def _has(name: str) -> bool:
    from gendesk.training.checkpoint import checkpoint_exists

    return checkpoint_exists(name)


__all__ = ["BUCKET_LABELS", "REGIME_TONE", "render"]
