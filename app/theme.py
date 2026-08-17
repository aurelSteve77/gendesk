"""Visual system for the GenDesk app.

One palette, one Plotly template, one set of mark rules, applied everywhere. The
categorical hues are assigned to *entities* (a strategy, a row archetype) in a fixed
order and never cycled, so filtering the chart never repaints the survivors.

The eight-slot categorical palette and its dark-mode steps are the validated
reference set (adjacent-pair CVD dE >= 8, normal-vision dE >= 15 on both surfaces).
"""

from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Categorical slots, light and dark steps of the same eight hues.
CATEGORICAL_LIGHT = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
CATEGORICAL_DARK = (
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
)

STATUS = {
    "good": "#1baf7a",
    "warning": "#eda100",
    "critical": "#e34948",
    "neutral": "#8a8a85",
}


@dataclass(frozen=True)
class Palette:
    mode: str
    surface: str
    surface_alt: str
    text_primary: str
    text_secondary: str
    text_muted: str
    grid: str
    series: tuple[str, ...]

    def color(self, slot: int) -> str:
        """Colour for a fixed slot index. Slots past the palette fold into muted."""
        return self.series[slot] if 0 <= slot < len(self.series) else self.text_muted


LIGHT = Palette(
    mode="light",
    surface="#fcfcfb",
    surface_alt="#f2f2ef",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    text_muted="#8a8a85",
    grid="#e3e3df",
    series=CATEGORICAL_LIGHT,
)

DARK = Palette(
    mode="dark",
    surface="#1a1a19",
    surface_alt="#242422",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    text_muted="#8f8e86",
    grid="#33332f",
    series=CATEGORICAL_DARK,
)


def active_palette() -> Palette:
    """Palette matching the viewer's Streamlit theme."""
    try:
        base = st.get_option("theme.base") or "light"
    except Exception:  # pragma: no cover - Streamlit not initialised
        base = "light"
    return DARK if str(base).lower() == "dark" else LIGHT


# Fixed entity -> slot assignments. Colour follows the entity, never its rank.
STRATEGY_SLOTS: dict[str, int] = {
    "gendesk_rl": 0,
    "gendesk_wbc": 6,
    "gendesk_pretrain": 4,
    "pipeline_multistage": 1,
    "teacher_book": 3,
    "benchmark_spy": 2,
    "equal_weight": 5,
    "momentum_12_1": 7,
    "low_volatility": 5,
    "risk_parity": 3,
}

ARCHETYPE_SLOTS: dict[str, int] = {
    "MOMENTUM_LEADERS": 0,
    "TREND_BREAKOUT": 1,
    "QUALITY_BALLAST": 2,
    "MEAN_REVERSION": 3,
    "DISPERSION_HARVEST": 4,
    "HIGH_BETA_RISK_ON": 5,
    "CROWDING_UNWIND": 6,
    "MACRO_HEDGE": 7,
}


def strategy_color(name: str, palette: Palette | None = None) -> str:
    palette = palette or active_palette()
    return palette.color(STRATEGY_SLOTS.get(name, 8))


def archetype_color(name: str, palette: Palette | None = None) -> str:
    palette = palette or active_palette()
    return palette.color(ARCHETYPE_SLOTS.get(name, 8))


def install_template(palette: Palette | None = None) -> str:
    """Register and activate the Plotly template. Returns its name."""
    palette = palette or active_palette()
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor=palette.surface,
        plot_bgcolor=palette.surface,
        font={
            "family": "Inter, -apple-system, Segoe UI, sans-serif",
            "size": 13,
            "color": palette.text_secondary,
        },
        title={"font": {"size": 15, "color": palette.text_primary}, "x": 0, "xanchor": "left"},
        colorway=list(palette.series),
        margin={"l": 56, "r": 24, "t": 48, "b": 44},
        hovermode="x unified",
        hoverlabel={"bgcolor": palette.surface_alt, "font": {"color": palette.text_primary}},
        xaxis={
            "gridcolor": palette.grid,
            "zerolinecolor": palette.grid,
            "linecolor": palette.grid,
            "ticks": "outside",
            "tickcolor": palette.grid,
            "showspikes": True,
            "spikethickness": 1,
            "spikedash": "dot",
            "spikecolor": palette.text_muted,
            "spikemode": "across",
        },
        yaxis={
            "gridcolor": palette.grid,
            "zerolinecolor": palette.grid,
            "linecolor": palette.grid,
            "ticks": "outside",
            "tickcolor": palette.grid,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": palette.text_secondary},
        },
    )
    pio.templates["gendesk"] = template
    pio.templates.default = "gendesk"
    return "gendesk"


def inject_css(palette: Palette | None = None) -> None:
    """Page chrome: stat tiles, page cards, token chips."""
    p = palette or active_palette()
    st.markdown(
        f"""
        <style>
        .gd-tiles {{ display: flex; flex-wrap: wrap; gap: .5rem; margin: .25rem 0 1rem 0; }}
        .gd-tile {{
            flex: 1 1 128px; min-width: 128px; padding: .6rem .75rem;
            background: {p.surface_alt}; border-radius: 10px;
            border: 1px solid {p.grid};
        }}
        .gd-tile .k {{ font-size: .72rem; letter-spacing: .04em; text-transform: uppercase;
                       color: {p.text_muted}; }}
        .gd-tile .v {{ font-size: 1.32rem; font-weight: 600; color: {p.text_primary};
                       line-height: 1.5; font-variant-numeric: tabular-nums; }}
        .gd-tile .s {{ font-size: .76rem; color: {p.text_secondary}; }}

        .gd-row-head {{ display:flex; align-items:baseline; gap:.6rem; margin:.2rem 0 .1rem 0; }}
        .gd-row-title {{ font-size: 1.02rem; font-weight: 650; color: {p.text_primary}; }}
        .gd-row-weight {{ font-size: .82rem; color: {p.text_secondary};
                          font-variant-numeric: tabular-nums; }}
        .gd-row-thesis {{ font-size: .84rem; color: {p.text_secondary}; margin-bottom:.4rem; }}
        .gd-swatch {{ display:inline-block; width:10px; height:10px; border-radius:3px; }}

        .gd-chips {{ display:flex; flex-wrap:wrap; gap:.28rem; line-height:1.9; }}
        .gd-chip {{
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .74rem;
            padding: .12rem .42rem; border-radius: 6px; border: 1px solid {p.grid};
            color: {p.text_primary}; background: {p.surface_alt}; white-space: nowrap;
        }}
        .gd-note {{ font-size:.82rem; color:{p.text_muted}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def stat_tiles(items: list[tuple[str, str, str]]) -> None:
    """Render a row of stat tiles: ``(label, value, sublabel)``.

    A tile, not a chart: a handful of single numbers has no shape worth plotting.
    """
    html = ['<div class="gd-tiles">']
    for label, value, sub in items:
        html.append(
            f'<div class="gd-tile"><div class="k">{label}</div>'
            f'<div class="v">{value}</div><div class="s">{sub}</div></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def initialise(palette: Palette | None = None) -> Palette:
    """Install the Plotly template and page CSS. Call once per rerun."""
    palette = palette or active_palette()
    install_template(palette)
    inject_css(palette)
    return palette


def apply_line_style(figure: go.Figure, width: int = 2) -> go.Figure:
    """Thin marks, no point-by-point labels."""
    figure.update_traces(selector={"type": "scatter"}, line={"width": width})
    return figure
