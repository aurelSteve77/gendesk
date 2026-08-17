"""GenDesk - Streamlit front end.

Run with ``make app`` (or ``streamlit run app/main.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.theme import active_palette, initialise  # noqa: E402
from app.views import ablations, backtest, desk, emergence, internals  # noqa: E402

st.set_page_config(
    page_title="GenDesk",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

palette = active_palette()
initialise(palette)

st.sidebar.title("GenDesk")
st.sidebar.caption(
    "LLM-native generative construction of a research desk page, after Netflix's "
    "GenRec and GenPage."
)

TABS = {
    "The Desk": desk.render,
    "Out-of-sample": backtest.render,
    "Ablations": ablations.render,
    "RL & emergence": emergence.render,
    "Under the hood": internals.render,
}

tabs = st.tabs(list(TABS))
for tab, render in zip(tabs, TABS.values(), strict=True):
    with tab:
        render(palette)

st.sidebar.divider()
st.sidebar.caption(
    "Research prototype on public end-of-day data. Not investment advice, and not a "
    "production trading system."
)
