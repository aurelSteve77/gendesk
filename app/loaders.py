"""Cached resource loading for the app.

Streamlit reruns the whole script on every interaction, so anything expensive --
the feature store, the checkpoints, the report artifacts -- is memoised here and
nowhere else.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from gendesk.config import Config, load_config
from gendesk.features.store import FeatureStore, load_features
from gendesk.tokenization.vocab import Vocab, build_vocab
from gendesk.training.checkpoint import checkpoint_exists, load_checkpoint
from gendesk.training.schedule import resolve_device
from gendesk.utils.paths import REPORT_DIR, RUN_DIR

CHECKPOINT_LABELS = {
    "rl": "Stage 3 - page-level RL (Dr. GRPO)",
    "wbc": "Stage 2 - weighted binary classification",
    "pretrain": "Stage 1 - pretrained on positive pages",
}


@st.cache_resource(show_spinner="Loading configuration...")
def get_config() -> Config:
    return load_config()


@st.cache_resource(show_spinner="Loading market data and features...")
def get_store() -> FeatureStore:
    return load_features()


@st.cache_resource
def get_vocab() -> Vocab:
    config, store = get_config(), get_store()
    return build_vocab(store.catalog, tuple(p.name for p in config.personas))


@st.cache_resource(show_spinner="Loading model...")
def get_model(checkpoint: str):
    vocab = get_vocab()
    device = resolve_device(get_config().training.device)
    model, metrics = load_checkpoint(checkpoint, vocab, device=device)
    model.eval()
    return model, metrics


def available_checkpoints() -> list[str]:
    return [name for name in CHECKPOINT_LABELS if checkpoint_exists(name)]


@st.cache_data(show_spinner=False)
def load_report(name: str) -> dict | list | None:
    path = REPORT_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def load_returns(window: str = "test") -> pd.DataFrame | None:
    path = REPORT_DIR / f"returns_{window}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_ablations() -> pd.DataFrame | None:
    path = REPORT_DIR / "ablations.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_rl_trace(run_name: str) -> pd.DataFrame | None:
    path: Path = RUN_DIR / run_name / "rl" / "metrics.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    frame = pd.DataFrame([r for r in rows if r.get("stage") == "rl"])
    return frame if not frame.empty else None


@st.cache_data(show_spinner=False)
def load_training_trace(run_name: str, stage: str) -> pd.DataFrame | None:
    path: Path = RUN_DIR / run_name / stage / "metrics.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    return frame if not frame.empty else None


def missing_artifact(label: str, command: str) -> None:
    """Uniform empty state pointing at the command that produces the artifact."""
    st.info(f"{label} has not been generated yet. Run `{command}` to produce it.")
