"""Repository-relative path resolution.

Every artifact location is derived from a single project root so that the CLI, the
test-suite and the Streamlit app agree on where things live regardless of the
working directory they are invoked from.
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the repository root.

    Resolution order: the ``GENDESK_ROOT`` environment variable, then the first
    ancestor of this file that contains a ``pyproject.toml``.
    """
    env = os.environ.get("GENDESK_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[3]


ROOT = project_root()

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_DIR = DATA_DIR / "manifests"

ARTIFACT_DIR = ROOT / "artifacts"
CHECKPOINT_DIR = ARTIFACT_DIR / "checkpoints"
RUN_DIR = ARTIFACT_DIR / "runs"
REPORT_DIR = ARTIFACT_DIR / "reports"

CONFIG_DIR = ROOT / "configs"


def ensure_dirs() -> None:
    """Create every directory the pipeline writes into."""
    for path in (
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        MANIFEST_DIR,
        CHECKPOINT_DIR,
        RUN_DIR,
        REPORT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
