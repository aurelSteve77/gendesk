"""Content hashing for data manifests.

Every processed dataset is written alongside a manifest recording the git-style
content hash of its inputs, so a stale cache can never silently poison a backtest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def hash_bytes(payload: bytes) -> str:
    """Return the first 16 hex chars of the SHA-256 of ``payload``."""
    return hashlib.sha256(payload).hexdigest()[:16]


def hash_obj(obj: Any) -> str:
    """Hash any JSON-serialisable object with stable key ordering."""
    return hash_bytes(json.dumps(obj, sort_keys=True, default=str).encode())


def hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream-hash a file on disk."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()[:16]
