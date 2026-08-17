"""Checkpointing and run logging.

Every stage writes into ``artifacts/runs/<run>/<stage>/`` with the config that
produced it and a JSONL metric stream. A checkpoint is only loadable against a
matching vocabulary fingerprint, which stops a stale catalog from silently
reinterpreting entity ids.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from gendesk.config import Config
from gendesk.model.gendesk import GenDeskModel
from gendesk.tokenization.vocab import Vocab
from gendesk.utils.logging import get_logger
from gendesk.utils.paths import CHECKPOINT_DIR, RUN_DIR

log = get_logger(__name__)


@dataclass
class RunLogger:
    """JSONL metric stream plus a config stamp."""

    run_name: str
    stage: str
    config: Config
    directory: Path = field(init=False)

    def __post_init__(self) -> None:
        self.directory = RUN_DIR / self.run_name / self.stage
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "config.json").write_text(
            json.dumps(self.config.dump(), indent=2, default=str)
        )
        self._path = self.directory / "metrics.jsonl"
        self._path.write_text("")

    def log(self, **fields: Any) -> None:
        payload = {"ts": datetime.now(UTC).isoformat(), **fields}
        with self._path.open("a") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")

    def summary(self, payload: dict) -> None:
        (self.directory / "summary.json").write_text(json.dumps(payload, indent=2, default=str))

    def read_metrics(self) -> list[dict]:
        if not self._path.exists():
            return []
        return [json.loads(line) for line in self._path.read_text().splitlines() if line.strip()]


def save_checkpoint(
    model: GenDeskModel,
    name: str,
    config: Config,
    metrics: dict | None = None,
    directory: Path | None = None,
) -> Path:
    directory = directory or CHECKPOINT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.pt"
    payload = model.checkpoint()
    payload["metrics"] = metrics or {}
    payload["run_config"] = config.dump()
    payload["saved_at"] = datetime.now(UTC).isoformat()
    torch.save(payload, path)
    log.info("checkpoint_saved", name=name, path=str(path), **(metrics or {}))
    return path


def load_checkpoint(
    name: str, vocab: Vocab, directory: Path | None = None, device: str | torch.device = "cpu"
) -> tuple[GenDeskModel, dict]:
    directory = directory or CHECKPOINT_DIR
    path = directory / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"checkpoint {path} not found")
    payload = torch.load(path, map_location=device, weights_only=False)
    model = GenDeskModel.from_checkpoint(payload, vocab)
    model.to(device)
    return model, payload.get("metrics", {})


def checkpoint_exists(name: str, directory: Path | None = None) -> bool:
    return ((directory or CHECKPOINT_DIR) / f"{name}.pt").exists()
