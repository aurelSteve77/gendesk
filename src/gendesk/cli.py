"""Command line interface.

Every stage is reachable as ``gendesk <group> <command>``. The full reproduction is
``make pipeline``, which is just these commands in order.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from gendesk.config import Config, load_config
from gendesk.utils.logging import configure_logging, get_logger
from gendesk.utils.paths import REPORT_DIR, ensure_dirs

app = typer.Typer(add_completion=False, help="GenDesk: generative research-desk construction.")
data_app = typer.Typer(help="Market data and features.")
corpus_app = typer.Typer(help="Page corpus.")
train_app = typer.Typer(help="Model training stages.")
eval_app = typer.Typer(help="Backtests, ablations and reports.")

app.add_typer(data_app, name="data")
app.add_typer(corpus_app, name="corpus")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")

console = Console()
log = get_logger(__name__)

ConfigOption = typer.Option(None, "--config", "-c", help="Path to a YAML config.")
ForceOption = typer.Option(False, "--force", "-f", help="Ignore cached artifacts.")


def _config(path: Path | None) -> Config:
    configure_logging()
    ensure_dirs()
    return load_config(path)


def _bootstrap(path: Path | None):
    """Load config, feature store, corpus and vocabulary in one go."""
    from gendesk.corpus.build import load_corpus
    from gendesk.features.store import load_features
    from gendesk.tokenization.vocab import build_vocab

    config = _config(path)
    store = load_features()
    corpus = load_corpus()
    vocab = build_vocab(store.catalog, tuple(p.name for p in config.personas))
    return config, store, corpus, vocab


# ------------------------------------------------------------------ data


@data_app.command("build")
def data_build(config: Path | None = ConfigOption, force: bool = ForceOption) -> None:
    """Download and cache the aligned price panel."""
    from gendesk.data.panel import build_panel

    cfg = _config(config)
    panel = build_panel(cfg, force=force)
    console.print(
        f"[green]panel[/green] {len(panel.symbols)} instruments x {len(panel.calendar)} sessions "
        f"({panel.calendar.min().date()} -> {panel.calendar.max().date()})"
    )


@data_app.command("features")
def data_features(config: Path | None = ConfigOption, force: bool = ForceOption) -> None:
    """Build point-in-time features and regime buckets."""
    from gendesk.features.store import build_features

    cfg = _config(config)
    store = build_features(cfg, force=force)
    console.print(f"[green]features[/green] tensor {store.values.shape}")


# ---------------------------------------------------------------- corpus


@corpus_app.command("build")
def corpus_build(config: Path | None = ConfigOption, force: bool = ForceOption) -> None:
    """Generate, score and filter the page corpus."""
    from gendesk.corpus.build import build_corpus
    from gendesk.features.store import load_features

    cfg = _config(config)
    corpus = build_corpus(cfg, load_features(), force=force)
    console.print(f"[green]corpus[/green] {len(corpus)} examples")
    console.print(json.dumps(corpus.meta.get("counts", {}), indent=2))


# ---------------------------------------------------------------- train


@train_app.command("pretrain")
def train_pretrain(config: Path | None = ConfigOption) -> None:
    """Stage 1: next-token pretraining on outcome-filtered pages."""
    from gendesk.training.pretrain import pretrain

    cfg, store, corpus, vocab = _bootstrap(config)
    _, metrics = pretrain(cfg, store, corpus, vocab)
    console.print(metrics.as_dict())


@train_app.command("wbc")
def train_wbc_cmd(config: Path | None = ConfigOption) -> None:
    """Stage 2: weighted binary classification post-training."""
    from gendesk.training.checkpoint import load_checkpoint
    from gendesk.training.schedule import resolve_device
    from gendesk.training.wbc import train_wbc

    cfg, store, corpus, vocab = _bootstrap(config)
    device = resolve_device(cfg.training.device)
    model, _ = load_checkpoint("pretrain", vocab, device=device)
    _, metrics = train_wbc(cfg, store, corpus, vocab, model)
    console.print(metrics.as_dict())


@train_app.command("rl")
def train_rl_cmd(config: Path | None = ConfigOption) -> None:
    """Stage 3: Dr. GRPO page-level reinforcement learning."""
    from gendesk.training.checkpoint import load_checkpoint
    from gendesk.training.rl import train_rl
    from gendesk.training.schedule import resolve_device

    cfg, store, corpus, vocab = _bootstrap(config)
    device = resolve_device(cfg.training.device)
    model, _ = load_checkpoint("wbc", vocab, device=device)
    _, trace = train_rl(cfg, store, corpus, vocab, model)
    console.print(
        f"[green]rl[/green] {len(trace)} steps, final reward {trace[-1]['mean_reward']:.4f}"
    )


@train_app.command("all")
def train_all(config: Path | None = ConfigOption) -> None:
    """Run all three training stages back to back."""
    train_pretrain(config)
    train_wbc_cmd(config)
    train_rl_cmd(config)


# ----------------------------------------------------------------- eval


@eval_app.command("backtest")
def eval_backtest(
    config: Path | None = ConfigOption,
    window: str = typer.Option("test", help="'test' (out-of-sample) or 'valid'."),
) -> None:
    """Walk-forward backtest of every model variant against every baseline."""
    from gendesk.evaluation.experiments import run_backtest_suite

    cfg = _config(config)
    report = run_backtest_suite(cfg, window=window)
    _print_table(report["summary"], title=f"GenDesk backtest ({window})")


@eval_app.command("ablations")
def eval_ablations(
    config: Path | None = ConfigOption,
    epochs: int = 3,
    subsample: float = typer.Option(0.6, help="Fraction of the training split per cell."),
) -> None:
    """Context enrichment versus model capacity, plus architectural ablations."""
    from gendesk.evaluation.ablations import run_ablations

    cfg = _config(config)
    frame = run_ablations(cfg, epochs=epochs, subsample=subsample)
    _print_table(frame.to_dict(orient="records"), title="Ablations")


@eval_app.command("latency")
def eval_latency(config: Path | None = ConfigOption, repeats: int = 20) -> None:
    """Benchmark hybrid row decoding against full autoregression."""
    from gendesk.evaluation.experiments import run_latency_study

    cfg = _config(config)
    rows = run_latency_study(cfg, repeats=repeats)
    _print_table(rows, title="Serving latency")


@eval_app.command("report")
def eval_report(config: Path | None = ConfigOption) -> None:
    """Render the markdown results report from saved run artifacts."""
    from gendesk.evaluation.report import render_report

    cfg = _config(config)
    path = render_report(cfg)
    console.print(f"[green]report[/green] {path}")


@app.command("generate")
def generate_page(
    config: Path | None = ConfigOption,
    persona: str = typer.Option("endowment_balanced"),
    date: str | None = typer.Option(None, help="As-of date; defaults to the last session."),
    checkpoint: str = typer.Option("rl"),
    instruction: str | None = typer.Option(None, help="Plain-English steering."),
) -> None:
    """Generate one desk page and print it."""
    from gendesk.evaluation.strategies import GenDeskStrategy
    from gendesk.steering import apply_instruction
    from gendesk.training.checkpoint import load_checkpoint
    from gendesk.training.schedule import resolve_device

    cfg, store, _, vocab = _bootstrap(config)
    device = resolve_device(cfg.training.device)
    model, _ = load_checkpoint(checkpoint, vocab, device=device)

    mandate = next(p for p in cfg.personas if p.name == persona)
    if instruction:
        mandate, cfg = apply_instruction(instruction, mandate, cfg, catalog=store.catalog)
        console.print(f"[cyan]steering[/cyan] {instruction}")

    position = store.date_position(date) if date else len(store.dates) - 1
    strategy = GenDeskStrategy(model, vocab, store, cfg, mandate)
    weights = strategy(position)
    page = strategy.pages[list(strategy.pages)[-1]]

    table = Table(title=f"{persona} - {store.dates[position].date()}")
    table.add_column("Row")
    table.add_column("Instruments")
    table.add_column("Weight", justify="right")
    for row in page.rows:
        weight = sum(float(weights.get(s, 0.0)) for s in row.symbols)
        table.add_row(row.archetype, ", ".join(row.symbols), f"{weight:.1%}")
    console.print(table)


def _print_table(rows: list[dict] | dict, title: str) -> None:
    if isinstance(rows, dict):
        rows = [{"metric": k, "value": v} for k, v in rows.items()]
    if not rows:
        console.print("[yellow]no rows[/yellow]")
        return
    table = Table(title=title)
    for column in rows[0]:
        table.add_column(str(column))
    for row in rows:
        table.add_row(*[f"{v:.4f}" if isinstance(v, float) else str(v) for v in row.values()])
    console.print(table)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":  # pragma: no cover
    app()
