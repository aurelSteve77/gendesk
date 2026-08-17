#!/usr/bin/env python
"""Inject the generated results into README.md between the RESULTS markers.

The README quotes numbers; those numbers must come from artifacts rather than from
someone's memory of what the run said. This script is the only thing allowed to write
that section.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "artifacts" / "reports"
BEGIN = "<!-- RESULTS:BEGIN -->"
END = "<!-- RESULTS:END -->"

LABELS = {
    "gendesk_rl": "**GenDesk (RL)**",
    "gendesk_wbc": "GenDesk (WBC head)",
    "gendesk_pretrain": "GenDesk (pretrained)",
    "pipeline_multistage": "Multi-stage pipeline",
    "teacher_book": "Teacher screen",
    "benchmark_spy": "S&P 500 ETF",
    "equal_weight": "Equal weight",
    "momentum_12_1": "12-1 momentum",
    "low_volatility": "Low volatility",
    "risk_parity": "Risk parity",
}


def _table(frame: pd.DataFrame, formats: dict[str, str]) -> str:
    out = frame.copy()
    for column, spec in formats.items():
        if column in out.columns:
            out[column] = out[column].map(lambda v, s=spec: s.format(v) if pd.notna(v) else "-")
    header = "| " + " | ".join(str(c) for c in out.columns) + " |"
    divider = "| " + " | ".join("---" for _ in out.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in out.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *body])


def build_section() -> str:
    parts: list[str] = []

    path = REPORTS / "backtest_test.json"
    if path.exists():
        report = json.loads(path.read_text())
        summary = pd.DataFrame(report["summary"])
        summary["strategy"] = summary["strategy"].map(lambda s: LABELS.get(s, s))
        columns = ["strategy", "cagr", "vol", "sharpe", "max_drawdown", "annual_turnover"]
        table = summary[[c for c in columns if c in summary.columns]].rename(
            columns={
                "strategy": "Strategy",
                "cagr": "CAGR",
                "vol": "Vol",
                "sharpe": "Sharpe",
                "max_drawdown": "Max DD",
                "annual_turnover": "Turnover",
            }
        )
        parts += [
            f"**Out-of-sample, {report['start']} to {report['end']}** "
            f"({report['config']['backtest']['cost_bps']:.0f} bp costs, rebalanced every "
            f"{report['config']['backtest']['rebalance_days']} sessions, equal-weighted across "
            "six mandates).",
            "",
            _table(
                table,
                {
                    "CAGR": "{:.1%}",
                    "Vol": "{:.1%}",
                    "Sharpe": "{:.2f}",
                    "Max DD": "{:.1%}",
                    "Turnover": "{:.1f}x",
                },
            ),
            "",
        ]

        comparisons = report.get("comparisons", {})
        if comparisons:
            rows = pd.DataFrame(
                [
                    {
                        "Comparison": key.replace("gendesk_rl_vs_", "GenDesk (RL) vs ").replace("_", " "),
                        "Sharpe difference": payload["estimate"],
                        "95% CI": f"[{payload['ci_low']:.2f}, {payload['ci_high']:.2f}]",
                        "p": payload["p_value"],
                    }
                    for key, payload in comparisons.items()
                ]
            )
            parts += [
                "Paired block-bootstrap comparisons (both legs resampled on the same blocks):",
                "",
                _table(rows, {"Sharpe difference": "{:+.2f}", "p": "{:.3f}"}),
                "",
            ]

        pbo = report.get("pbo", {})
        if pbo.get("pbo") == pbo.get("pbo"):
            parts.append(
                f"Probability of backtest overfitting (CSCV over {pbo['n_configs']} "
                f"configurations): **{pbo['pbo']:.0%}**."
            )
            parts.append("")

    path = REPORTS / "ablations.csv"
    if path.exists():
        sys.path.insert(0, str(ROOT / "src"))
        from gendesk.evaluation.ablations import summarise_headline

        frame = pd.read_csv(path)
        headline = summarise_headline(frame)
        if headline:
            parts += [
                "**Prompt content beats parameter count.** Same corpus, same validation split, "
                "same epochs, one variable at a time:",
                "",
                f"- Full context vs a rows-only prompt at fixed capacity: "
                f"**{headline['context_loss_reduction']:.1%}** validation-loss reduction "
                f"({headline['context_mrr_gain']:+.1%} MRR).",
                f"- {headline['capacity_multiple']:.0f}x parameters at fixed full context: "
                f"**{headline['capacity_loss_reduction']:.1%}** "
                f"({headline['capacity_mrr_gain']:+.1%} MRR).",
                "",
            ]

    path = REPORTS / "latency.json"
    if path.exists():
        rows = json.loads(path.read_text())
        by_mode = {r["mode"]: r for r in rows}
        if "reduction" in by_mode:
            reduction = by_mode["reduction"]
            parts += [
                f"**Hybrid row decoding** cuts sequential model invocations per page from "
                f"{by_mode['autoregressive']['sequential_model_calls']:.0f} to "
                f"{by_mode['hybrid_row']['sequential_model_calls']:.0f} "
                f"({reduction['sequential_model_calls']:.0%} fewer) for a "
                f"{reduction['median_ms']:.0%} reduction in median wall-clock latency.",
                "",
            ]

    if not parts:
        return "*Run `make pipeline` to populate this section.*"
    parts.append(
        "Full report, including per-mandate results, deflated Sharpe ratios and the RL "
        "diversification study: [`artifacts/reports/RESULTS.md`](artifacts/reports/RESULTS.md)."
    )
    return "\n".join(parts)


def main() -> int:
    readme = ROOT / "README.md"
    text = readme.read_text()
    if BEGIN not in text or END not in text:
        print("markers not found in README.md", file=sys.stderr)
        return 1

    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{BEGIN}\n{build_section()}\n{END}{tail}")
    print(f"README results section updated ({readme})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
