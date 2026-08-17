"""Evaluation: diversity, statistics, backtesting, ablations."""

from gendesk.evaluation.diversity import DiversityMetrics, page_diversity
from gendesk.evaluation.statistics import (
    SharpeTest,
    block_bootstrap_sharpe,
    deflated_sharpe_ratio,
    performance_summary,
    probability_of_backtest_overfitting,
)

__all__ = [
    "DiversityMetrics",
    "SharpeTest",
    "block_bootstrap_sharpe",
    "deflated_sharpe_ratio",
    "page_diversity",
    "performance_summary",
    "probability_of_backtest_overfitting",
]
