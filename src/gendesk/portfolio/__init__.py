"""Turning a generated page into a portfolio, and scoring what it earned."""

from gendesk.portfolio.reward import PageReward, evaluate_page, slot_rewards
from gendesk.portfolio.weights import page_weights, row_budgets

__all__ = ["PageReward", "evaluate_page", "page_weights", "row_budgets", "slot_rewards"]
