"""Three-stage training: pretraining, WBC post-training, page-level RL."""

from gendesk.training.dataset import PageBatch, PageDataset, collate_pages
from gendesk.training.pretrain import pretrain
from gendesk.training.rl import train_rl
from gendesk.training.wbc import train_wbc

__all__ = ["PageBatch", "PageDataset", "collate_pages", "pretrain", "train_rl", "train_wbc"]
