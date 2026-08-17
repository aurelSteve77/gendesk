"""Point-in-time feature and regime construction.

Every quantity in this package is computed from a trailing window that ends on the
observation date, so a feature row stamped ``t`` is knowable at the close of ``t``
and is legitimately usable to trade at the open of ``t+1``. ``tests/test_leakage.py``
enforces this by perturbing the future and asserting that no feature moves.
"""

from gendesk.features.regimes import REGIME_AXES, build_regimes
from gendesk.features.store import FeatureStore, build_features, load_features

__all__ = [
    "REGIME_AXES",
    "FeatureStore",
    "build_features",
    "build_regimes",
    "load_features",
]
