"""
Shim for backward compatibility.
Moved to biorsp.core.geometry
"""

import warnings

from biorsp.core.geometry import *  # noqa: F403

warnings.warn(
    "biorsp.preprocess.geometry has been moved to biorsp.core.geometry. "
    "Please update your imports. This shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)
