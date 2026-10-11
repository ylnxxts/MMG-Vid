"""MMG-Vid: training-free visual token pruning for video LLMs.

The pruning algorithm (`prune_video_tokens` and friends) depends only on torch,
so it can be imported and reused on its own. The `MMGVID` patcher additionally
needs a LLaVA-NeXT install, so it is resolved lazily.
"""

from .tg_dpc import guided_dpc_knn, typical_dpc_knn
from .token_pruning import (
    allocate_segment_budget,
    divide_scene,
    prune_video_tokens,
    select_segment_tokens,
)

__all__ = [
    "MMGVID",
    "allocate_segment_budget",
    "divide_scene",
    "guided_dpc_knn",
    "prune_video_tokens",
    "select_segment_tokens",
    "typical_dpc_knn",
]


def __getattr__(name):
    if name == "MMGVID":
        from .main import MMGVID

        return MMGVID
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
