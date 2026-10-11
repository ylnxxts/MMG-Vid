"""Smoke tests for MMG-Vid's pruning. Requires only torch:

    python -m pytest tests/ -q
"""

import torch

from MMGVID.tg_dpc import guided_dpc_knn, typical_dpc_knn
from MMGVID.token_pruning import divide_scene, prune_video_tokens, select_feature_channel


def _video(num_frames=32, tokens_per_frame=169, dim=128, seed=0):
    torch.manual_seed(seed)
    return torch.randn(num_frames, tokens_per_frame, dim)


def test_mask_shape_matches_video():
    video = _video()
    mask = prune_video_tokens(video, retain_ratio=0.25)
    assert mask.shape == (32, 169)


def test_kept_fraction_tracks_retain_ratio():
    video = _video(num_frames=64, tokens_per_frame=182)
    for ratio in (0.10, 0.15, 0.20, 0.25):
        kept = prune_video_tokens(video, retain_ratio=ratio).sum().item()
        fraction = kept / (64 * 182)
        # Per-segment floors and integer rounding allow some slack, but the
        # global budget must stay in the neighbourhood of the target.
        assert 0.5 * ratio <= fraction <= 1.5 * ratio, (ratio, fraction)


def test_lower_ratio_keeps_fewer_tokens():
    video = _video()
    counts = [prune_video_tokens(video, retain_ratio=r).sum().item() for r in (0.10, 0.20, 0.30)]
    assert counts == sorted(counts)


def test_channel_selection_halves_dimension():
    x = torch.randn(500, 256)
    assert select_feature_channel(x).shape == (500, 128)


def test_segments_tile_the_video_without_gaps():
    video = _video(num_frames=32, tokens_per_frame=64)
    flat = select_feature_channel(video.reshape(-1, 128))
    scene_idx, embeds = divide_scene(flat, num_frames=32, tokens_per_frame=64)
    assert scene_idx[0][0] == 0
    assert scene_idx[-1][1] == 31
    for (_, end), (start, _) in zip(scene_idx, scene_idx[1:]):
        assert start == end + 1
    assert len(embeds) == len(scene_idx)


def test_dpc_selects_requested_number_of_centers():
    x = torch.randn(1, 169, 128)
    assert typical_dpc_knn(x, 40)[0].shape == (1, 40)
    guide = torch.randn(1, 25, 128)
    assert guided_dpc_knn(x, guide, 40)[0].shape == (1, 40)


def test_tg_dpc_avoids_tokens_already_retained():
    """Tokens duplicating the guide set should lose to novel ones."""
    torch.manual_seed(0)
    novel = torch.randn(1, 20, 64)
    guide = torch.randn(1, 20, 64)
    candidates = torch.cat([guide, novel], dim=1)  # first 20 duplicate the guide
    picked = guided_dpc_knn(candidates, guide, 10)[0][0].tolist()
    assert sum(i >= 20 for i in picked) > sum(i < 20 for i in picked)
