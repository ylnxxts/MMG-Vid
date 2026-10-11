"""Density peak clustering used for token-level selection.

`typical_dpc_knn` is standard DPC over a single frame. `guided_dpc_knn` is the
paper's Temporal-Guided DPC (TG-DPC): it redefines density with respect to the
tokens already kept from earlier frames, so a token scores highly only when it
is *novel across time* as well as *distinctive within its own frame*.
"""

import torch


def typical_dpc_knn(x, cluster_num, k=7):
    """Plain DPC: density from the k nearest neighbours inside `x`."""
    batch_size, seq_len, embed_dim = x.shape

    dist_matrix = torch.cdist(x.float(), x.float()) / (embed_dim**0.5)

    k = min(k, seq_len - 1)
    dist_nearest, index_nearest = torch.topk(dist_matrix, k, dim=-1, largest=False)
    density = (-(dist_nearest**2).mean(dim=-1)).exp()
    # Tiny noise so no two tokens share a density and the strict comparison
    # below always induces a total order.
    density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6

    # For each token, the distance to the nearest token of higher density.
    # Tokens clustered around a peak attach to that peak and score low.
    mask = (density[:, None, :] > density[:, :, None]).type(x.dtype)
    dist_max = dist_matrix.flatten(1).max(dim=-1).values[:, None, None]
    dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

    score = dist * density
    _, index_center = score.topk(cluster_num, dim=-1)

    return index_center, dist_matrix


def guided_dpc_knn(
    x_target: torch.Tensor,
    x_guide: torch.Tensor,
    selection_num: int,
    dc_percentile: float = 0.5,
):
    """TG-DPC: select tokens from `x_target` that are novel w.r.t. `x_guide`.

    `x_guide` holds the tokens already retained from previous frames of the same
    segment. Density is the *distance* to that set, so tokens duplicating
    already-kept content are suppressed, while the separation term still favours
    tokens that are spread out within the current frame.
    """
    batch_size, target_seq_len, embed_dim = x_target.shape

    dist_matrix_internal = torch.cdist(x_target.float(), x_target.float()) / (embed_dim**0.5)
    dist_matrix_guided = torch.cdist(x_target.float(), x_guide.float()) / (embed_dim**0.5)

    # Inter-frame uniqueness: far from the retained set means high density here.
    k = x_guide.shape[1]
    dist_nearest, index_nearest = torch.topk(dist_matrix_guided, k, dim=-1, largest=False)
    density = 1 - (-(dist_nearest**2).mean(dim=-1)).exp()
    density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6

    # Intra-frame diversity: distance to the nearest token that is more unique.
    mask = (density[:, None, :] > density[:, :, None]).type(x_target.dtype)
    dist_max = dist_matrix_internal.flatten(1).max(dim=-1).values[:, None, None]
    separation_delta, _ = (dist_matrix_internal * mask + dist_max * (1 - mask)).min(dim=-1)

    score = density * separation_delta
    _, index_selected = score.topk(selection_num, dim=-1)

    return index_selected, dist_matrix_internal
