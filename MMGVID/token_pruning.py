"""MMG-Vid: maximizing marginal gains at segment level and token level.

The three stages of the paper live here:

1. `divide_scene`            - similarity-based frame segmentation
2. `allocate_segment_budget` - marginal gain-based segment budgeting
3. `select_segment_tokens`   - marginal gain-based token pruning (TG-DPC)

`prune_video_tokens` chains them and returns a per-frame boolean keep-mask.
"""

import torch
import torch.nn.functional as F

from .tg_dpc import guided_dpc_knn, typical_dpc_knn


def prune_video_tokens(image_feature, retain_ratio=0.25, token_merge_temperature=0.5):
    """Return a [num_frames, tokens_per_frame] boolean mask of tokens to keep."""
    num_frames, tokens_per_frame, hidden_dim = image_feature.shape

    flattened_image_feature = image_feature.reshape(-1, hidden_dim)
    flattened_image_feature = select_feature_channel(flattened_image_feature)
    scene_idx, scene_embedding_list = divide_scene(
        flattened_image_feature, num_frames=num_frames, tokens_per_frame=tokens_per_frame
    )
    each_scene_mask = allocate_segment_budget(
        flattened_image_feature,
        scene_idx,
        scene_embedding_list,
        retain_ratio=retain_ratio,
        token_merge_temperature=token_merge_temperature,
        num_frames=num_frames,
        tokens_per_frame=tokens_per_frame,
    )

    return each_scene_mask


def select_feature_channel(input_tensor, k=0.5):
    """Keep the lowest-variance half of the channels.

    High-variance channels are dominated by attention-sink dimensions that
    swamp the distance metric; dropping them also halves the cost of every
    pairwise distance computed downstream.
    """
    channel_var = input_tensor.var(dim=0, unbiased=False)
    k = channel_var.shape[0] // 2
    _, topk_indices = torch.topk(channel_var, k=k, largest=False)
    return input_tensor[:, topk_indices]


def divide_scene(image_features, num_frames=64, tokens_per_frame=182, threshold=0.95):
    """Stage 1: split the video into semantically coherent segments.

    `image_features` is [num_frames * tokens_per_frame, embedding_dim]; returns
    the [start, end] index pairs of each segment plus their token embeddings.
    """
    reshaped_features = image_features.view(num_frames, tokens_per_frame, -1)
    # Average-pool each frame's patches into one frame-level descriptor.
    global_frame_features = torch.mean(reshaped_features, dim=1)
    scene_idx = [[-1, -1]]
    sim_list = []
    scene_embedding_list = []

    # A similarity drop between adjacent frames marks a segment boundary.
    for i in range(len(global_frame_features) - 1):
        sim = F.cosine_similarity(
            global_frame_features[i].unsqueeze(0), global_frame_features[i + 1].unsqueeze(0)
        )
        sim_list.append(sim.item())
        if sim < threshold:
            scene_idx.append([scene_idx[-1][1] + 1, i])

    scene_idx.append([scene_idx[-1][1] + 1, len(global_frame_features) - 1])
    scene_idx = scene_idx[1:]  # drop the (-1, -1) sentinel

    # Merge isolated single-frame segments into whichever neighbour they
    # resemble more, so every segment spans at least two frames.
    i = 0
    while i < len(scene_idx):
        scene_start_idx, scene_end_idx = scene_idx[i]
        if scene_start_idx == scene_end_idx:
            left_sim = sim_list[scene_start_idx - 1] if i != 0 else -100
            right_sim = sim_list[scene_start_idx] if i != len(scene_idx) - 1 else -100
            if left_sim > right_sim:
                scene_idx[i - 1][1] = scene_start_idx
            else:
                scene_idx[i + 1][0] = scene_end_idx
            del scene_idx[i]
        else:
            i += 1

    for scene in scene_idx:
        scene_start_idx, scene_end_idx = scene
        scene_embedding_list.append(reshaped_features[scene_start_idx : scene_end_idx + 1, :, :])
    return scene_idx, scene_embedding_list


def get_importance_score_by_distance(x, y):
    """Multi-scale RBF affinity; a similarity measure that is easier to temper."""
    alphas = [2**k for k in range(-3, 2)]

    l2_distance_square = torch.sum((x - y) ** 2, dim=-1)
    k_xy = sum([torch.exp(-l2_distance_square / (2 * alpha)) for alpha in alphas])
    return k_xy


def allocate_segment_budget(
    image_features,
    scene_idx,
    scene_embedding_list,
    retain_ratio=0.25,
    token_merge_temperature=0.5,
    num_frames=64,
    tokens_per_frame=169,
):
    """Stage 2: hand out the token budget segment by segment.

    Every segment is guaranteed a floor of `r_min`; the discretionary remainder
    is given to whichever unselected segment currently has the highest marginal
    value, which combines how representative it is of the *remaining* video with
    how much new information it adds on top of the *already selected* segments.
    Because the reference sets shrink as segments are picked, a segment that
    merely repeats what has been kept sees its value decay.

    Most of the arithmetic below exists to keep the running budget consistent
    with the global `retain_ratio` despite per-segment integer rounding.
    """
    cur_total_tokens, hidden_dim = image_features.shape
    reshaped_features = image_features.view(num_frames, tokens_per_frame, -1)

    each_scene_mask = torch.zeros_like(reshaped_features[:, :, 0], dtype=torch.bool)

    scene_num = len(scene_idx)
    local_scene_idx = scene_idx
    global_scene_idx = scene_idx

    selected_scene_mean_features_list = []

    r_min = retain_ratio * 0.8
    r_max = retain_ratio * 1.2
    extra_retain_ratio = retain_ratio - r_min
    extra_token_budget = int(extra_retain_ratio * image_features.shape[0])

    for i in range(scene_num):
        scene_mean_features_list = [
            scene_embedding.mean(0) for scene_embedding in scene_embedding_list
        ]
        scene_mean_features = torch.stack(scene_mean_features_list)  # [scene_num, N, D]
        cur_video_mean_features = scene_mean_features.mean(0).mean(0)  # [D]

        # Representativeness: agreement with the video that is still unselected.
        each_scene_to_video_importance = F.cosine_similarity(
            scene_mean_features, cur_video_mean_features, dim=2
        )
        each_scene_to_video_importance = torch.sum(each_scene_to_video_importance, dim=1)

        if i == 0:
            each_scene_importance = each_scene_to_video_importance
        else:
            # Diversity: novelty relative to what has already been selected.
            selected_scene_mean_features = torch.stack(selected_scene_mean_features_list).mean(0)
            each_scene_diversity_scores = F.cosine_similarity(
                scene_mean_features, selected_scene_mean_features, dim=2
            )
            each_scene_diversity_scores = 1 - each_scene_diversity_scores
            each_scene_diversity_scores = torch.sum(each_scene_diversity_scores, dim=1)
            each_scene_importance = (
                0.5 * each_scene_to_video_importance + 0.5 * each_scene_diversity_scores
            )

        cur_select_idx = torch.argmax(each_scene_importance).item()

        cur_scene_importance = each_scene_importance[cur_select_idx]
        scene_std = each_scene_importance.std()
        if each_scene_importance.numel() > 1 and scene_std > 1e-8:
            # Scale the extra budget by how far above average this segment is.
            # Flooring the z-score at 1 keeps an average segment at exactly
            # `retain_ratio` and guards against a degenerate zero std.
            z_score_best = (cur_scene_importance - each_scene_importance.mean()) / scene_std
            if z_score_best < 1:
                z_score_best = 1
            cur_scene_extra_retain_ratio = extra_retain_ratio * z_score_best
        else:
            cur_scene_extra_retain_ratio = extra_retain_ratio
        cur_scene_retain_ratio = r_min + cur_scene_extra_retain_ratio

        if cur_scene_retain_ratio > r_max:
            cur_scene_retain_ratio = r_max
            cur_scene_extra_retain_ratio = r_max - r_min

        cur_scene_features = scene_embedding_list[cur_select_idx]
        cur_scene_num_tokens = cur_scene_features.shape[0] * cur_scene_features.shape[1]

        cur_scene_extra_token_budget = int(cur_scene_extra_retain_ratio * cur_scene_num_tokens)
        if cur_scene_extra_token_budget > extra_token_budget:
            cur_scene_extra_token_budget = extra_token_budget
            cur_scene_extra_retain_ratio = cur_scene_extra_token_budget / cur_scene_num_tokens
            cur_scene_retain_ratio = r_min + cur_scene_extra_retain_ratio
        elif extra_token_budget <= 0:
            cur_scene_extra_token_budget = 0
            cur_scene_extra_retain_ratio = 0
            cur_scene_retain_ratio = r_min

        cur_scene_token_budget = int(cur_scene_retain_ratio * cur_scene_num_tokens)
        cur_scene_mask = select_segment_tokens(
            cur_scene_features,
            cur_video_mean_features,
            cur_scene_token_budget,
            cur_scene_retain_ratio,
        )

        global_cur_scene_start_idx, global_cur_scene_end_idx = global_scene_idx[cur_select_idx]
        global_scene_idx = global_scene_idx[:cur_select_idx] + global_scene_idx[cur_select_idx + 1 :]
        each_scene_mask[global_cur_scene_start_idx : global_cur_scene_end_idx + 1] = cur_scene_mask

        extra_token_budget = extra_token_budget - cur_scene_extra_token_budget

        # Drop the segment just handled so the next iteration scores marginal
        # value against a strictly smaller remainder.
        cur_total_tokens = cur_total_tokens - cur_scene_num_tokens
        if cur_total_tokens > 0:
            extra_retain_ratio = extra_token_budget / cur_total_tokens
            scene_embedding_list = (
                scene_embedding_list[:cur_select_idx] + scene_embedding_list[cur_select_idx + 1 :]
            )
            cur_scene_start_idx, cur_scene_end_idx = local_scene_idx[cur_select_idx]
            reshaped_features = torch.cat(
                (reshaped_features[:cur_scene_start_idx], reshaped_features[cur_scene_end_idx + 1 :]),
                dim=0,
            )
            selected_scene_mean_features_list.append(scene_mean_features_list[cur_select_idx])
            offest = cur_scene_end_idx - cur_scene_start_idx + 1
            local_scene_idx[cur_select_idx + 1 :] = [
                (x - offest, y - offest) for x, y in local_scene_idx[cur_select_idx + 1 :]
            ]
            local_scene_idx = local_scene_idx[:cur_select_idx] + local_scene_idx[cur_select_idx + 1 :]
        else:
            extra_retain_ratio = 0

    return each_scene_mask


def select_segment_tokens(
    cur_scene_features, cur_video_mean_features, cur_scene_token_budget, cur_scene_retain_ratio
):
    """Stage 3: prune tokens inside one segment, frame by frame, with TG-DPC.

    The first frame is clustered with plain DPC to seed a retained-token set.
    Each later frame is then pruned *conditioned* on that set, so tokens merely
    repeating already-kept content lose out to genuinely new ones. The set grows
    as we walk forward, which is what makes the gain marginal rather than
    absolute.
    """
    num_frames, tokens_per_frame, hidden_dim = cur_scene_features.shape
    first_frame_features = cur_scene_features[0]
    each_frame_retain_tokens = int(cur_scene_retain_ratio * tokens_per_frame) + 1
    if num_frames == 1:
        each_frame_retain_tokens = cur_scene_token_budget
    cluster_num = each_frame_retain_tokens
    K = 7
    first_frame_select_tokens_idx, dist_matrix = typical_dpc_knn(
        first_frame_features.unsqueeze(0), cluster_num, k=min(K, cluster_num)
    )

    first_frame_select_tokens_idx = first_frame_select_tokens_idx[0]
    cur_tokens_embedding_subset = first_frame_features[first_frame_select_tokens_idx]
    cur_scene_select_tokens_idx = first_frame_select_tokens_idx

    if num_frames > 1:
        for i in range(1, num_frames):
            wait_to_select_token_num = cur_scene_token_budget - i * each_frame_retain_tokens
            if wait_to_select_token_num <= 0:
                break
            # Per-frame budgets are rounded down, so give the last frame
            # whatever the segment has left over.
            if i == num_frames - 1:
                each_frame_retain_tokens = cur_scene_token_budget - (num_frames - 1) * each_frame_retain_tokens

            cur_frame_embedding = cur_scene_features[i]
            cluster_num = each_frame_retain_tokens

            cur_frame_select_tokens_idx, dist_matrix = guided_dpc_knn(
                cur_frame_embedding.unsqueeze(0), cur_tokens_embedding_subset.unsqueeze(0), cluster_num
            )
            cur_frame_select_tokens_idx = cur_frame_select_tokens_idx[0]
            cur_frame_select_embedding = cur_frame_embedding[cur_frame_select_tokens_idx]
            cur_tokens_embedding_subset = torch.cat(
                (cur_tokens_embedding_subset, cur_frame_select_embedding), dim=0
            )

            cur_frame_select_tokens_idx = cur_frame_select_tokens_idx + i * tokens_per_frame
            cur_scene_select_tokens_idx = torch.cat(
                (cur_scene_select_tokens_idx, cur_frame_select_tokens_idx), dim=0
            )

    reshape_cur_scene_features = cur_scene_features.reshape(-1, hidden_dim)
    mask = torch.zeros_like(reshape_cur_scene_features[:, 0])
    mask[cur_scene_select_tokens_idx] = 1
    return mask.reshape(num_frames, tokens_per_frame)
