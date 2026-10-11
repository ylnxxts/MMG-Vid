"""Entry point that patches a LLaVA model in place with MMG-Vid pruning.

MMG-Vid is training-free, so enabling it is just a matter of swapping two
methods: the multimodal input builder (where LLaVA-OneVision is pruned) and the
Qwen2 forward pass (where LLaVA-Video is pruned, during prefill only).
"""

import os

from .llava_arch import LlavaMetaForCausalLM_MMGVID
from .modeling_qwen2 import Qwen2Model_MMGVID


def MMGVID(model):
    retain_ratio = float(os.environ.get("retain_ratio", 0.25))
    print(f"[MMG-Vid] token pruning enabled, retain_ratio={retain_ratio}")

    from llava.model.llava_arch import LlavaMetaForCausalLM

    LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = (
        LlavaMetaForCausalLM_MMGVID.prepare_inputs_labels_for_multimodal
    )

    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

    Qwen2Model.forward = Qwen2Model_MMGVID.forward
    Qwen2Model.add_token_per_grid = Qwen2Model_MMGVID.add_token_per_grid

    return model
