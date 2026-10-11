#!/usr/bin/env bash
# Evaluate LLaVA-Video-7B with MMG-Vid visual token pruning.
# Edit the settings below, then run:  bash scripts/eval/eval_llava_video.sh
set -euo pipefail

# ------------------------------- edit me -------------------------------------
MODEL_PATH="/path/to/LLaVA-Video-7B-Qwen2"
TASKS="videomme"        # videomme | longvideobench_val_v | mlvu_dev | mvbench
FRAMES=64               # frames sampled from each video
RETAIN_RATIO=0.25       # fraction of visual tokens kept (0.25 prunes 75%)
METHOD="MMGVID"         # MMGVID | none  (none = unpruned baseline)
NUM_PROCESSES=8         # number of GPUs
OUTPUT_PATH="./logs/llava_video_${METHOD}_r${RETAIN_RATIO}"
# ------------------------------------------------------------------------------

export PYTHONPATH="$(pwd):$(pwd)/lmms-eval:${PYTHONPATH:-}"

python3 -m accelerate.commands.launch \
    --num_processes="${NUM_PROCESSES}" \
    -m lmms_eval \
    --model=llava_vid \
    --model_args="pretrained=${MODEL_PATH},conv_template=qwen_1_5,max_frames_num=${FRAMES},mm_spatial_pool_mode=average,attn_implementation=flash_attention_2,method=${METHOD},retain_ratio=${RETAIN_RATIO}" \
    --tasks="${TASKS}" \
    --batch_size=1 \
    --log_samples \
    --log_samples_suffix="llava_vid_${METHOD}" \
    --output_path="${OUTPUT_PATH}"
