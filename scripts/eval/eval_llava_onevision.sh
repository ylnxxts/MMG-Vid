#!/usr/bin/env bash
# Evaluate LLaVA-OneVision-7B with MMG-Vid visual token pruning.
# Edit the settings below, then run:  bash scripts/eval/eval_llava_onevision.sh
set -euo pipefail

# ------------------------------- edit me -------------------------------------
MODEL_PATH="/path/to/llava-onevision-qwen2-7b-ov"
TASKS="videomme"        # videomme | longvideobench_val_v | mlvu_dev | mvbench
FRAMES=32               # frames sampled from each video
RETAIN_RATIO=0.25       # fraction of visual tokens kept (0.25 prunes 75%)
METHOD="MMGVID"         # MMGVID | none  (none = unpruned baseline)
NUM_PROCESSES=8         # number of GPUs
OUTPUT_PATH="./logs/llava_ov_${METHOD}_r${RETAIN_RATIO}"
# ------------------------------------------------------------------------------

export PYTHONPATH="$(pwd):$(pwd)/lmms-eval:${PYTHONPATH:-}"

python3 -m accelerate.commands.launch \
    --num_processes="${NUM_PROCESSES}" \
    -m lmms_eval \
    --model=llava_onevision \
    --model_args="pretrained=${MODEL_PATH},conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=${FRAMES},method=${METHOD},retain_ratio=${RETAIN_RATIO}" \
    --tasks="${TASKS}" \
    --batch_size=1 \
    --log_samples \
    --log_samples_suffix="llava_ov_${METHOD}" \
    --output_path="${OUTPUT_PATH}"
