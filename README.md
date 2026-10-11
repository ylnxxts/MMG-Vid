# MMG-Vid: Maximizing Marginal Gains at Segment-level and Token-level for Efficient Video LLMs

Official implementation of **MMG-Vid** (AAAI 2026).

[[Paper](https://arxiv.org/abs/2508.21044)] · [中文说明](README_zh.md)

MMG-Vid is a **training-free** visual token pruning framework for Video LLMs.
Prior methods treat video as a stack of independent frames: they hand every
segment the same token budget and decide diversity and importance in two
disconnected passes. MMG-Vid instead maximizes **marginal gain** — the value of
a segment or a token measured *against what has already been kept*.

On LLaVA-OneVision-7B it removes 75% of visual tokens while retaining 99.5% of
the original accuracy and accelerating prefill by 3.9x.

## Method

The pipeline has three stages, all in `MMGVID/`:

**1. Similarity-based frame segmentation** (`divide_scene`). Adjacent frames are
compared in embedding space; a similarity drop opens a new segment. Isolated
single-frame segments are merged into their closest neighbour so every segment
spans at least two frames.

**2. Marginal gain-based segment budgeting** (`allocate_segment_budget`). Every
segment is guaranteed a floor of tokens; the remainder is auctioned off
iteratively. At each step a segment's marginal value combines how representative
it is of the *still-unselected* video with how much novelty it adds over the
*already-selected* segments. A segment repeating content that is already covered
sees its value decay, so redundant footage stops consuming budget.

**3. Marginal gain-based token pruning via TG-DPC** (`select_segment_tokens`,
`guided_dpc_knn`). The first frame of a segment is clustered with standard
density peak clustering to seed a retained set. Every later frame is pruned
*conditioned* on that set: density is redefined as distance to the tokens
already kept, so inter-frame uniqueness and intra-frame diversity are scored
jointly rather than in two sequential passes.

## Results

LLaVA-Video-7B, average across MVBench / LongVideoBench / MLVU / VideoMME, as a
percentage of the unpruned model (Table 1 of the paper):

| Retention | FastV | VisionZip | PruneVid | FrameFusion | **MMG-Vid** |
|---|---|---|---|---|---|
| 25% | 89.4 | 96.3 | 95.2 | 96.8 | **98.1** |
| 20% | 86.4 | 94.7 | 94.4 | 95.2 | **97.0** |
| 15% | 81.7 | 93.8 | 93.3 | 92.1 | **96.0** |
| 10% | 77.1 | 90.4 | 91.2 | 88.3 | **93.8** |

Inference latency (Table 2):

| Model | Retention | Prefill | Generate | Acc% |
|---|---|---|---|---|
| LLaVA-OneVision | 100% | 207.5 ms (1.0x) | 329.7 ms (1.0x) | 100 |
| LLaVA-OneVision | 25% | 52.8 ms (**3.9x**) | 107.2 ms (3.1x) | 99.5 |
| LLaVA-OneVision | 15% | 34.6 ms (**6.0x**) | 78.3 ms (4.2x) | 98.3 |
| LLaVA-Video | 100% | 277.5 ms (1.0x) | 411.6 ms (1.0x) | 100 |
| LLaVA-Video | 25% | 65.0 ms (**4.3x**) | 142.9 ms (2.9x) | 98.1 |
| LLaVA-Video | 15% | 42.5 ms (**6.5x**) | 109.6 ms (3.8x) | 96.0 |

The advantage widens as compression gets more aggressive.

## Repository layout

```
MMGVID/                Core method
  token_pruning.py       Stages 1-3: segmentation, budgeting, token selection
  tg_dpc.py              Temporal-Guided DPC
  main.py                Patches a loaded LLaVA model in place
  llava_arch.py          Patched prepare_inputs_labels_for_multimodal
  modeling_qwen2.py      Patched Qwen2 forward (prunes during prefill)
llava/                 LLaVA-NeXT (vendored)
lmms-eval/             lmms-eval harness with the MMG-Vid hooks
scripts/eval/          Ready-to-run evaluation scripts
tests/                 Torch-only smoke tests
```

MMG-Vid is enabled by monkey-patching two methods on an already-loaded model, so
nothing about the checkpoint or the training pipeline changes:

```python
from MMGVID import MMGVID
model = MMGVID(model)   # reads retain_ratio from the environment
```

Edits to the vendored harness are bracketed by `MMG-Vid Start` / `MMG-Vid End`
comments; `grep -rn "MMG-Vid Start"` shows the full integration surface.

## Installation

```bash
conda create -n mmgvid python=3.10 -y
conda activate mmgvid
pip install -r requirements.txt
pip install flash-attn==2.5.7 --no-build-isolation   # match your CUDA/torch build
pip install -e .
```

The pruning code needs nothing but torch, so you can check it works before
downloading any checkpoint:

```bash
python -m pytest tests/ -q
```

## Running evaluation

Each script in `scripts/eval/` opens with an `edit me` block. Set your paths
there and run it:

```bash
bash scripts/eval/eval_llava_video.sh
```

```bash
# ------------------------------- edit me -------------------------------------
MODEL_PATH="/path/to/LLaVA-Video-7B-Qwen2"
TASKS="videomme"        # videomme | longvideobench_val_v | mlvu_dev | mvbench
FRAMES=64               # frames sampled from each video
RETAIN_RATIO=0.25       # fraction of visual tokens kept (0.25 prunes 75%)
METHOD="MMGVID"         # MMGVID | none  (none = unpruned baseline)
NUM_PROCESSES=8         # number of GPUs
```

`RETAIN_RATIO` is the fraction of visual tokens kept, so `0.25` reproduces the
75%-pruned row above; set `METHOD="none"` for the unpruned baseline. A matching
script for LLaVA-OneVision sits alongside it.

Benchmarks are pulled from the Hub by lmms-eval on first use, so no manual
dataset setup is needed. On a machine without internet access, download them
elsewhere and point `HF_HOME` at that directory:

```bash
export HF_HOME=/path/to/huggingface
```

## Key hyper-parameters

| Name | Where | Default | Meaning |
|---|---|---|---|
| `retain_ratio` | eval `--model_args` | 0.25 | Fraction of visual tokens kept |
| `threshold` | `divide_scene` | 0.95 | Cosine similarity below which a new segment starts |
| `r_min` / `r_max` | `allocate_segment_budget` | 0.8x / 1.2x ratio | Floor and ceiling on any one segment's budget |
| `K` | `select_segment_tokens` | 7 | Neighbours used for the first frame's DPC density |

## Citation

```bibtex
@inproceedings{ma2026mmg,
  title={Mmg-vid: Maximizing marginal gains at segment-level and token-level for efficient video llms},
  author={Ma, Junpeng and Zhang, Qizhe and Lu, Ming and Wang, Zhibin and Zhou, Qiang and Song, Jun and Zhang, Shanghang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={29},
  pages={24253--24261},
  year={2026}
}
```

## Acknowledgements

Built on [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) and
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). The token-level stage
is inspired by [DivPrune](https://github.com/vbdi/divprune)'s Max-Min diversity
formulation.

## License

Released under the Apache 2.0 License, following LLaVA-NeXT. See [LICENSE](LICENSE).
