# MMG-Vid：在段级与词元级最大化边际增益的高效视频大模型

**MMG-Vid**（AAAI 2026）官方代码实现。

[[论文](https://arxiv.org/abs/2508.21044)] · [English](README.md)

MMG-Vid 是一种**免训练**的视频大模型视觉词元剪枝框架。
以往方法把视频当作彼此独立的多帧图像：给每个片段分配相同的词元预算，
并把多样性与重要性拆成两个互不相通的阶段来判断。
MMG-Vid 转而最大化**边际增益**——一个片段或一个词元的价值，
是相对于*已经保留的内容*来衡量的。

在 LLaVA-OneVision-7B 上，它剪掉 75% 的视觉词元，仍保持原模型 99.5% 的精度，
并将 prefill 阶段加速 3.9 倍。

## 方法

整个流程分三个阶段，代码都在 `MMGVID/` 下：

**1. 基于相似度的帧分段**（`divide_scene`）：在特征空间比较相邻帧，
相似度骤降处切分出新片段；只含单帧的孤立片段会被并入最相似的邻居，
保证每个片段至少跨两帧。

**2. 基于边际增益的段级预算分配**（`allocate_segment_budget`）：
每个片段先获得保底词元数，剩余预算再逐轮"竞拍"。
每一轮中，片段的边际价值由两部分构成：它对*尚未选中*视频内容的代表性，
以及它相对*已选中*片段带来的新信息量。
重复已有内容的片段价值会衰减，冗余镜头因此不再继续占用预算。

**3. 基于边际增益的词元剪枝 TG-DPC**（`select_segment_tokens`、`guided_dpc_knn`）：
片段首帧先用标准密度峰值聚类挑出初始保留集合；
之后每一帧都*以该集合为条件*进行剪枝——密度被重新定义为到已保留词元的距离，
从而把帧间独特性与帧内多样性联合建模，而不是分两步串行处理。

## 实验结果

LLaVA-Video-7B 在 MVBench / LongVideoBench / MLVU / VideoMME 上的平均分，
以未剪枝模型为 100%（论文表 1）：

| 保留比例 | FastV | VisionZip | PruneVid | FrameFusion | **MMG-Vid** |
|---|---|---|---|---|---|
| 25% | 89.4 | 96.3 | 95.2 | 96.8 | **98.1** |
| 20% | 86.4 | 94.7 | 94.4 | 95.2 | **97.0** |
| 15% | 81.7 | 93.8 | 93.3 | 92.1 | **96.0** |
| 10% | 77.1 | 90.4 | 91.2 | 88.3 | **93.8** |

推理延迟（论文表 2）：

| 模型 | 保留比例 | Prefill | Generate | 精度% |
|---|---|---|---|---|
| LLaVA-OneVision | 100% | 207.5 ms (1.0x) | 329.7 ms (1.0x) | 100 |
| LLaVA-OneVision | 25% | 52.8 ms (**3.9x**) | 107.2 ms (3.1x) | 99.5 |
| LLaVA-OneVision | 15% | 34.6 ms (**6.0x**) | 78.3 ms (4.2x) | 98.3 |
| LLaVA-Video | 100% | 277.5 ms (1.0x) | 411.6 ms (1.0x) | 100 |
| LLaVA-Video | 25% | 65.0 ms (**4.3x**) | 142.9 ms (2.9x) | 98.1 |
| LLaVA-Video | 15% | 42.5 ms (**6.5x**) | 109.6 ms (3.8x) | 96.0 |

压缩越激进，优势越明显。

## 仓库结构

```
MMGVID/                核心方法
  token_pruning.py       三个阶段：分段、预算分配、词元选择
  tg_dpc.py              时序引导的密度峰值聚类（TG-DPC）
  main.py                对已加载的 LLaVA 模型做原地替换
  llava_arch.py          改写后的 prepare_inputs_labels_for_multimodal
  modeling_qwen2.py      改写后的 Qwen2 forward（仅在 prefill 阶段剪枝）
llava/                 LLaVA-NeXT（内置）
lmms-eval/             带 MMG-Vid 接入点的 lmms-eval 评测框架
scripts/eval/          开箱即用的评测脚本
tests/                 仅依赖 torch 的冒烟测试
```

MMG-Vid 通过对已加载模型替换两个方法来启用，不改动权重，也不涉及训练流程：

```python
from MMGVID import MMGVID
model = MMGVID(model)   # retain_ratio 从环境变量读取
```

对内置评测框架的改动都用 `MMG-Vid Start` / `MMG-Vid End` 注释包裹，
执行 `grep -rn "MMG-Vid Start"` 即可看到全部接入点。

## 安装

```bash
conda create -n mmgvid python=3.10 -y
conda activate mmgvid
pip install -r requirements.txt
pip install flash-attn==2.5.7 --no-build-isolation   # 请匹配你的 CUDA/torch 版本
pip install -e .
```

剪枝部分只依赖 torch，因此无需下载任何权重即可先验证安装是否正常：

```bash
python -m pytest tests/ -q
```

## 运行评测

`scripts/eval/` 下每个脚本开头都有一段 `edit me` 配置区，改好路径直接运行即可：

```bash
bash scripts/eval/eval_llava_video.sh
```

```bash
# ------------------------------- edit me -------------------------------------
MODEL_PATH="/path/to/LLaVA-Video-7B-Qwen2"
TASKS="videomme"        # videomme | longvideobench_val_v | mlvu_dev | mvbench
FRAMES=64               # 每个视频采样的帧数
RETAIN_RATIO=0.25       # 保留的视觉词元比例（0.25 即剪掉 75%）
METHOD="MMGVID"         # MMGVID | none（none 为未剪枝基线）
NUM_PROCESSES=8         # GPU 数量
```

`RETAIN_RATIO` 是保留的视觉词元比例，`0.25` 即对应上表中剪掉 75% 的那一行；
设为 `METHOD="none"` 可跑未剪枝的基线。同目录下另有 LLaVA-OneVision 的对应脚本。

评测数据由 lmms-eval 在首次运行时自动从 Hub 拉取，无需手动准备。
机器不能联网时，可在别处下载好数据集再把 `HF_HOME` 指过去：

```bash
export HF_HOME=/path/to/huggingface
```

## 主要超参数

| 名称 | 位置 | 默认值 | 含义 |
|---|---|---|---|
| `retain_ratio` | 评测 `--model_args` | 0.25 | 保留的视觉词元比例 |
| `threshold` | `divide_scene` | 0.95 | 低于该余弦相似度即切分新片段 |
| `r_min` / `r_max` | `allocate_segment_budget` | 0.8 / 1.2 倍比例 | 单个片段预算的下限与上限 |
| `K` | `select_segment_tokens` | 7 | 首帧 DPC 计算密度所用的近邻数 |

## 引用

```bibtex
@inproceedings{ma2026mmgvid,
  title     = {MMG-Vid: Maximizing Marginal Gains at Segment-level and Token-level
               for Efficient Video LLMs},
  author    = {Ma, Junpeng and Zhang, Qizhe and Lu, Ming and Wang, Zhibin and
               Zhou, Qiang and Song, Jun and Zhang, Shanghang},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2026}
}
```

## 致谢

本项目基于 [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) 与
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) 构建；
词元级阶段的设计受 [DivPrune](https://github.com/vbdi/divprune) 的 Max-Min 多样性建模启发。

## 许可证

遵循 LLaVA-NeXT 采用 Apache 2.0 许可证，详见 [LICENSE](LICENSE)。
