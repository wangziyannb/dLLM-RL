# TraDo-8B-Thinking ablation pipeline

这套 pipeline 是给 `Gen-Verse/TraDo-8B-Thinking`（以及同类 TraDo block diffusion model）做 **throughput / performance trade-off** 实验用的，目标是直接回答两件事：

1. **block size sweep**：不同 `block_size` 对吞吐和精度的影响。
2. **confidence-based parallel decoding sweep**：固定 block size，只扫 `dynamic_threshold`，观察并行解码激进程度对吞吐和精度的影响。

它围绕官方 `dLLM-RL` 仓库的推理/评测入口，不重写采样器，只负责：

- 自动准备/下载 `GSM8K` 与 `MATH500`；
- 自动生成每个 ablation point 的 config；
- 调用官方 `sample/trado_sample.py` 与 `reward/reward.py`；
- 统计 generation throughput、accuracy、平均输出长度；
- 如果 `output_unmasking_history=True`，额外统计 **realized parallelism**；
- 输出 CSV/JSON，并画 trade-off 图。

## 为什么要把两组 sweep 分开

### A. block size sweep
如果你把 `block_size` 从 4 改到 8，但 `denoising_steps_per_block` 仍固定为 4，那么你同时改了两件事：

- block 内 token 数变大；
- 每一步最少会 transfer 的 token 数也变大。

这会把 **block granularity** 和 **parallel decoding aggressiveness** 混在一起。

所以本 pipeline 默认采用：

- `block_size ∈ {2, 4, 8, 16}`
- `denoising_steps_per_block = block_size`

### B. confidence-based parallel decoding sweep
固定：

- `block_size = 4`
- `denoising_steps_per_block = 4`
- `remasking_strategy = low_confidence_dynamic`

只扫：

- `dynamic_threshold ∈ {0.70, 0.80, 0.85, 0.90, 0.95, 0.98}`

## 快速开始

假设你已经把官方仓库 clone 到本地并装好依赖：

```bash
git clone https://github.com/Gen-Verse/dLLM-RL.git
cd dLLM-RL
```

### 1. 快速 smoke test

```bash
python /path/to/run_trado_ablations.py \
  --repo-root /path/to/dLLM-RL \
  --model /path/to/TraDo-8B-Thinking \
  --datasets GSM8K MATH500 \
  --subset-size 32 \
  --max-token 4096 \
  --max-active 2
```

### 2. 正式实验

```bash
python /path/to/run_trado_ablations.py \
  --repo-root /path/to/dLLM-RL \
  --model /path/to/TraDo-8B-Thinking \
  --datasets GSM8K MATH500 \
  --block-sizes 2,4,8,16 \
  --parallel-thresholds 0.70,0.80,0.85,0.90,0.95,0.98 \
  --parallel-base-block-size 4 \
  --max-token 10000 \
  --max-active 2 \
  --tensor-parallel-size 1
```

## 输出

每个 sweep point 会写到：

```text
<repo-root>/runs/trado_ablation/<run-name>/<dataset>/<sweep>/<label>/
```

包含：

- `config.yaml`
- `logs/sample.log`
- `logs/reward.log`
- `metrics.json`

整体汇总会写到：

```text
<repo-root>/runs/trado_ablation/<run-name>/
```

包含：

- `summary.csv`
- `summary.jsonl`
- `plots/*.png`

## 重点指标

### throughput

- `generation_wall_clock_sec`
- `samples_per_sec`
- `output_tokens_per_sec`

### performance

- `accuracy`
- `avg_output_tokens`

### realized parallelism

当 `output_unmasking_history=True` 时，还会写：

- `avg_decode_rounds`
- `avg_realized_tokens_per_round`
- `avg_first_unmask_round`

推荐最后重点看：

- `output_tokens_per_sec -> accuracy`
- `avg_realized_tokens_per_round -> accuracy`

## 单独重画图

```bash
python /path/to/plot_trado_tradeoff.py \
  --summary-csv /path/to/summary.csv \
  --output-dir /path/to/plots
```

可选：

```bash
python /path/to/plot_trado_tradeoff.py \
  --summary-csv /path/to/summary.csv \
  --output-dir /path/to/plots \
  --throughput-metric samples_per_sec
```

或：

```bash
python /path/to/plot_trado_tradeoff.py \
  --summary-csv /path/to/summary.csv \
  --output-dir /path/to/plots \
  --throughput-metric avg_realized_tokens_per_round
```

## 默认 baseline（接近官方 long-CoT 配置）

- `block_size = 4`
- `denoising_steps_per_block = 4`
- `dynamic_threshold = 0.90`
- `remasking_strategy = low_confidence_dynamic`
- `start_with_think = True`
- `num_response_per_task = 1`
- `max_active = 2`

## 常见坑

1. 不要在 block sweep 里把 `block_size` 改了，同时把 `denoising_steps_per_block` 固定死，除非你就是想把两种效应一起测。
2. throughput 最好只统计 generation 阶段，不把 `reward.py` 的 symbolic checking 时间算进去。
3. 样本太少时，模型加载时间会污染 throughput；scouting 建议至少 `subset-size=64/128`。
4. 不要只盯着 `threshold`，更该看 `avg_realized_tokens_per_round`。
