#!/usr/bin/env python3
"""Run block-size and confidence-threshold ablations for TraDo-8B-Thinking.

This wrapper stays close to the official dLLM-RL evaluation path:
  sample/trado_sample.py -> reward/reward.py

For each sweep point it:
  1. writes a dedicated YAML config,
  2. runs the official sampler,
  3. runs the official reward script,
  4. parses the output JSON,
  5. aggregates throughput / accuracy / realized-parallelism metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, Iterable, Optional

import yaml


OFFICIAL_DATASETS = {
    "PrimeIntellect",
    "MATH_train",
    "demon_openr1math",
    "MATH500",
    "GSM8K",
    "AIME2024",
    "LiveBench",
    "LiveCodeBench",
    "MBPP",
    "HumanEval",
}


@dataclass
class SweepPoint:
    dataset: str
    sweep_name: str
    sweep_label: str
    sweep_order: int
    block_size: int
    denoising_steps_per_block: int
    remasking_strategy: str
    dynamic_threshold: float


@dataclass(frozen=True)
class PreparedTask:
    task_index: int
    requested_dataset: str
    effective_dataset: str
    run_name: str
    project_name: str
    run_dir: Path
    metrics_path: Path
    config_path: Path
    point: SweepPoint


def parse_int_list(raw: str) -> list[int]:
    pieces = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if not pieces:
        raise argparse.ArgumentTypeError("empty integer list")
    try:
        return [int(piece) for piece in pieces]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer list: {raw}") from exc


def parse_float_list(raw: str) -> list[float]:
    pieces = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if not pieces:
        raise argparse.ArgumentTypeError("empty float list")
    try:
        return [float(piece) for piece in pieces]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float list: {raw}") from exc


def format_threshold_value(value: float) -> str:
    return f"{value:.2f}"


def slugify(text: str) -> str:
    safe: list[str] = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe.append(ch)
        elif ch in {"/", " ", ":", "=", ","}:
            safe.append("-")
    slug = "".join(safe).strip("-._")
    return slug or "run"


def ensure_repo_layout(repo_root: Path) -> None:
    required = [
        repo_root / "sample" / "trado_sample.py",
        repo_root / "reward" / "reward.py",
        repo_root / "configs",
        repo_root / "data",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "repo-root 看起来不是 dLLM-RL 仓库，缺少这些路径:\n- " + "\n- ".join(missing)
        )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def safe_mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values]
    if not vals:
        return None
    return sum(vals) / len(vals)


def run_subprocess(
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    env: Optional[dict[str, str]] = None,
) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    merged_env.setdefault("PYTHONUNBUFFERED", "1")
    if env:
        merged_env.update(env)

    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("$ " + " ".join(cmd) + "\n\n")
        log_handle.flush()
        subprocess.run(
            cmd,
            cwd=str(cwd),
            env=merged_env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return time.perf_counter() - start


def maybe_download_dataset(repo_root: Path, dataset_name: str, python_bin: str) -> None:
    data_path = repo_root / "data" / f"{dataset_name}.json"
    if data_path.exists():
        return
    if dataset_name not in OFFICIAL_DATASETS:
        raise FileNotFoundError(
            f"找不到 {data_path}，而且 {dataset_name} 也不是官方下载脚本支持的数据集。"
        )
    cmd = [python_bin, "download_data.py", "--dataset", dataset_name]
    subprocess.run(cmd, cwd=str(repo_root / "data"), check=True)


def maybe_make_subset(repo_root: Path, dataset_name: str, subset_size: Optional[int], subset_seed: int) -> str:
    if subset_size is None:
        return dataset_name

    src_path = repo_root / "data" / f"{dataset_name}.json"
    payload = load_json(src_path)
    if not isinstance(payload, list):
        raise ValueError(f"{src_path} 不是 JSON list，不能自动做 subset。")
    if subset_size <= 0:
        raise ValueError("subset-size 必须 > 0")
    if subset_size >= len(payload):
        return dataset_name

    subset_name = f"{dataset_name}_subset{subset_size}_seed{subset_seed}"
    dst_path = repo_root / "data" / f"{subset_name}.json"
    if dst_path.exists():
        return subset_name

    rng = random.Random(subset_seed)
    indices = sorted(rng.sample(range(len(payload)), subset_size))
    subset = [payload[i] for i in indices]
    dump_json(dst_path, subset)
    return subset_name


def resolve_block_sweep_thresholds(args: argparse.Namespace) -> list[float]:
    if args.block_sweep_thresholds is not None:
        return args.block_sweep_thresholds
    return [args.block_sweep_threshold]


def build_points(args: argparse.Namespace, dataset_name: str) -> list[SweepPoint]:
    points: list[SweepPoint] = []
    block_sweep_thresholds = resolve_block_sweep_thresholds(args)
    use_threshold_specific_names = len(block_sweep_thresholds) > 1

    if not args.skip_block_sweep:
        for threshold in block_sweep_thresholds:
            sweep_name = "block_size"
            if use_threshold_specific_names:
                sweep_name = f"block_size_thr-{format_threshold_value(threshold)}"

            for order, block_size in enumerate(args.block_sizes):
                if args.block_sweep_steps_mode == "match-block":
                    steps = block_size
                elif args.block_sweep_steps_mode == "constant":
                    steps = args.block_sweep_constant_steps
                else:
                    raise ValueError(f"unsupported block-sweep-steps-mode: {args.block_sweep_steps_mode}")

                points.append(
                    SweepPoint(
                        dataset=dataset_name,
                        sweep_name=sweep_name,
                        sweep_label=f"bs={block_size}",
                        sweep_order=order,
                        block_size=block_size,
                        denoising_steps_per_block=steps,
                        remasking_strategy=args.remasking_strategy,
                        dynamic_threshold=threshold,
                    )
                )

    if not args.skip_parallel_sweep:
        for order, threshold in enumerate(args.parallel_thresholds):
            points.append(
                SweepPoint(
                    dataset=dataset_name,
                    sweep_name="parallel_threshold",
                    sweep_label=f"thr={threshold:.2f}",
                    sweep_order=order,
                    block_size=args.parallel_base_block_size,
                    denoising_steps_per_block=args.parallel_denoising_steps,
                    remasking_strategy=args.remasking_strategy,
                    dynamic_threshold=threshold,
                )
            )

    return points


def build_config(
    args: argparse.Namespace,
    point: SweepPoint,
    project_name: str,
    effective_dataset_name: str,
) -> dict[str, Any]:
    return {
            "experiment": {
                "project": project_name,
                "num_node": 1,
                "node_index": 0,
            },
            "model": args.model,
            "model_base": "trado",
            "dataset": {
                "eval_dataset": effective_dataset_name,
                "data_type": "math",
            },
            "execute": {
                "num_chunk": args.num_chunk,
            },
            "rollout": {
                "tensor_parallel_size": args.tensor_parallel_size,
                "max_active": args.max_active,
                "num_response_per_task": args.num_response_per_task,
                "temperature": args.temperature,
                "max_token": args.max_token,
                "block_size": point.block_size,
                "denoising_steps_per_block": point.denoising_steps_per_block,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "remasking_strategy": point.remasking_strategy,
                "dynamic_threshold": point.dynamic_threshold,
                "start_with_think": args.start_with_think,
                "output_unmasking_history": args.output_unmasking_history,
                "stop_token_list": args.stop_tokens,
            },
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.ablation_gpus is None:
        return
    if args.tensor_parallel_size != 1:
        raise ValueError(
            "--ablation-gpus 目前要求 --tensor-parallel-size=1，因为每个 ablation point 只绑定一张卡。"
        )
    if any(gpu_id < 0 for gpu_id in args.ablation_gpus):
        raise ValueError("--ablation-gpus 里的 GPU id 必须是非负整数。")
    if len(set(args.ablation_gpus)) != len(args.ablation_gpus):
        raise ValueError("--ablation-gpus 里有重复 GPU id，请去重后再运行。")


def make_task(
    *,
    task_index: int,
    repo_root: Path,
    output_root_rel: Path,
    config_root: Path,
    run_name: str,
    requested_dataset: str,
    effective_dataset: str,
    point: SweepPoint,
) -> PreparedTask:
    run_rel_dir = output_root_rel / point.dataset / point.sweep_name / slugify(point.sweep_label)
    project_name = str(run_rel_dir).replace("\\", "/")
    run_dir = repo_root / run_rel_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    config_path = config_root / f"{point.dataset}__{point.sweep_name}__{slugify(point.sweep_label)}.yaml"
    return PreparedTask(
        task_index=task_index,
        requested_dataset=requested_dataset,
        effective_dataset=effective_dataset,
        run_name=run_name,
        project_name=project_name,
        run_dir=run_dir,
        metrics_path=metrics_path,
        config_path=config_path,
        point=point,
    )


def run_prepared_task(
    args: argparse.Namespace,
    repo_root: Path,
    task: PreparedTask,
    gpu_id: Optional[int],
) -> dict[str, Any]:
    gpu_prefix = f"[gpu={gpu_id}] " if gpu_id is not None else ""

    if task.metrics_path.exists() and not args.overwrite:
        print(f"[skip] {gpu_prefix}{task.point.dataset} | {task.point.sweep_name} | {task.point.sweep_label}")
        return load_json(task.metrics_path)

    config = build_config(
        args,
        task.point,
        project_name=task.project_name,
        effective_dataset_name=task.effective_dataset,
    )
    with task.config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    shutil.copy2(task.config_path, task.run_dir / "config.yaml")

    config_from_sample = os.path.relpath(task.config_path, repo_root / "sample")
    config_from_reward = os.path.relpath(task.config_path, repo_root / "reward")

    sample_cmd = [args.python_bin, "trado_sample.py", f"config={config_from_sample}"]
    reward_cmd = [args.python_bin, "reward.py", f"config={config_from_reward}"]
    task_env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)} if gpu_id is not None else None

    print(f"[run] {gpu_prefix}{task.point.dataset} | {task.point.sweep_name} | {task.point.sweep_label}")
    generation_seconds = run_subprocess(
        sample_cmd,
        cwd=repo_root / "sample",
        log_path=task.run_dir / "logs" / "sample.log",
        env=task_env,
    )
    reward_seconds = run_subprocess(
        reward_cmd,
        cwd=repo_root / "reward",
        log_path=task.run_dir / "logs" / "reward.log",
        env=task_env,
    )

    output_json_path = find_single_output_json(task.run_dir)
    metrics = parse_metrics(output_json_path, generation_seconds=generation_seconds, reward_seconds=reward_seconds)
    row = {
        "requested_dataset": task.requested_dataset,
        "effective_dataset": task.effective_dataset,
        "run_name": task.run_name,
        "project_name": task.project_name,
        **asdict(task.point),
        **metrics,
        "config_path": str(task.config_path),
        "sample_log": str(task.run_dir / "logs" / "sample.log"),
        "reward_log": str(task.run_dir / "logs" / "reward.log"),
    }
    dump_json(task.metrics_path, row)
    return row


def run_tasks(
    args: argparse.Namespace,
    repo_root: Path,
    tasks: list[PreparedTask],
) -> list[dict[str, Any]]:
    if not tasks:
        return []

    rows_by_index: dict[int, dict[str, Any]] = {}

    if args.ablation_gpus is None:
        for task in tasks:
            rows_by_index[task.task_index] = run_prepared_task(args, repo_root, task, gpu_id=None)
        return [rows_by_index[idx] for idx in sorted(rows_by_index)]

    gpu_queue: Queue[int] = Queue()
    for gpu_id in args.ablation_gpus:
        gpu_queue.put(gpu_id)

    def worker(task: PreparedTask) -> tuple[int, dict[str, Any]]:
        if task.metrics_path.exists() and not args.overwrite:
            print(f"[skip] {task.point.dataset} | {task.point.sweep_name} | {task.point.sweep_label}")
            return task.task_index, load_json(task.metrics_path)

        gpu_id = gpu_queue.get()
        try:
            row = run_prepared_task(args, repo_root, task, gpu_id=gpu_id)
            return task.task_index, row
        finally:
            gpu_queue.put(gpu_id)

    with ThreadPoolExecutor(max_workers=len(args.ablation_gpus)) as executor:
        future_to_task = {executor.submit(worker, task): task for task in tasks}
        try:
            for future in as_completed(future_to_task):
                task_index, row = future.result()
                rows_by_index[task_index] = row
        except Exception:
            for future in future_to_task:
                future.cancel()
            raise

    return [rows_by_index[idx] for idx in sorted(rows_by_index)]


def find_single_output_json(project_dir: Path) -> Path:
    temp_dir = project_dir / "temp_data"
    candidates = sorted(temp_dir.glob("outputs-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"没有在 {temp_dir} 下找到 outputs-*.json")
    return candidates[0]


def parse_metrics(output_json_path: Path, generation_seconds: float, reward_seconds: float) -> dict[str, Any]:
    payload = load_json(output_json_path)
    if not isinstance(payload, list):
        raise ValueError(f"{output_json_path} 不是 JSON list")

    num_tasks = len(payload)
    correctness: list[int] = []
    response_lengths: list[int] = []
    decode_rounds: list[int] = []
    realized_tokens_per_round: list[float] = []
    mean_first_unmask_rounds: list[float] = []
    masked_sampled_probs: list[float] = []
    accepted_sampled_probs: list[float] = []
    above_threshold_fractions: list[float] = []
    accepted_fraction_of_masked: list[float] = []
    fallback_round_fractions: list[float] = []
    rounds_per_block: list[float] = []
    accepted_tokens_per_denoise_round: list[float] = []
    confidence_summary_count = 0

    for item in payload:
        corr_list = item.get("correctness", []) or []
        len_list = item.get("response_length", []) or []
        step_collection = item.get("step_map", []) or []
        confidence_collection = item.get("confidence_summary", []) or []

        correctness.extend(int(bool(v)) for v in corr_list)
        response_lengths.extend(int(v) for v in len_list)

        max_count = max(len(len_list), len(step_collection), len(confidence_collection))
        for idx in range(max_count):
            response_len = int(len_list[idx]) if idx < len(len_list) else None
            steps = step_collection[idx] if idx < len(step_collection) else None
            confidence_summary = confidence_collection[idx] if idx < len(confidence_collection) else None
            if not isinstance(steps, list) or not steps:
                numeric_steps = []
            else:
                numeric_steps = [int(step) for step in steps if isinstance(step, (int, float))]
            if numeric_steps:
                rounds = max(numeric_steps) + 1
                decode_rounds.append(rounds)
                if response_len is not None and rounds > 0:
                    realized_tokens_per_round.append(response_len / rounds)
                mean_first_unmask_rounds.append(sum(numeric_steps) / len(numeric_steps))

            if isinstance(confidence_summary, dict):
                confidence_summary_count += 1
                masked_prob = confidence_summary.get("avg_masked_sampled_prob")
                accepted_prob = confidence_summary.get("avg_accepted_sampled_prob")
                above_fraction = confidence_summary.get("avg_above_threshold_fraction")
                accepted_fraction = confidence_summary.get("avg_accepted_fraction_of_masked")
                fallback_fraction = confidence_summary.get("fallback_round_fraction")
                avg_rounds = confidence_summary.get("avg_rounds_per_block")
                accepted_per_round = confidence_summary.get("avg_accepted_tokens_per_denoise_round")
                if isinstance(masked_prob, (int, float)):
                    masked_sampled_probs.append(float(masked_prob))
                if isinstance(accepted_prob, (int, float)):
                    accepted_sampled_probs.append(float(accepted_prob))
                if isinstance(above_fraction, (int, float)):
                    above_threshold_fractions.append(float(above_fraction))
                if isinstance(accepted_fraction, (int, float)):
                    accepted_fraction_of_masked.append(float(accepted_fraction))
                if isinstance(fallback_fraction, (int, float)):
                    fallback_round_fractions.append(float(fallback_fraction))
                if isinstance(avg_rounds, (int, float)):
                    rounds_per_block.append(float(avg_rounds))
                if isinstance(accepted_per_round, (int, float)):
                    accepted_tokens_per_denoise_round.append(float(accepted_per_round))

    total_output_tokens = sum(response_lengths)
    return {
        "num_tasks": num_tasks,
        "num_scored_responses": len(correctness),
        "accuracy": safe_mean(correctness),
        "total_output_tokens": total_output_tokens,
        "avg_output_tokens": safe_mean(response_lengths),
        "generation_wall_clock_sec": generation_seconds,
        "reward_wall_clock_sec": reward_seconds,
        "samples_per_sec": (num_tasks / generation_seconds) if generation_seconds > 0 else None,
        "output_tokens_per_sec": (total_output_tokens / generation_seconds) if generation_seconds > 0 else None,
        "avg_decode_rounds": safe_mean(decode_rounds),
        "avg_realized_tokens_per_round": safe_mean(realized_tokens_per_round),
        "avg_first_unmask_round": safe_mean(mean_first_unmask_rounds),
        "step_map_coverage": (len(decode_rounds) / len(response_lengths)) if response_lengths else 0.0,
        "confidence_summary_coverage": (confidence_summary_count / len(response_lengths)) if response_lengths else 0.0,
        "avg_masked_sampled_prob": safe_mean(masked_sampled_probs),
        "avg_accepted_sampled_prob": safe_mean(accepted_sampled_probs),
        "avg_above_threshold_fraction": safe_mean(above_threshold_fractions),
        "avg_accepted_fraction_of_masked": safe_mean(accepted_fraction_of_masked),
        "avg_fallback_round_fraction": safe_mean(fallback_round_fractions),
        "avg_rounds_per_block": safe_mean(rounds_per_block),
        "avg_accepted_tokens_per_denoise_round": safe_mean(accepted_tokens_per_denoise_round),
        "output_json": str(output_json_path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def maybe_copy_plotter(wrapper_root: Path, output_root: Path) -> None:
    src = wrapper_root / "plot_trado_tradeoff.py"
    dst = output_root / "plot_trado_tradeoff.py"
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)


def run_plotter(python_bin: str, wrapper_root: Path, summary_csv: Path, plots_dir: Path, throughput_metric: str) -> None:
    cmd = [
        python_bin,
        str(wrapper_root / "plot_trado_tradeoff.py"),
        "--summary-csv",
        str(summary_csv),
        "--output-dir",
        str(plots_dir),
        "--throughput-metric",
        throughput_metric,
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TraDo block-size and parallel-threshold sweeps.")
    parser.add_argument("--repo-root", required=True, help="Local path to the dLLM-RL repo root.")
    parser.add_argument("--model", required=True, help="Local model path or HF model id.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable used to run official scripts.")
    parser.add_argument("--datasets", nargs="+", default=["GSM8K", "MATH500"], help="Datasets to run.")
    parser.add_argument("--subset-size", type=int, default=None, help="Optional subset size for quick sweeps.")
    parser.add_argument("--subset-seed", type=int, default=0, help="Sampling seed for subset creation.")
    parser.add_argument("--run-name", default=None, help="Optional explicit run name.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Repo-relative or repo-contained path for outputs. Default: runs/trado_ablation/<run-name>",
    )

    parser.add_argument(
        "--block-sizes",
        type=parse_int_list,
        default=parse_int_list("2,4,8,16"),
        help="Comma-separated block sizes. Default: 2,4,8,16",
    )
    parser.add_argument(
        "--block-sweep-steps-mode",
        choices=["match-block", "constant"],
        default="match-block",
        help="How to set denoising_steps_per_block in block sweep.",
    )
    parser.add_argument(
        "--block-sweep-constant-steps",
        type=int,
        default=4,
        help="Used when --block-sweep-steps-mode constant.",
    )
    parser.add_argument(
        "--block-sweep-threshold",
        type=float,
        default=0.90,
        help="Fixed dynamic_threshold used during block-size sweep.",
    )
    parser.add_argument(
        "--block-sweep-thresholds",
        type=parse_float_list,
        default=None,
        help=(
            "Optional comma-separated dynamic_threshold values for block-size sweep. "
            "When multiple values are provided, each threshold gets its own block-size sweep family."
        ),
    )
    parser.add_argument(
        "--parallel-thresholds",
        type=parse_float_list,
        default=parse_float_list("0.70,0.80,0.85,0.90,0.95,0.98"),
        help="Comma-separated dynamic_threshold sweep values.",
    )
    parser.add_argument(
        "--parallel-base-block-size",
        type=int,
        default=4,
        help="Fixed block_size used during parallel-threshold sweep.",
    )
    parser.add_argument(
        "--parallel-denoising-steps",
        type=int,
        default=4,
        help="Fixed denoising_steps_per_block used during parallel-threshold sweep.",
    )
    parser.add_argument("--skip-block-sweep", action="store_true", help="Do not run block-size sweep.")
    parser.add_argument("--skip-parallel-sweep", action="store_true", help="Do not run parallel-threshold sweep.")

    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--ablation-gpus",
        type=parse_int_list,
        default=None,
        help="Comma-separated GPU ids for parallel ablation scheduling. Each ablation point gets one GPU.",
    )
    parser.add_argument("--max-active", type=int, default=2)
    parser.add_argument("--num-response-per-task", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-token", type=int, default=10000)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument(
        "--remasking-strategy",
        default="low_confidence_dynamic",
        help="Sampling strategy. Default: low_confidence_dynamic",
    )
    parser.add_argument("--start-with-think", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-unmasking-history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-tokens", nargs="*", default=["<|im_end|>"])
    parser.add_argument("--num-chunk", type=int, default=128)

    parser.add_argument("--overwrite", action="store_true", help="Re-run sweep points even if metrics.json already exists.")
    parser.add_argument("--no-auto-download", action="store_true", help="Disable automatic data download.")
    parser.add_argument("--skip-plotting", action="store_true", help="Do not run the plot script at the end.")
    parser.add_argument(
        "--plot-throughput-metric",
        default="output_tokens_per_sec",
        choices=["output_tokens_per_sec", "samples_per_sec", "avg_realized_tokens_per_round"],
        help="X-axis used by the plotting script.",
    )

    args = parser.parse_args()
    validate_args(args)

    if args.skip_block_sweep and args.skip_parallel_sweep:
        raise ValueError("block sweep 和 parallel sweep 不能同时跳过。")

    repo_root = Path(args.repo_root).expanduser().resolve()
    ensure_repo_layout(repo_root)
    wrapper_root = Path(__file__).resolve().parent

    run_name = args.run_name or datetime.now().strftime("trado_ablation_%Y%m%d_%H%M%S")
    if args.output_root is None:
        output_root = repo_root / "runs" / "trado_ablation" / run_name
    else:
        requested = Path(args.output_root).expanduser()
        if requested.is_absolute():
            requested = requested.resolve()
            if repo_root not in requested.parents and requested != repo_root:
                raise ValueError("--output-root 必须位于 repo-root 下面，这样官方脚本才能正确写结果。")
            output_root = requested
        else:
            output_root = (repo_root / requested).resolve()

    output_root_rel = output_root.relative_to(repo_root)
    output_root.mkdir(parents=True, exist_ok=True)
    maybe_copy_plotter(wrapper_root, output_root)

    manifest = {
        "repo_root": str(repo_root),
        "model": args.model,
        "run_name": run_name,
        "output_root": str(output_root),
        "ablation_gpus": args.ablation_gpus,
        "argv": sys.argv,
        "datasets": args.datasets,
    }
    dump_json(output_root / "manifest.json", manifest)

    config_root = repo_root / "configs" / "autogen" / "trado_ablation" / run_name
    config_root.mkdir(parents=True, exist_ok=True)

    tasks: list[PreparedTask] = []
    task_index = 0

    for requested_dataset in args.datasets:
        if not args.no_auto_download:
            maybe_download_dataset(repo_root, requested_dataset, args.python_bin)
        else:
            data_path = repo_root / "data" / f"{requested_dataset}.json"
            if not data_path.exists():
                raise FileNotFoundError(f"数据不存在：{data_path}")

        effective_dataset = maybe_make_subset(repo_root, requested_dataset, args.subset_size, args.subset_seed)
        points = build_points(args, requested_dataset)

        for point in points:
            tasks.append(
                make_task(
                    task_index=task_index,
                    repo_root=repo_root,
                    output_root_rel=output_root_rel,
                    config_root=config_root,
                    run_name=run_name,
                    requested_dataset=requested_dataset,
                    effective_dataset=effective_dataset,
                    point=point,
                )
            )
            task_index += 1

    if args.ablation_gpus is None:
        print(f"Scheduling {len(tasks)} ablation points sequentially.")
    else:
        gpu_label = ",".join(str(gpu_id) for gpu_id in args.ablation_gpus)
        print(
            f"Scheduling {len(tasks)} ablation points across {len(args.ablation_gpus)} GPUs "
            f"(one point per GPU): {gpu_label}"
        )

    all_rows = run_tasks(args, repo_root, tasks)

    summary_csv = output_root / "summary.csv"
    summary_jsonl = output_root / "summary.jsonl"
    write_csv(summary_csv, all_rows)
    write_jsonl(summary_jsonl, all_rows)

    if not args.skip_plotting and all_rows:
        plots_dir = output_root / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        run_plotter(
            python_bin=args.python_bin,
            wrapper_root=wrapper_root,
            summary_csv=summary_csv,
            plots_dir=plots_dir,
            throughput_metric=args.plot_throughput_metric,
        )

    print(f"\nDone. Summary CSV: {summary_csv}")
    print(f"Summary JSONL: {summary_jsonl}")
    if not args.skip_plotting:
        print(f"Plots dir: {output_root / 'plots'}")


if __name__ == "__main__":
    main()
