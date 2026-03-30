#!/usr/bin/env python3
"""Plot throughput/accuracy trade-off charts from a TraDo ablation summary CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def is_block_size_sweep(sweep_name: str) -> bool:
    return sweep_name == "block_size" or sweep_name.startswith("block_size_")


def find_baseline(sub: pd.DataFrame, sweep_name: str) -> pd.Series | None:
    if sub.empty:
        return None
    if is_block_size_sweep(sweep_name):
        mask = sub["block_size"] == 4
    elif sweep_name == "parallel_threshold":
        mask = (sub["block_size"] == 4) & (sub["dynamic_threshold"].round(6) == round(0.90, 6))
    else:
        return None
    if mask.any():
        return sub.loc[mask].iloc[0]
    return sub.iloc[0]


def pareto_frontier(sub: pd.DataFrame, throughput_metric: str) -> pd.DataFrame:
    if sub.empty:
        return sub
    work = sub[[throughput_metric, "accuracy", "sweep_label"]].dropna().sort_values(
        [throughput_metric, "accuracy"], ascending=[False, False]
    )
    keep_rows = []
    best_acc = float("-inf")
    for _, row in work.iterrows():
        acc = float(row["accuracy"])
        if acc > best_acc:
            keep_rows.append(row)
            best_acc = acc
    if not keep_rows:
        return work.iloc[0:0]
    frontier = pd.DataFrame(keep_rows)
    return frontier.sort_values(throughput_metric)


def save_line_chart(sub: pd.DataFrame, dataset: str, sweep_name: str, throughput_metric: str, output_dir: Path) -> Path:
    sub = sub.sort_values("sweep_order")
    plt.figure(figsize=(10, 5))
    ax1 = plt.gca()
    ax1.plot(sub["sweep_label"], sub["accuracy"], marker="o")
    ax1.set_xlabel(sweep_name)
    ax1.set_ylabel("accuracy")
    ax1.set_title(f"{dataset} | {sweep_name}: accuracy vs {throughput_metric}")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(sub["sweep_label"], sub[throughput_metric], marker="s")
    ax2.set_ylabel(throughput_metric)

    plt.xticks(rotation=30)
    plt.tight_layout()
    path = output_dir / f"{dataset}__{sweep_name}__line.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def save_tradeoff_scatter(sub: pd.DataFrame, dataset: str, sweep_name: str, throughput_metric: str, output_dir: Path) -> Path:
    sub = sub.dropna(subset=[throughput_metric, "accuracy"]).sort_values("sweep_order")
    plt.figure(figsize=(7, 5.5))
    ax = plt.gca()
    ax.plot(sub[throughput_metric], sub["accuracy"], marker="o")
    for _, row in sub.iterrows():
        ax.annotate(str(row["sweep_label"]), (row[throughput_metric], row["accuracy"]), fontsize=8)
    ax.set_xlabel(throughput_metric)
    ax.set_ylabel("accuracy")
    ax.set_title(f"{dataset} | {sweep_name}: trade-off")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / f"{dataset}__{sweep_name}__tradeoff.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def save_realized_parallelism_scatter(sub: pd.DataFrame, dataset: str, sweep_name: str, output_dir: Path) -> Path | None:
    if "avg_realized_tokens_per_round" not in sub.columns:
        return None
    work = sub.dropna(subset=["avg_realized_tokens_per_round", "accuracy"]).sort_values("sweep_order")
    if work.empty:
        return None
    plt.figure(figsize=(7, 5.5))
    ax = plt.gca()
    ax.plot(work["avg_realized_tokens_per_round"], work["accuracy"], marker="o")
    for _, row in work.iterrows():
        ax.annotate(str(row["sweep_label"]), (row["avg_realized_tokens_per_round"], row["accuracy"]), fontsize=8)
    ax.set_xlabel("avg_realized_tokens_per_round")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{dataset} | {sweep_name}: realized parallelism vs accuracy")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / f"{dataset}__{sweep_name}__realized_parallelism.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def write_report(df: pd.DataFrame, throughput_metric: str, output_dir: Path) -> Path:
    lines: list[str] = []
    lines.append("# TraDo ablation report")
    lines.append("")
    lines.append(f"throughput metric: `{throughput_metric}`")
    lines.append("")

    for dataset in sorted(df["dataset"].dropna().unique()):
        dataset_df = df[df["dataset"] == dataset]
        lines.append(f"## {dataset}")
        lines.append("")
        for sweep_name in sorted(dataset_df["sweep_name"].dropna().unique()):
            sub = dataset_df[dataset_df["sweep_name"] == sweep_name].copy()
            sub = sub.sort_values("sweep_order")
            if sub.empty:
                continue
            baseline = find_baseline(sub, sweep_name)
            best_acc = sub.sort_values("accuracy", ascending=False).iloc[0]
            best_tp = sub.sort_values(throughput_metric, ascending=False).iloc[0]
            frontier = pareto_frontier(sub, throughput_metric)

            lines.append(f"### {sweep_name}")
            lines.append("")
            if baseline is not None:
                lines.append(
                    f"- baseline: `{baseline['sweep_label']}` | accuracy={baseline['accuracy']:.4f} | "
                    f"{throughput_metric}={baseline[throughput_metric]:.4f}"
                )
            lines.append(
                f"- best accuracy: `{best_acc['sweep_label']}` | accuracy={best_acc['accuracy']:.4f} | "
                f"{throughput_metric}={best_acc[throughput_metric]:.4f}"
            )
            lines.append(
                f"- best throughput: `{best_tp['sweep_label']}` | accuracy={best_tp['accuracy']:.4f} | "
                f"{throughput_metric}={best_tp[throughput_metric]:.4f}"
            )
            if baseline is not None:
                lines.append(
                    f"- best-throughput delta vs baseline: Δacc={best_tp['accuracy'] - baseline['accuracy']:+.4f}, "
                    f"Δ{throughput_metric}={best_tp[throughput_metric] - baseline[throughput_metric]:+.4f}"
                )
            if not frontier.empty:
                frontier_labels = ", ".join(frontier["sweep_label"].tolist())
                lines.append(f"- pareto frontier: {frontier_labels}")
            lines.append("")
        lines.append("")

    report_path = output_dir / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TraDo throughput/accuracy trade-offs.")
    parser.add_argument("--summary-csv", required=True, help="Path to summary.csv from run_trado_ablations.py")
    parser.add_argument("--output-dir", required=True, help="Directory for plots and report")
    parser.add_argument(
        "--throughput-metric",
        default="output_tokens_per_sec",
        choices=["output_tokens_per_sec", "samples_per_sec", "avg_realized_tokens_per_round"],
    )
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)
    if df.empty:
        raise ValueError(f"{summary_csv} 为空")

    required = {"dataset", "sweep_name", "sweep_label", "sweep_order", "accuracy", args.throughput_metric}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"summary.csv 缺少这些列: {missing}")

    for dataset in sorted(df["dataset"].dropna().unique()):
        dataset_df = df[df["dataset"] == dataset]
        for sweep_name in sorted(dataset_df["sweep_name"].dropna().unique()):
            sub = dataset_df[dataset_df["sweep_name"] == sweep_name].copy()
            if sub.empty:
                continue
            save_line_chart(sub, dataset, sweep_name, args.throughput_metric, output_dir)
            save_tradeoff_scatter(sub, dataset, sweep_name, args.throughput_metric, output_dir)
            save_realized_parallelism_scatter(sub, dataset, sweep_name, output_dir)

    write_report(df, args.throughput_metric, output_dir)


if __name__ == "__main__":
    main()
