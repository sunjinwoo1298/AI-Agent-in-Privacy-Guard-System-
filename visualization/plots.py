#!/usr/bin/env python3
"""Plotting utilities for experiment summaries."""

import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_metric(df: pd.DataFrame, metric: str, out_dir: str = "results/plots"):
    os.makedirs(out_dir, exist_ok=True)
    x = df["n_agents"]
    y_mean = df[f"{metric}_mean"]
    y_std = df[f"{metric}_std"]

    plt.figure(figsize=(6, 4))
    plt.plot(x, y_mean, marker="o", label="mean")
    plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2, label="std")
    plt.xticks(x)
    plt.xlabel("n_agents")
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend()
    out_path = os.path.join(out_dir, f"{metric}_vs_n.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_all(summary_csv: str = "results/summary.csv", out_dir: str = "results/plots"):
    if not os.path.exists(summary_csv):
        raise FileNotFoundError(summary_csv)
    df = pd.read_csv(summary_csv)
    for metric in ["total_latency", "coordination_tax", "sync_delay", "f1", "efficiency", "total_throughput"]:
        if f"{metric}_mean" in df.columns:
            plot_metric(df, metric, out_dir=out_dir)
    print("Plots complete.")


if __name__ == "__main__":
    plot_all()


def compare_metric(thread_df: pd.DataFrame, process_df: pd.DataFrame, metric: str, out_dir: str = "results/plots/comparison"):
    os.makedirs(out_dir, exist_ok=True)
    x_t = thread_df["n_agents"]
    y_t = thread_df[f"{metric}_mean"]
    y_t_std = thread_df.get(f"{metric}_std")

    x_p = process_df["n_agents"]
    y_p = process_df[f"{metric}_mean"]
    y_p_std = process_df.get(f"{metric}_std")

    plt.figure(figsize=(6, 4))
    plt.plot(x_t, y_t, marker="o", label="ThreadPool")
    if y_t_std is not None:
        plt.fill_between(x_t, y_t - y_t_std, y_t + y_t_std, alpha=0.15)
    plt.plot(x_p, y_p, marker="s", label="ProcessPool")
    if y_p_std is not None:
        plt.fill_between(x_p, y_p - y_p_std, y_p + y_p_std, alpha=0.15)

    xticks = sorted(set(list(x_t) + list(x_p)))
    plt.xticks(xticks)
    plt.xlabel("n_agents")
    plt.ylabel(metric)
    plt.title(f"{metric} — Thread vs Process")
    plt.grid(True)
    plt.legend()
    out_path = os.path.join(out_dir, f"{metric}_thread_vs_process.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def compare_all(thread_summary_csv: str = "results/summary_heavy_thread.csv", process_summary_csv: str = "results/summary_heavy_process.csv", out_dir: str = "results/plots/comparison"):
    if not os.path.exists(thread_summary_csv):
        raise FileNotFoundError(thread_summary_csv)
    if not os.path.exists(process_summary_csv):
        raise FileNotFoundError(process_summary_csv)
    tdf = pd.read_csv(thread_summary_csv)
    pdf = pd.read_csv(process_summary_csv)
    metrics = ["total_latency", "coordination_tax", "total_throughput", "efficiency"]
    for m in metrics:
        if f"{m}_mean" in tdf.columns and f"{m}_mean" in pdf.columns:
            compare_metric(tdf, pdf, m, out_dir=out_dir)
    print("Comparison plots complete.")
