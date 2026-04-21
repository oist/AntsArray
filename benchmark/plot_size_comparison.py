#!/usr/bin/env python3
"""Visualize tag size comparison results across dictionaries.

Reads the CSV from run_size_comparison.py and produces:
1. Grouped bar chart: detection rate by tag size x dictionary
2. Per-camera scatter: detection rate by dictionary, colored by tag size
3. Improvement heatmap: d50 vs d1000 per camera

Usage:
    python benchmark/plot_size_comparison.py \
        --csv benchmark/results/size_comparison_20260409_154620.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_all(csv_path: str, output_dir: str):
    df = pd.read_csv(csv_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sizes = ["1.5mm", "2.0mm", "2.5mm"]
    dicts = df["dict"].unique().tolist()
    size_colors = {"1.5mm": "#e74c3c", "2.0mm": "#f39c12", "2.5mm": "#27ae60"}
    dict_colors = {"4x4_1000": "#3498db", "4x4_250": "#e67e22", "4x4_50": "#2ecc71"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Panel 1: Grouped bar chart (size x dict) ---
    ax = axes[0]
    x = np.arange(len(sizes))
    width = 0.22
    offsets = np.linspace(-width, width, len(dicts))

    for i, dn in enumerate(dicts):
        means = []
        stds = []
        for sz in sizes:
            sub = df[(df["dict"] == dn) & (df["size"] == sz)]
            means.append(sub["det_rate"].mean())
            stds.append(sub["det_rate"].std())
        bars = ax.bar(x + offsets[i], means, width * 0.9, yerr=stds,
                      label=dn, color=dict_colors.get(dn, f"C{i}"),
                      capsize=3, alpha=0.85, edgecolor="white", linewidth=0.5)
        # Value labels on bars
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{m:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=12)
    ax.set_ylabel("Detection Rate (%)", fontsize=12)
    ax.set_title("Detection Rate by Tag Size & Dictionary", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(82, 115)
    ax.axhline(100, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 2: Per-camera dot plot ---
    ax = axes[1]
    cams = sorted(df["cam"].unique())
    y_pos = {cam: i for i, cam in enumerate(cams)}

    for dn in dicts:
        sub = df[df["dict"] == dn]
        for _, row in sub.iterrows():
            y = y_pos[row["cam"]]
            marker = {"4x4_1000": "s", "4x4_250": "D", "4x4_50": "o"}.get(dn, "o")
            ax.scatter(row["det_rate"], y,
                       color=size_colors.get(row["size"], "gray"),
                       marker=marker, s=50, alpha=0.8,
                       edgecolors="black", linewidths=0.3)

    ax.set_yticks(range(len(cams)))
    ax.set_yticklabels(cams, fontsize=9)
    ax.set_xlabel("Detection Rate (%)", fontsize=12)
    ax.set_title("Per-Camera Detection Rate", fontsize=13, fontweight="bold")
    ax.axvline(100, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for sz, c in size_colors.items():
        legend_elements.append(Line2D([0], [0], marker="o", color="w",
                                       markerfacecolor=c, markersize=8, label=sz))
    legend_elements.append(Line2D([0], [0], marker="s", color="w",
                                   markerfacecolor="gray", markersize=8, label="d1000"))
    legend_elements.append(Line2D([0], [0], marker="D", color="w",
                                   markerfacecolor="gray", markersize=8, label="d250"))
    legend_elements.append(Line2D([0], [0], marker="o", color="w",
                                   markerfacecolor="gray", markersize=8, label="d50"))
    ax.legend(handles=legend_elements, fontsize=8, loc="lower left")

    # --- Panel 3: Improvement d50 vs d1000 ---
    ax = axes[2]
    d1000 = df[df["dict"] == "4x4_1000"].set_index("cam")
    d50 = df[df["dict"] == "4x4_50"].set_index("cam")

    common_cams = sorted(set(d1000.index) & set(d50.index))
    improvements = []
    cam_labels = []
    bar_colors = []

    for cam in common_cams:
        r1000 = d1000.loc[cam, "det_rate"]
        r50 = d50.loc[cam, "det_rate"]
        imp = r50 - r1000
        improvements.append(imp)
        cam_labels.append(cam)
        bar_colors.append(size_colors.get(d1000.loc[cam, "size"], "gray"))

    y = np.arange(len(cam_labels))
    bars = ax.barh(y, improvements, color=bar_colors, alpha=0.85,
                   edgecolor="white", linewidth=0.5)

    # Value labels
    for bar, imp in zip(bars, improvements):
        x_pos = bar.get_width() + 0.2 if imp >= 0 else bar.get_width() - 0.2
        ha = "left" if imp >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{imp:+.1f}pp", ha=ha, va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(cam_labels, fontsize=9)
    ax.set_xlabel("Improvement (pp)", fontsize=12)
    ax.set_title("d50 vs d1000 Improvement", fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plot_path = out / "size_comparison_visualization.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {plot_path}")
    plt.close()

    # --- Additional figure: FP analysis (rates > 100%) ---
    fp_rows = df[df["det_rate"] > 100]
    if not fp_rows.empty:
        print(f"\nNote: {len(fp_rows)} entries have det_rate > 100% (FP from other IDs):")
        for _, row in fp_rows.iterrows():
            print(f"  {row['cam']} {row['dict']}: {row['det_rate']:.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output-dir", default="benchmark/results")
    args = p.parse_args()
    plot_all(args.csv, args.output_dir)


if __name__ == "__main__":
    main()
