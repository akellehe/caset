#!/usr/bin/env python3
# MIT License -- Copyright (c) 2025 Andrew Kelleher
"""
Compare two benchmark JSON logs and produce before/after charts.

Usage:
    python examples/benchmarks/compare_benchmarks.py \\
        --before docs/source/assets/benchmarks/benchmark_results.json \\
        --after  /tmp/bench_after/benchmark_results.json \\
        --save   docs/source/assets/benchmarks/
"""
import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


DIMENSIONS = [2, 3, 4]  # skip 1D (trivial sizes)
DIMENSION_COLORS = {2: "#2171b5", 3: "#238b45", 4: "#cb181d"}
DIMENSION_MARKERS = {2: "o", 3: "D", 4: "^"}


def load(path):
    with open(path) as f:
        data = json.load(f)
    return data["results"], data.get("metadata", {})


def plot_time_comparison(before, after, ax):
    """Side-by-side build time comparison."""
    for dim in DIMENSIONS:
        b = [r for r in before if r["dimension"] == dim]
        a = [r for r in after if r["dimension"] == dim]
        if not b or not a:
            continue
        x = [r["target_n"] for r in b]
        yb = [r["time_mean_s"] for r in b]
        ya = [r["time_mean_s"] for r in a]
        color = DIMENSION_COLORS[dim]
        ax.plot(x, yb, f"{DIMENSION_MARKERS[dim]}--", color=color,
                alpha=0.45, linewidth=1.5, markersize=6,
                label=f"{dim}D before")
        ax.plot(x, ya, f"{DIMENSION_MARKERS[dim]}-", color=color,
                linewidth=2.5, markersize=7,
                label=f"{dim}D after")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Target simplices", fontsize=12)
    ax.set_ylabel("Build time (seconds)", fontsize=12)
    ax.set_title("Build Time: Before vs. After", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{int(v):,}"))


def plot_speedup(before, after, ax):
    """Speedup ratio (before / after) per dimension and size."""
    for dim in DIMENSIONS:
        b = {r["target_n"]: r for r in before if r["dimension"] == dim}
        a = {r["target_n"]: r for r in after if r["dimension"] == dim}
        common = sorted(set(b.keys()) & set(a.keys()))
        if not common:
            continue
        x = common
        speedups = [b[n]["time_mean_s"] / a[n]["time_mean_s"] for n in common]
        ax.plot(x, speedups, f"{DIMENSION_MARKERS[dim]}-",
                color=DIMENSION_COLORS[dim], linewidth=2, markersize=7,
                label=f"{dim}D")

    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Target simplices", fontsize=12)
    ax.set_ylabel("Speedup (before / after)", fontsize=12)
    ax.set_title("Speedup by Dimension", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.25)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{int(v):,}"))


def plot_throughput_comparison(before, after, ax):
    """Grouped bar chart of throughput before/after."""
    sizes = sorted({r["target_n"] for r in before
                    if r["dimension"] in DIMENSIONS})
    n_dims = len(DIMENSIONS)
    bar_width = 0.35
    group_width = n_dims * bar_width * 2 + 0.3

    for gi, size in enumerate(sizes):
        base_x = gi * group_width
        for di, dim in enumerate(DIMENSIONS):
            b = next((r for r in before
                      if r["dimension"] == dim and r["target_n"] == size),
                     None)
            a = next((r for r in after
                      if r["dimension"] == dim and r["target_n"] == size),
                     None)
            if not b or not a:
                continue
            x_before = base_x + di * (bar_width * 2 + 0.05)
            x_after = x_before + bar_width
            color = DIMENSION_COLORS[dim]
            ax.bar(x_before, b["simplices_per_sec"], bar_width,
                   color=color, alpha=0.4, edgecolor=color, linewidth=0.8)
            ax.bar(x_after, a["simplices_per_sec"], bar_width,
                   color=color, alpha=0.9, edgecolor="white", linewidth=0.5)

    # Legend entries
    ax.bar([], [], bar_width, color="gray", alpha=0.4, label="Before")
    ax.bar([], [], bar_width, color="gray", alpha=0.9, label="After")
    for dim in DIMENSIONS:
        ax.bar([], [], bar_width, color=DIMENSION_COLORS[dim],
               label=f"{dim}D")

    tick_positions = [gi * group_width + (n_dims - 1) * bar_width
                      for gi in range(len(sizes))]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{s:,}" for s in sizes], fontsize=9)
    ax.set_xlabel("Target simplices", fontsize=12)
    ax.set_ylabel("Simplices / second", fontsize=12)
    ax.set_title("Build Throughput: Before vs. After", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, axis="y", alpha=0.25)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{int(v):,}"))


def plot_summary_table(before, after, ax):
    """Table showing percentage improvement."""
    ax.axis("off")
    rows = []
    for dim in DIMENSIONS:
        b_map = {r["target_n"]: r for r in before if r["dimension"] == dim}
        a_map = {r["target_n"]: r for r in after if r["dimension"] == dim}
        for size in sorted(set(b_map.keys()) & set(a_map.keys())):
            bt = b_map[size]["time_mean_s"]
            at = a_map[size]["time_mean_s"]
            pct = (bt - at) / bt * 100
            rows.append([
                f"{dim}D",
                f"{size:,}",
                f"{bt:.4f}",
                f"{at:.4f}",
                f"{pct:+.1f}%",
            ])

    table = ax.table(
        cellText=rows,
        colLabels=["Dim", "Target", "Before (s)", "After (s)", "Change"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)

    # Color the change column
    for i, row in enumerate(rows):
        pct_val = float(row[4].rstrip("%"))
        cell = table[i + 1, 4]
        if pct_val > 0:
            cell.set_facecolor("#d4edda")
        elif pct_val < 0:
            cell.set_facecolor("#f8d7da")

    ax.set_title("Build Time Improvement Summary", fontsize=13,
                 fontweight="bold", pad=20)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two benchmark runs")
    parser.add_argument("--before", required=True,
                        help="Path to baseline benchmark_results.json")
    parser.add_argument("--after", required=True,
                        help="Path to new benchmark_results.json")
    parser.add_argument("--save", type=str, default=None,
                        help="Directory to save comparison plots")
    args = parser.parse_args()

    before, meta_before = load(args.before)
    after, meta_after = load(args.after)

    print("=" * 64)
    print("  Benchmark Comparison")
    print(f"  Before: {args.before}")
    print(f"    timestamp: {meta_before.get('timestamp', 'unknown')}")
    print(f"  After:  {args.after}")
    print(f"    timestamp: {meta_after.get('timestamp', 'unknown')}")
    print("=" * 64)

    # Print summary
    print(f"\n{'Dim':>4} {'Target':>10} {'Before (s)':>12} "
          f"{'After (s)':>12} {'Change':>10}")
    print("-" * 56)
    for dim in DIMENSIONS:
        b_map = {r["target_n"]: r for r in before if r["dimension"] == dim}
        a_map = {r["target_n"]: r for r in after if r["dimension"] == dim}
        for size in sorted(set(b_map.keys()) & set(a_map.keys())):
            bt = b_map[size]["time_mean_s"]
            at = a_map[size]["time_mean_s"]
            pct = (bt - at) / bt * 100
            print(f"{dim:>3}D {size:>10,} {bt:>12.4f} "
                  f"{at:>12.4f} {pct:>+9.1f}%")

    # --- 4-panel comparison figure ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.suptitle("caset Build Benchmark Comparison",
                 fontsize=16, fontweight="bold", y=0.98)

    plot_time_comparison(before, after, axes[0, 0])
    plot_speedup(before, after, axes[0, 1])
    plot_throughput_comparison(before, after, axes[1, 0])
    plot_summary_table(before, after, axes[1, 1])

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if args.save:
        path = os.path.join(args.save, "benchmark_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\nSaved {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
