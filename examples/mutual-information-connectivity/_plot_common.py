"""Shared helpers for the H_4D scan plotters."""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np


def load_scan(scan_dir, pattern=r"(.*?)\.json$"):
    """Load every JSON in ``scan_dir`` whose basename matches ``pattern``.

    Returns a list of dicts with the JSON contents plus a ``__path``
    field.
    """
    pat = re.compile(pattern)
    rows = []
    for path in sorted(glob.glob(os.path.join(scan_dir, "*.json"))):
        if not pat.search(os.path.basename(path)):
            continue
        data = json.load(open(path))
        data["__path"] = path
        rows.append(data)
    return rows


def _select_three_regions(K):
    """Return three lists of snapshot indices spanning the evolution.

    The split mirrors the user's framing:
      • Region 0: just snapshot 0 (pre-evolution, T=0).
      • Region 1: snapshots in (0, K/2].
      • Region 2: snapshots in (K/2, K-1].
    """
    if K <= 1:
        return [[0]], "T=0", "(empty)", "(empty)"
    half = K // 2
    reg0 = [0]
    reg1 = list(range(1, half + 1))
    reg2 = list(range(half + 1, K))
    return [reg0, reg1, reg2]


def _aggregate_spatial(by_segment, region_indices):
    """Sum spatial counts across snapshots in the region."""
    spatial_counts = by_segment.get("spatial_per_snap", [])
    if not spatial_counts:
        return None
    width = len(spatial_counts[0])
    out = np.zeros(width, dtype=np.int64)
    for t in region_indices:
        if 0 <= t < len(spatial_counts):
            out += np.asarray(spatial_counts[t], dtype=np.int64)
    return out


def _aggregate_temporal(by_segment, region_indices):
    """Sum temporal counts across (source, destination) pairs that both
    lie in ``region_indices``.
    """
    pairs = by_segment.get("temporal_per_pair", [])
    if not pairs:
        return None
    width = len(pairs[0]["counts"])
    out = np.zeros(width, dtype=np.int64)
    region_set = set(region_indices)
    for p in pairs:
        src = int(p["source"])
        dst = src + int(p["stride"])
        if src in region_set and dst in region_set:
            out += np.asarray(p["counts"], dtype=np.int64)
    return out


def _hist_block_from_counts(edges, counts, kind, t_label, color, ax):
    """Render a count histogram on shared ``edges`` into ``ax``."""
    if counts is None or counts.sum() == 0:
        ax.text(0.5, 0.5, f"(empty {kind} region)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="grey")
        ax.set_title(f"{kind.title()} MI — {t_label}")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3, which="both")
        return

    edges = np.asarray(edges, dtype=np.float64)
    centers = np.sqrt(edges[:-1] * edges[1:])
    ax.step(centers, np.maximum(counts, 1e-12), where="mid",
            color=color, linewidth=1.4, alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("MI value (nats)")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3, which="both")

    # Median estimate from the binned distribution.
    cumsum = np.cumsum(counts.astype(np.float64))
    total = cumsum[-1]
    if total > 0:
        half = total / 2.0
        idx = int(np.searchsorted(cumsum, half))
        idx = min(idx, len(centers) - 1)
        median_est = centers[idx]
        ax.axvline(median_est, color="black", ls=":", lw=1)
        ax.text(median_est, 1.0, f"  med~{median_est:.1e}",
                color="black", fontsize=8, rotation=90, va="bottom")
    ax.set_title(f"{kind.title()} MI — {t_label}  (n={int(counts.sum())})")


def draw_six_histograms(ax_grid, rec, color_spatial="tab:blue",
                         color_temporal="tab:orange"):
    """Render the 6-panel time-region histogram block.

    Columns are the three time regions:
      • Col 0: T = 0 (snapshot 0 only — initial post-quench state).
      • Col 1: T in (0, T_max/2] (early evolution).
      • Col 2: T in (T_max/2, T_max] (late evolution).

    Row 0 is spatial MI (within-snapshot bond pairs that fall in the
    region). Row 1 is temporal MI (snapshot pairs where BOTH endpoints
    lie in the region).
    """
    K  = rec["graph"]["n_snapshots"]
    dt = rec["config"]["dt"]
    T_max = rec["config"]["T"]
    regions = _select_three_regions(K)

    region_labels = [
        f"T = 0 (snapshot 0)",
        f"T in (0, {T_max/2:.2f}]  ({len(regions[1])} snaps)",
        f"T in ({T_max/2:.2f}, {T_max:.2f}]  ({len(regions[2])} snaps)",
    ]

    by_segment = rec.get("mi_distributions", {}).get("by_segment")
    if not by_segment:
        for c in range(3):
            ax_grid[0, c].text(0.5, 0.5, "no by_segment data",
                                ha="center", va="center",
                                transform=ax_grid[0, c].transAxes,
                                fontsize=10, color="grey")
            ax_grid[1, c].text(0.5, 0.5, "no by_segment data",
                                ha="center", va="center",
                                transform=ax_grid[1, c].transAxes,
                                fontsize=10, color="grey")
        return

    edges = by_segment["edges"]
    for c, (reg_idx, label) in enumerate(zip(regions, region_labels)):
        sp = _aggregate_spatial(by_segment, reg_idx)
        tm = _aggregate_temporal(by_segment, reg_idx)
        _hist_block_from_counts(edges, sp, "spatial", label,
                                  color_spatial, ax_grid[0, c])
        _hist_block_from_counts(edges, tm, "temporal", label,
                                  color_temporal, ax_grid[1, c])
