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


def parse_filename_fields(path, regex):
    """Extract named groups from a basename — caller supplies the regex."""
    m = re.search(regex, os.path.basename(path))
    if not m:
        return {}
    return m.groupdict()


def select_three_snapshots(K):
    """Return three snapshot indices spanning the evolution: 1, K//2, K-1.

    Falls back gracefully when K < 3.
    """
    if K <= 1:
        return [0]
    if K == 2:
        return [0, 1]
    return [1, K // 2, K - 1]


def draw_hist_panel(ax, hist_dict, color, label, alpha=0.65):
    """Render a log-binned histogram block (as stored in the JSON)."""
    edges  = np.array(hist_dict.get("edges",  []), dtype=np.float64)
    counts = np.array(hist_dict.get("counts", []), dtype=np.float64)
    if edges.size < 2 or counts.size == 0:
        return False
    centers = np.sqrt(edges[:-1] * edges[1:])
    ax.step(centers, counts, where="mid",
            color=color, linewidth=1.4, alpha=alpha, label=label)
    return True


def draw_six_histograms(ax_grid, rec, color_spatial="tab:blue",
                         color_temporal="tab:orange"):
    """Render the 6-panel time-evolution histogram block.

    ``ax_grid`` is a 2x3 ndarray of axes. Row 0 = spatial, row 1 =
    temporal. Columns are T=1, T=T_max/2, T=T_max (snapshot indices
    chosen by ``select_three_snapshots``).

    ``rec`` is one JSON record (dict).
    """
    spatial_hists  = rec["mi_distributions"].get("spatial_per_snap", [])
    temporal_hists = rec["mi_distributions"].get("temporal_per_source", [])
    K = rec["graph"]["n_snapshots"]
    dt = rec["config"]["dt"]
    snap_indices = select_three_snapshots(K)

    titles_top = [f"Spatial MI at t={i * dt:.2f}" for i in snap_indices]
    titles_bot = [f"Temporal MI from t={i * dt:.2f}" for i in snap_indices]

    for c, snap_idx in enumerate(snap_indices):
        if snap_idx < len(spatial_hists):
            block = spatial_hists[snap_idx]
            draw_hist_panel(ax_grid[0, c], block,
                             color_spatial, label="")
            ax_grid[0, c].set_title(titles_top[c])
            ax_grid[0, c].set_xscale("log")
            ax_grid[0, c].set_yscale("log")
            ax_grid[0, c].set_xlabel("MI value (nats)")
            ax_grid[0, c].set_ylabel("count")
            ax_grid[0, c].grid(True, alpha=0.3, which="both")
            med = block.get("median", float("nan"))
            ax_grid[0, c].axvline(med, color="black", ls=":", lw=1)
            ax_grid[0, c].text(med, 1.0, f"  med={med:.1e}",
                                color="black", fontsize=8, rotation=90,
                                va="bottom")

        # Temporal source index t < K-1; if requested snap_idx is the
        # last index, pull the histogram from snap_idx - 1.
        t_src = snap_idx if snap_idx < len(temporal_hists) \
                          else max(0, len(temporal_hists) - 1)
        if t_src < len(temporal_hists):
            block = temporal_hists[t_src]
            draw_hist_panel(ax_grid[1, c], block,
                             color_temporal, label="")
            ax_grid[1, c].set_title(titles_bot[c])
            ax_grid[1, c].set_xscale("log")
            ax_grid[1, c].set_yscale("log")
            ax_grid[1, c].set_xlabel("MI value (nats)")
            ax_grid[1, c].set_ylabel("count")
            ax_grid[1, c].grid(True, alpha=0.3, which="both")
            med = block.get("median", float("nan"))
            ax_grid[1, c].axvline(med, color="black", ls=":", lw=1)
            ax_grid[1, c].text(med, 1.0, f"  med={med:.1e}",
                                color="black", fontsize=8, rotation=90,
                                va="bottom")
