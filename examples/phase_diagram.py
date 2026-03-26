#!/usr/bin/env python3
# MIT License -- Copyright (c) 2025 Andrew Kelleher
"""
Phase diagram of 4D Causal Dynamical Triangulations.

Reproduces Figure 3 from:
  Ambjorn, Jurkiewicz, Loll, "Reconstructing the Universe",
  Phys. Rev. D 72 (2005) [hep-th/0505154]
and the phase diagram from:
  Gorlich, "Introduction to Causal Dynamical Triangulations" (2013)

Scans the coupling-constant space (k0, Delta) and classifies each point
into one of three phases using order parameters:

  Phase A (branched polymer): large k0.
    Signature: highly irregular volume profile with many thin stalks.
    Order parameter: large variance-to-mean ratio of N_3(tau).

  Phase B (crumpled): small k0, small Delta.
    Signature: volume collapses to 1-2 time slices.
    Order parameter: high concentration (max_slice / total > 0.5).

  Phase C (de Sitter): moderate k0, nonzero Delta.
    Signature: smooth, extended volume profile.
    Order parameter: low concentration, moderate variance.

Parameters scanned:
  k0 in [0.5, 6.0]
  Delta in [0.0, 1.0]

Estimated runtime: ~5-20 minutes depending on grid resolution.

Parallelization
---------------
Each (k0, Delta) grid point is a short, self-contained CDT run:
build a spacetime, perform n_sweeps sweeps, classify the resulting
volume profile.  No grid point reads or writes state used by any
other, so all points can execute concurrently in threads (--workers).

The GIL is released inside the C++ sweep() call, giving threads real
CPU parallelism without forking processes or duplicating memory.
With a 10x10 grid (100 points) and 8 threads, up to 8 grid points
are evaluated simultaneously.
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

import caset


def classify_phase(profile):
    """
    Classify a volume profile into phase A, B, or C.

    Returns:
      0 = Phase B (crumpled)
      1 = Phase C (de Sitter)
      2 = Phase A (branched polymer)
    """
    if len(profile) == 0:
        return 0

    profile = np.array(profile, dtype=float)
    total = np.sum(profile)
    if total <= 0:
        return 0

    # Concentration: fraction of volume in the largest slice
    concentration = np.max(profile) / total

    # Variance-to-mean ratio (dispersion index)
    mean_vol = np.mean(profile)
    if mean_vol > 0:
        dispersion = np.var(profile) / mean_vol
    else:
        dispersion = 0

    # Number of "active" slices (above 10% of mean)
    active_slices = np.sum(profile > 0.1 * mean_vol) if mean_vol > 0 else 0
    active_fraction = active_slices / max(len(profile), 1)

    # Classification heuristics
    if concentration > 0.5:
        return 0  # Phase B: crumpled (most volume on 1-2 slices)
    elif dispersion > 5 * mean_vol and active_fraction < 0.3:
        return 2  # Phase A: branched polymer (irregular, thin)
    else:
        return 1  # Phase C: de Sitter (extended, smooth)


def run_point(k0, delta, n_simplices, n_sweeps, sweep_cb=None):
    """Run CDT at a single (k0, Delta) point and return the phase.

    GIL is released during sweep(), so threads get real C++ parallelism.
    """
    sig = caset.Signature(4, caset.Lorentzian)
    metric = caset.Metric(True, sig)
    st = caset.Spacetime(metric, caset.CDT, 1.0, 1.0, caset.PREFERRED,
                         caset.Toroid())
    st.build(n_simplices)

    target = st.getSimplexCount()
    cdt = caset.CDTSimulation(st, k0, 0.5, delta, 0.02, target)

    cdt.sweep(n_sweeps, progress=sweep_cb)

    profile = cdt.getVolumeProfile()
    return classify_phase(profile), profile


def main():
    parser = argparse.ArgumentParser(
        description="CDT phase diagram (Fig 3 of hep-th/0505154)")
    parser.add_argument("--n-simplices", type=int, default=200,
                        help="Simplices per simulation point")
    parser.add_argument("--n-sweeps", type=int, default=30,
                        help="Sweeps per point")
    parser.add_argument("--k0-min", type=float, default=0.5)
    parser.add_argument("--k0-max", type=float, default=6.0)
    parser.add_argument("--delta-min", type=float, default=0.0)
    parser.add_argument("--delta-max", type=float, default=1.0)
    parser.add_argument("--grid-size", type=int, default=10,
                        help="Number of grid points per axis")
    parser.add_argument("--workers", type=int,
                        default=min(os.cpu_count() or 1, 8),
                        help="Parallel worker threads (default: min(cpus, 8))")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    n_workers = max(1, args.workers)

    print("=" * 64)
    print("  CDT Phase Diagram Scan")
    print("  Reproduces Fig 3, Ambjorn, Jurkiewicz, Loll (2005)")
    print(f"  Grid: {args.grid_size}x{args.grid_size}, "
          f"N4={args.n_simplices}, sweeps={args.n_sweeps}")
    print(f"  k0 in [{args.k0_min}, {args.k0_max}], "
          f"Delta in [{args.delta_min}, {args.delta_max}]")
    print(f"  Workers: {n_workers} (threads, shared memory)")
    print("=" * 64)

    k0_values = np.linspace(args.k0_min, args.k0_max, args.grid_size)
    delta_values = np.linspace(args.delta_min, args.delta_max, args.grid_size)
    phase_map = np.zeros((len(delta_values), len(k0_values)))

    total_points = len(k0_values) * len(delta_values)
    phase_counts = {"A": 0, "B": 0, "C": 0}

    t0 = time.time()
    total_sweeps = total_points * args.n_sweeps

    pt_bar = tqdm(total=total_points, desc="Grid points", unit="pt",
                  position=0)
    sweep_bar = tqdm(total=total_sweeps, desc="Sweeps", unit="sweep",
                     position=1, leave=False)
    sweep_cb = lambda i, n: sweep_bar.update(1)

    # Each grid point is independent — threads share memory, GIL released
    # during sweep() so all threads compute in parallel.
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for i, delta in enumerate(delta_values):
            for j, k0 in enumerate(k0_values):
                f = pool.submit(run_point, k0, delta,
                                args.n_simplices, args.n_sweeps,
                                sweep_cb)
                futures[f] = (i, j, k0, delta)

        for future in as_completed(futures):
            i, j, k0, delta = futures[future]
            phase, profile = future.result()
            phase_map[i, j] = phase
            label = ["B", "C", "A"][phase]
            phase_counts[label] += 1
            pt_bar.set_postfix_str(
                f"k0={k0:.1f} D={delta:.2f} -> {label}")
            pt_bar.update(1)

    sweep_bar.close()
    pt_bar.close()

    elapsed = time.time() - t0
    print(f"\nScan complete: {elapsed:.1f}s "
          f"({elapsed/total_points:.2f}s per point)")
    print(f"  Phase A (polymer):    {phase_counts['A']:3d} points "
          f"({100*phase_counts['A']/total_points:.0f}%)")
    print(f"  Phase B (crumpled):   {phase_counts['B']:3d} points "
          f"({100*phase_counts['B']/total_points:.0f}%)")
    print(f"  Phase C (de Sitter):  {phase_counts['C']:3d} points "
          f"({100*phase_counts['C']/total_points:.0f}%)")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(9, 7))

    cmap = ListedColormap(["#4477AA", "#66CC66", "#EE6677"])
    phase_labels = ["Phase B\n(crumpled)", r"Phase $C_{dS}$""\n(de Sitter)",
                    "Phase A\n(polymer)"]

    im = ax.pcolormesh(k0_values, delta_values, phase_map,
                       cmap=cmap, vmin=-0.5, vmax=2.5, shading="nearest")

    # Add colorbar with phase labels
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(phase_labels)

    ax.set_xlabel(r"$\kappa_0$", fontsize=14)
    ax.set_ylabel(r"$\Delta$", fontsize=14)
    ax.set_title("CDT Phase Diagram\n"
                 "(cf. Fig 3, Ambjorn et al. 2005;\n"
                 "Gorlich 2013, Phase diagram slide)",
                 fontsize=13)

    # Mark the de Sitter point used in the paper
    ax.plot(2.2, 0.6, "w*", markersize=15, markeredgecolor="black",
            label=r"Paper: $\kappa_0=2.2, \Delta=0.6$")
    ax.legend(fontsize=11, loc="upper right")

    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
