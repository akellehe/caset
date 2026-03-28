#!/usr/bin/env python3
# MIT License -- Copyright (c) 2025 Andrew Kelleher
"""
Volume profiles in the three CDT phases (A, B, C).

Reproduces Figures 4, 5, 6 from:
  Ambjorn, Jurkiewicz, Loll, "Reconstructing the Universe",
  Phys. Rev. D 72 (2005) [hep-th/0505154]

Paper parameters: k0=2.2, Delta=0.6, N4=10k-362k, T=80.
Our parameters are smaller; the qualitative shape differences between
phases emerge at N4 > ~5000.

The Regge action is (Eq. 2 of hep-th/0505154):

  S_E = -(k0 + 6*Delta)*N0 + (k4 + 2*Delta)*N41 + (k4 + Delta)*N32

To reproduce the paper results (Figs 4-6):
  python examples/volume_profile_phases.py \
      --n-simplices 80000 --n-therm 500 --n-meas 100 \
      --meas-interval 50

The paper uses N4 up to 362k; 80k is sufficient to clearly
distinguish the three phase profiles (blob, crumpled, polymer).

Parallelization
---------------
The three phases (A, B, C_dS) use different coupling constants (k0,
Delta) and each builds its own spacetime from scratch.  No state is
shared between phases, so all three run concurrently in threads
(--workers).

The GIL is released inside the C++ sweep() call, giving threads real
CPU parallelism without forking processes or duplicating memory.
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

import caset


def _phase_worker(label, k0, delta, n_simplices, n_therm, n_meas,
                  meas_interval, sweep_cb=None):
    """Run one phase simulation: build, tune, thermalize, measure.

    Each phase uses different coupling constants and is fully independent.
    The GIL is released during sweep(), so threads run in parallel.
    """
    sig = caset.Signature(4, caset.Lorentzian)
    metric = caset.Metric(True, sig)
    st = caset.Spacetime(metric, caset.CDT, 1.0, 1.0, caset.PREFERRED,
                         caset.Toroid())
    st.build(n_simplices)
    target = st.getN41()  # [RU] eq. 6: volume-fix targets N41
    cdt = caset.CDTSimulation(st, k0, 0.5, delta, 0.02, target)

    cdt.tune()
    cdt.sweep(n_therm, progress=sweep_cb)

    profiles = []
    for _ in range(n_meas):
        cdt.sweep(meas_interval, progress=sweep_cb)
        profiles.append(cdt.getVolumeProfile())

    return label, profiles, cdt.getAcceptanceRates(), cdt.getK4()


def average_profile(profiles):
    max_len = max(len(p) for p in profiles)
    avg = np.zeros(max_len)
    counts = np.zeros(max_len)
    for p in profiles:
        avg[:len(p)] += p
        counts[:len(p)] += 1
    counts[counts == 0] = 1
    return avg / counts


def plot_universe_surface(profile, title, ax, color_map=cm.coolwarm):
    T = len(profile)
    tau = np.linspace(0, 1, T)
    radius = np.sqrt(np.maximum(profile, 0))
    if radius.max() > 0:
        radius = radius / radius.max()

    n_theta = 60
    theta = np.linspace(0, 2 * np.pi, n_theta)
    tau_grid, theta_grid = np.meshgrid(tau, theta)
    r_grid = np.tile(radius, (n_theta, 1))

    X = r_grid * np.cos(theta_grid)
    Y = r_grid * np.sin(theta_grid)
    Z = tau_grid

    ax.plot_surface(X, Y, Z, cmap=color_map, alpha=0.85, edgecolor="none")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(r"$\tau$")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)


def main():
    parser = argparse.ArgumentParser(
        description="CDT volume profiles in phases A, B, C "
                    "(Figs 4-6 of hep-th/0505154)")
    parser.add_argument("--n-simplices", type=int, default=800)
    parser.add_argument("--n-therm", type=int, default=80)
    parser.add_argument("--n-meas", type=int, default=30)
    parser.add_argument("--meas-interval", type=int, default=5)
    parser.add_argument("--workers", type=int,
                        default=min(os.cpu_count() or 1, 8),
                        help="Parallel worker threads (default: min(cpus, 8))")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    n_workers = max(1, args.workers)

    print("=" * 64)
    print("  CDT Volume Profiles in Phases A, B, C")
    print("  Reproduces Figs 4-6, Ambjorn, Jurkiewicz, Loll (2005)")
    print(f"  N4={args.n_simplices}, therm={args.n_therm}, "
          f"meas={args.n_meas}, interval={args.meas_interval}")
    print(f"  Workers: {n_workers} (threads, shared memory)")
    print("=" * 64)

    phases = {
        "Phase A\n"r"($\kappa_0=5.0,\;\Delta=0$)": (5.0, 0.0),
        "Phase B\n"r"($\kappa_0=1.6,\;\Delta=0$)": (1.6, 0.0),
        r"Phase $C_{dS}$""\n"r"($\kappa_0=2.2,\;\Delta=0.6$)": (2.2, 0.6),
    }

    fig_surf = plt.figure(figsize=(18, 6))
    fig_surf.suptitle(
        "CDT Universe Snapshots  (cf. Figs 4-6, Ambjorn, Jurkiewicz, Loll,\n"
        r"$\it{Reconstructing\ the\ Universe}$, Phys. Rev. D 72, 2005"
        f"  [N4={args.n_simplices}])",
        fontsize=13)

    fig_line, ax_line = plt.subplots(figsize=(10, 6))
    t_total = time.time()

    # All three phases are independent — run in parallel.
    # We intentionally do NOT pass a per-sweep progress callback here:
    # a Python callback would reacquire the GIL every sweep, serializing
    # the threads.  Instead we report progress at the phase level.
    n_phases = len(phases)

    phase_bar = tqdm(total=n_phases, desc="Phases", unit="phase")

    phase_results = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_phase_worker, label, k0, delta,
                        args.n_simplices, args.n_therm,
                        args.n_meas, args.meas_interval): label
            for label, (k0, delta) in phases.items()
        }
        for f in as_completed(futures):
            label, profiles, rates, k4 = f.result()
            phase_results[label] = (profiles, rates)
            short = label.split("\n")[0]
            phase_bar.set_postfix_str(short)
            phase_bar.update(1)

    phase_bar.close()

    for label, (profiles, rates) in phase_results.items():
        short = label.split("\n")[0]
        avg = average_profile(profiles)
        print(f"  {short}: slices={len(avg)}, "
              f"peak N3={avg.max():.1f}, rates={rates}")

    phase_c_avg = None  # track for cos^3 overlay

    for idx, (label, (k0, delta)) in enumerate(phases.items()):
        profiles, rates = phase_results[label]
        avg = average_profile(profiles)
        peak_slice = np.argmax(avg)

        ax_surf = fig_surf.add_subplot(1, 3, idx + 1, projection="3d")
        plot_universe_surface(avg, label, ax_surf)

        short_label = label.split("\n")[0]
        ax_line.plot(np.arange(len(avg)), avg, "o-", label=short_label,
                     linewidth=2, markersize=4)

        if "C_{dS}" in label:
            phase_c_avg = avg

    # Overlay cos^3 reference on the line plot (Eq. 28, hep-th/0505154)
    # On a torus the de Sitter blob sits on a nonzero stalk; use periodic
    # distance from the measured peak.
    if phase_c_avg is not None:
        T_ref = len(phase_c_avg)
        peak_tau = np.argmax(phase_c_avg)
        stalk = float(np.min(phase_c_avg))
        amplitude = float(phase_c_avg.max()) - stalk
        tau_ref = np.linspace(0, T_ref - 1, 200)
        dist = np.abs(tau_ref - peak_tau)
        dist = np.minimum(dist, T_ref - dist)
        cos_arg = np.pi * dist / T_ref
        cos3_ref = stalk + amplitude * np.maximum(np.cos(cos_arg), 0) ** 3
        ax_line.plot(tau_ref, cos3_ref, "k--",
                     alpha=0.4, linewidth=1.5, label=r"$\cos^3$ reference")

    ax_line.set_xlabel(r"Time slice $\tau$", fontsize=13)
    ax_line.set_ylabel(r"$N_3(\tau)$", fontsize=13)
    ax_line.set_title(
        r"Spatial volume profile $N_3(\tau)$"
        "\n(cf. Figs 4-6, Ambjorn, Jurkiewicz, Loll, "
        r"$\it{Reconstructing\ the\ Universe}$, 2005)")
    ax_line.legend(fontsize=11)
    ax_line.grid(True, alpha=0.3)

    fig_surf.tight_layout(rect=[0, 0, 1, 0.90])
    fig_line.tight_layout()

    print(f"\nTotal elapsed: {time.time()-t_total:.1f}s")
    if args.save:
        fig_surf.savefig(args.save.replace(".png", "_surface.png"), dpi=150)
        fig_line.savefig(args.save.replace(".png", "_profile.png"), dpi=150)
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
