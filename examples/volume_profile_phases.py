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
"""
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

import caset


def run_cdt(n_simplices, k0, delta, n_therm, n_meas, meas_interval,
            phase_label=""):
    sig = caset.Signature(4, caset.Lorentzian)
    metric = caset.Metric(True, sig)
    st = caset.Spacetime(metric, caset.CDT, 1.0, 1.0, caset.PREFERRED,
                         caset.Toroid())
    st.build(n_simplices)
    target = st.getSimplexCount()
    cdt = caset.CDTSimulation(st, k0, 0.5, delta, 0.02, target)
    prefix = f"  [{phase_label}] " if phase_label else "  "
    for _ in tqdm(range(n_therm), desc=f"{prefix}Thermalizing",
                  unit="sweep", leave=False):
        cdt.sweep()
    profiles = []
    total_sweeps = n_meas * meas_interval
    pbar = tqdm(total=total_sweeps, desc=f"{prefix}Measuring",
                unit="sweep", leave=False)
    for _ in range(n_meas):
        for _ in range(meas_interval):
            cdt.sweep()
            pbar.update(1)
        profiles.append(cdt.getVolumeProfile())
    pbar.close()
    return profiles, cdt.getAcceptanceRates()


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
    parser.add_argument("--n-simplices", type=int, default=500)
    parser.add_argument("--n-therm", type=int, default=50)
    parser.add_argument("--n-meas", type=int, default=20)
    parser.add_argument("--meas-interval", type=int, default=5)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    print("=" * 64)
    print("  CDT Volume Profiles in Phases A, B, C")
    print("  Reproduces Figs 4-6, Ambjorn, Jurkiewicz, Loll (2005)")
    print(f"  N4={args.n_simplices}, therm={args.n_therm}, "
          f"meas={args.n_meas}, interval={args.meas_interval}")
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

    for idx, (label, (k0, delta)) in enumerate(phases.items()):
        short = label.split("\n")[0]
        print(f"\n--- {short} (k0={k0}, Delta={delta}) "
              f"[{idx+1}/{len(phases)}] ---")
        t0 = time.time()
        profiles, rates = run_cdt(
            args.n_simplices, k0, delta,
            args.n_therm, args.n_meas, args.meas_interval,
            phase_label=short)
        elapsed = time.time() - t0
        avg = average_profile(profiles)
        peak_slice = np.argmax(avg)
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"  Time slices: {len(avg)}, "
              f"peak at tau={peak_slice} (N3={avg[peak_slice]:.1f})")
        print(f"  Volume: mean={avg.mean():.1f}, max={avg.max():.1f}, "
              f"total={avg.sum():.0f}")
        print(f"  Acceptance rates: {rates}")

        ax_surf = fig_surf.add_subplot(1, 3, idx + 1, projection="3d")
        plot_universe_surface(avg, label, ax_surf)

        short_label = label.split("\n")[0]
        ax_line.plot(np.arange(len(avg)), avg, "o-", label=short_label,
                     linewidth=2, markersize=4)

    # Overlay cos^3 reference on the line plot (Eq. 28, hep-th/0505154)
    T_ref = 7
    tau_ref = np.linspace(0, T_ref - 1, 100)
    cos3_ref = np.cos(np.pi * (tau_ref - (T_ref - 1) / 2) / T_ref) ** 3
    ax_line.plot(tau_ref, cos3_ref * ax_line.get_ylim()[1] * 0.9, "k--",
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
