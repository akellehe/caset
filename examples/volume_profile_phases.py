#!/usr/bin/env python3
# MIT License -- Copyright (c) 2025 Andrew Kelleher
"""
Volume profiles in the three CDT phases (A, B, C).

Reproduces Figures 4, 5, 6 from:
  Ambjorn, Jurkiewicz, Loll, "Reconstructing the Universe",
  Phys. Rev. D 72 (2005) [hep-th/0505154]

The volume profile N_3(tau) -- the number of spatial tetrahedra at each
integer time tau -- is the primary observable distinguishing the phases:

  Phase A  (large k0):  branched-polymer, thin elongated profile.
  Phase B  (small k0, small Delta): crumpled, collapsed to ~2 slices.
  Phase C  (moderate k0, Delta > 0):  extended de Sitter universe
           with N_3(tau) ~ cos^4(pi*tau/T).

The figures in the paper use 3D surface-of-revolution plots where the
circumference at each time tau is proportional to N_3(tau).  We reproduce
this using a PyTorch-optimized layout for the vertex positions and
matplotlib's 3D surface rendering.

Parameters from the paper (Table 1, Section 3):
  k0 = 2.2,  Delta = 0.6  (phase C)
  k0 = 5.0,  Delta = 0    (phase A)
  k0 = 1.6,  Delta = 0    (phase B)
  t = 80 time slices, epsilon ~ 0.02

Estimated runtime: ~2-5 minutes per phase at N4 ~ 500.
"""
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import caset


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def run_cdt(n_simplices, k0, delta, n_therm, n_meas, meas_interval):
    """Run a CDT simulation and collect volume profiles."""
    sig = caset.Signature(4, caset.Lorentzian)
    metric = caset.Metric(True, sig)
    topo = caset.Toroid()
    st = caset.Spacetime(metric, caset.CDT, 1.0, 1.0, caset.PREFERRED, topo)
    st.build(n_simplices)

    target = st.getSimplexCount()
    cdt = caset.CDTSimulation(st, k0, 0.5, delta, 0.02, target)

    # Thermalize
    for _ in range(n_therm):
        cdt.sweep()

    # Measure
    profiles = []
    for _ in range(n_meas):
        for _ in range(meas_interval):
            cdt.sweep()
        profiles.append(cdt.getVolumeProfile())

    return profiles, cdt.getAcceptanceRates()


def average_profile(profiles):
    """Average a list of variable-length profiles, padding with zeros."""
    max_len = max(len(p) for p in profiles)
    avg = np.zeros(max_len)
    counts = np.zeros(max_len)
    for p in profiles:
        avg[: len(p)] += p
        counts[: len(p)] += 1
    counts[counts == 0] = 1
    return avg / counts


# ---------------------------------------------------------------------------
# Plotting: surface-of-revolution (Figs 4-6 style)
# ---------------------------------------------------------------------------

def plot_universe_surface(profile, title, ax, color_map=cm.coolwarm):
    """
    Render the volume profile as a 3D surface of revolution, matching the
    style of Figures 4-6 in 'Reconstructing the Universe'.

    At each time slice tau the circumference is proportional to sqrt(N_3(tau)),
    giving a surface whose cross-sectional area is proportional to N_3.
    """
    T = len(profile)
    tau = np.arange(T)
    radius = np.sqrt(np.maximum(profile, 0))
    if radius.max() > 0:
        radius = radius / radius.max()  # normalize to [0, 1]

    # Create surface of revolution
    n_theta = 60
    theta = np.linspace(0, 2 * np.pi, n_theta)
    tau_grid, theta_grid = np.meshgrid(tau, theta)
    r_grid = np.tile(radius, (n_theta, 1))

    X = r_grid * np.cos(theta_grid)
    Y = r_grid * np.sin(theta_grid)
    Z = tau_grid.astype(float)

    ax.plot_surface(X, Y, Z, cmap=color_map, alpha=0.8, edgecolor="none")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(r"$\tau$")
    ax.set_title(title)


def plot_profile_line(profile, label, ax, **kwargs):
    """Simple line plot of N_3(tau) vs tau."""
    tau = np.arange(len(profile))
    ax.plot(tau, profile, label=label, linewidth=2, **kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CDT volume profiles in phases A, B, C "
                    "(Figs 4-6 of hep-th/0505154)")
    parser.add_argument("--n-simplices", type=int, default=500,
                        help="Initial number of simplices (default 500)")
    parser.add_argument("--n-therm", type=int, default=50,
                        help="Number of thermalization sweeps")
    parser.add_argument("--n-meas", type=int, default=20,
                        help="Number of measurements")
    parser.add_argument("--meas-interval", type=int, default=5,
                        help="Sweeps between measurements")
    parser.add_argument("--save", type=str, default=None,
                        help="Save figure to this path instead of showing")
    args = parser.parse_args()

    # Phase parameters from the paper
    phases = {
        "Phase A\n"r"($\kappa_0=5.0,\;\Delta=0$)": (5.0, 0.0),
        "Phase B\n"r"($\kappa_0=1.6,\;\Delta=0$)": (1.6, 0.0),
        r"Phase $C_{dS}$""\n"r"($\kappa_0=2.2,\;\Delta=0.6$)": (2.2, 0.6),
    }

    # ---- Surface-of-revolution plots (Figs 4-6 style) ----
    fig_surf = plt.figure(figsize=(18, 6))
    fig_surf.suptitle(
        "CDT Universe Snapshots  (cf. Figs 4-6, Ambjorn et al. 2005)",
        fontsize=14)

    # ---- Line plot of N_3(tau) ----
    fig_line, ax_line = plt.subplots(figsize=(10, 6))

    for idx, (label, (k0, delta)) in enumerate(phases.items()):
        print(f"\n{'='*60}")
        print(f"Running {label.replace(chr(10), ' ')}:  k0={k0}, delta={delta}")
        print(f"{'='*60}")

        t0 = time.time()
        profiles, rates = run_cdt(
            args.n_simplices, k0, delta,
            args.n_therm, args.n_meas, args.meas_interval)
        elapsed = time.time() - t0

        avg = average_profile(profiles)
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"  Profile ({len(avg)} slices): {np.round(avg[:10], 1)}...")
        print(f"  Acceptance rates: {rates}")

        # Surface subplot
        ax_surf = fig_surf.add_subplot(1, 3, idx + 1, projection="3d")
        plot_universe_surface(avg, label, ax_surf)

        # Line plot
        short_label = label.split("\n")[0]
        plot_profile_line(avg, short_label, ax_line)

    ax_line.set_xlabel(r"Time slice $\tau$", fontsize=13)
    ax_line.set_ylabel(r"$N_3(\tau)$", fontsize=13)
    ax_line.set_title(r"Spatial volume profile $N_3(\tau)$"
                      "\n(cf. Figs 4-6, Ambjorn et al. 2005)")
    ax_line.legend(fontsize=11)
    ax_line.grid(True, alpha=0.3)

    fig_surf.tight_layout(rect=[0, 0, 1, 0.93])
    fig_line.tight_layout()

    if args.save:
        fig_surf.savefig(args.save.replace(".png", "_surface.png"), dpi=150)
        fig_line.savefig(args.save.replace(".png", "_profile.png"), dpi=150)
        print(f"\nSaved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
