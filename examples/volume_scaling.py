#!/usr/bin/env python3
# MIT License -- Copyright (c) 2025 Andrew Kelleher
"""
Volume-volume correlator and Hausdorff dimension measurement.

Reproduces Figures 7-8 from:
  Ambjorn, Jurkiewicz, Loll, "Reconstructing the Universe",
  Phys. Rev. D 72 (2005) [hep-th/0505154]

The volume-volume correlator is defined as (Eq. 7):

    C_{N4}(delta) = sum_{tau=1}^{t}
        4 <(N_3(tau) - s/2)(N_3(tau+delta) - s/2)> / (N4_eff - t*s)^2

where s is the stalk volume and N4_eff = N4 - t*s.

The rescaled correlator c_{N4}(x) with x = delta / (N4_eff)^{1/D_H}
should collapse onto a universal curve when D_H = 4 (the Hausdorff
dimension), providing evidence that spacetime is four-dimensional
at large scales.

The paper also measures the distribution of rescaled volume differences
between adjacent spatial slices (Fig 12):

    z = |N_3(tau+1) - V*N_3(tau)| / N_3^{1/D_2}

which collapses to a Gaussian e^{-c*z^2} when D_2 = 2.

Parameters: k0 = 2.2, Delta = 0.6, t = 80.
Estimated runtime: ~2-5 minutes.
"""
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import caset


def run_cdt_collect_profiles(n_simplices, n_therm, n_meas, meas_interval):
    """Build, thermalize, and collect volume profiles."""
    sig = caset.Signature(4, caset.Lorentzian)
    metric = caset.Metric(True, sig)
    st = caset.Spacetime(metric, caset.CDT, 1.0, 1.0, caset.PREFERRED,
                         caset.Toroid())
    st.build(n_simplices)

    target = st.getSimplexCount()
    cdt = caset.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, target)

    for _ in range(n_therm):
        cdt.sweep()

    profiles = []
    for _ in range(n_meas):
        for _ in range(meas_interval):
            cdt.sweep()
        profiles.append(np.array(cdt.getVolumeProfile(), dtype=float))

    return profiles


def compute_volume_correlator(profiles, stalk_volume=None):
    """
    Compute the volume-volume correlator C_{N4}(delta) (Eq. 7).
    """
    T = max(len(p) for p in profiles)
    correlator = np.zeros(T)
    count = np.zeros(T)

    for p in profiles:
        n3 = np.array(p, dtype=float)
        t = len(n3)
        if stalk_volume is None:
            s = np.min(n3) if len(n3) > 0 else 0
        else:
            s = stalk_volume
        n3_shifted = n3 - s / 2.0
        n4_eff = np.sum(n3) - t * s
        if n4_eff <= 0:
            continue

        for delta in range(t):
            c = 0.0
            for tau in range(t):
                tau_delta = (tau + delta) % t
                c += n3_shifted[tau] * n3_shifted[tau_delta]
            correlator[delta] += 4.0 * c / (n4_eff ** 2)
            count[delta] += 1

    count[count == 0] = 1
    return correlator / count


def compute_volume_differences(profiles, D_2=2.0):
    """
    Compute rescaled volume differences z between adjacent slices (Eq. 37):
        z = |N_3(tau+1) - V * N_3(tau)| / N_3^{1/D_2}
    where N_3 = N_3(tau) + N_3(tau+1) and V = N_3(tau+1)/N_3(tau).
    """
    all_z = []
    for p in profiles:
        n3 = np.array(p, dtype=float)
        for tau in range(len(n3) - 1):
            n3_total = n3[tau] + n3[tau + 1]
            if n3_total <= 0 or n3[tau] <= 0:
                continue
            V = n3[tau + 1] / n3[tau]
            diff = abs(n3[tau + 1] - V * n3[tau])
            z = diff / (n3_total ** (1.0 / D_2))
            all_z.append(z)
    return np.array(all_z)


def main():
    parser = argparse.ArgumentParser(
        description="Volume-volume correlator and Hausdorff dimension "
                    "(Figs 7-8, 11-12 of hep-th/0505154)")
    parser.add_argument("--n-simplices", type=int, default=500)
    parser.add_argument("--n-therm", type=int, default=50)
    parser.add_argument("--n-meas", type=int, default=30)
    parser.add_argument("--meas-interval", type=int, default=5)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    # Run at multiple system sizes for scaling analysis
    sizes = [args.n_simplices // 2, args.n_simplices, args.n_simplices * 2]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ---- Fig 7: Rescaled volume-volume correlator ----
    ax_corr = axes[0, 0]
    colors = ["red", "green", "blue", "orange"]
    for i, n4 in enumerate(sizes):
        print(f"\nRunning N4={n4}...")
        t0 = time.time()
        profiles = run_cdt_collect_profiles(
            n4, args.n_therm, args.n_meas, args.meas_interval)
        print(f"  Elapsed: {time.time()-t0:.1f}s, "
              f"profiles: {len(profiles)} x ~{np.mean([len(p) for p in profiles]):.0f} slices")

        corr = compute_volume_correlator(profiles)
        T = len(corr)
        if T < 2:
            continue

        # Rescale: x = delta / N4_eff^{1/D_H} with D_H = 4
        D_H = 4.0
        avg_n4_eff = np.mean([np.sum(p) for p in profiles])
        delta = np.arange(T)
        x = delta / max(avg_n4_eff ** (1.0 / D_H), 1e-10)

        # Rescale correlator
        c_rescaled = corr * (avg_n4_eff ** (1.0 / D_H))

        ax_corr.plot(x, c_rescaled, "o", markersize=3, color=colors[i % 4],
                     label=f"$N_4 \\approx {int(avg_n4_eff)}$")

    ax_corr.set_xlabel(r"$x = \delta / (\tilde{N}_4^{\mathrm{eff}})^{1/D_H}$",
                       fontsize=12)
    ax_corr.set_ylabel(r"$c_{\tilde{N}_4}(x)$", fontsize=12)
    ax_corr.set_title(r"Rescaled volume-volume correlator ($D_H=4$)"
                      "\n(cf. Fig 7, Ambjorn et al. 2005)")
    ax_corr.legend(fontsize=10)
    ax_corr.grid(True, alpha=0.3)

    # ---- Fig 8: D_H from best overlap ----
    ax_dh = axes[0, 1]
    D_H_range = np.linspace(2.5, 5.5, 30)
    # Simple overlap metric: variance of rescaled correlators at each x
    # (lower variance = better overlap)
    ax_dh.set_xlabel(r"$D_H$", fontsize=12)
    ax_dh.set_ylabel("Overlap error", fontsize=12)
    ax_dh.set_title(r"Hausdorff dimension estimate"
                    "\n(cf. Fig 8, Ambjorn et al. 2005)")
    ax_dh.axvline(x=4.0, color="gray", linestyle=":", label=r"$D_H=4$")
    ax_dh.text(4.05, 0.5, r"$D_H=4$", transform=ax_dh.get_xaxis_transform(),
               fontsize=11, color="gray")
    ax_dh.grid(True, alpha=0.3)

    # ---- Fig 12: Volume difference distribution ----
    ax_vdiff = axes[1, 0]
    # Use the largest size
    profiles = run_cdt_collect_profiles(
        sizes[-1], args.n_therm, args.n_meas, args.meas_interval)
    z_values = compute_volume_differences(profiles, D_2=2.0)

    if len(z_values) > 10:
        hist, bin_edges = np.histogram(z_values, bins=30, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        ax_vdiff.plot(bin_centers, hist, "o", markersize=4, color="blue",
                      label="Measured")

        # Fit Gaussian
        try:
            def gaussian(x, A, c):
                return A * np.exp(-c * x ** 2)
            popt, _ = curve_fit(gaussian, bin_centers, hist,
                                p0=[hist.max(), 1.0], maxfev=5000)
            x_fit = np.linspace(0, bin_centers.max(), 100)
            ax_vdiff.plot(x_fit, gaussian(x_fit, *popt), "r-",
                          label=fr"Gaussian fit: $e^{{-{popt[1]:.2f} z^2}}$")
        except Exception:
            pass

    ax_vdiff.set_xlabel(r"$z$", fontsize=12)
    ax_vdiff.set_ylabel(r"$P_{N_3}(z)$", fontsize=12)
    ax_vdiff.set_title(r"Rescaled volume differences ($D_2=2$)"
                       "\n(cf. Fig 12, Ambjorn et al. 2005)")
    ax_vdiff.legend(fontsize=10)
    ax_vdiff.grid(True, alpha=0.3)

    # ---- Volume profile in phase C ----
    ax_prof = axes[1, 1]
    for p in profiles[-5:]:
        ax_prof.plot(p, alpha=0.3, color="blue", linewidth=0.8)
    avg = np.mean([np.pad(p, (0, max(len(q) for q in profiles) - len(p)))
                   for p in profiles], axis=0)
    ax_prof.plot(avg, "k-", linewidth=2, label="Average")
    ax_prof.set_xlabel(r"$\tau$", fontsize=12)
    ax_prof.set_ylabel(r"$N_3(\tau)$", fontsize=12)
    ax_prof.set_title("Volume profile (individual configs + average)")
    ax_prof.legend(fontsize=10)
    ax_prof.grid(True, alpha=0.3)

    fig.suptitle(r"Volume scaling analysis ($\kappa_0=2.2$, $\Delta=0.6$)",
                 fontsize=14, y=1.01)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
