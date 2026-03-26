#!/usr/bin/env python3
# MIT License -- Copyright (c) 2025 Andrew Kelleher
"""
Effective action and minisuperspace comparison.

Reproduces Figures 11-13 from:
  Ambjorn, Jurkiewicz, Loll, "Reconstructing the Universe",
  Phys. Rev. D 72 (2005) [hep-th/0505154]

The effective action for CDT is extracted by measuring how the spatial
volume evolves between adjacent time slices.  For large three-volumes
N_3(tau), the Euclidean effective action takes the form (Eq. 38):

    S_eff ~ sum_tau [ c1/N_3(tau) * (Delta N_3 / Delta tau)^2
                     + c2 * N_3^alpha - lambda * N_3(tau) ]

The paper shows that:
  1. The kinetic term scaling dimension is D_2 ~ 2 (Fig 11).
  2. The distribution of rescaled volume differences is Gaussian (Fig 12).
  3. The volume-volume correlator from the effective action matches the
     Monte Carlo data (Fig 13), confirming that CDT dynamics is
     described by the minisuperspace action (Eq. 40):

        S = (1/G) integral[ a(tau) (da/dtau)^2 + a(tau) - lambda a^3(tau) ] dtau

     but with a flipped sign on the kinetic term compared to the classical
     Einstein-Hilbert action (the "conformal mode problem" is solved
     nonperturbatively).

Parameters: k0 = 2.2, Delta = 0.6.
Estimated runtime: ~2-5 minutes.
"""
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

import caset


def collect_profiles(n_simplices, n_therm, n_meas, interval):
    """Run CDT and collect volume profiles."""
    sig = caset.Signature(4, caset.Lorentzian)
    metric = caset.Metric(True, sig)
    st = caset.Spacetime(metric, caset.CDT, 1.0, 1.0, caset.PREFERRED,
                         caset.Toroid())
    st.build(n_simplices)

    target = st.getSimplexCount()
    cdt = caset.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, target)

    for _ in tqdm(range(n_therm), desc="  Thermalizing",
                  unit="sweep", leave=False):
        cdt.sweep()

    profiles = []
    total_sweeps = n_meas * interval
    pbar = tqdm(total=total_sweeps, desc="  Measuring",
                unit="sweep", leave=False)
    for _ in range(n_meas):
        for _ in range(interval):
            cdt.sweep()
            pbar.update(1)
        profiles.append(np.array(cdt.getVolumeProfile(), dtype=float))
    pbar.close()

    return profiles


def main():
    parser = argparse.ArgumentParser(
        description="Effective action analysis "
                    "(Figs 11-13 of hep-th/0505154)")
    parser.add_argument("--n-simplices", type=int, default=500)
    parser.add_argument("--n-therm", type=int, default=50)
    parser.add_argument("--n-meas", type=int, default=40)
    parser.add_argument("--meas-interval", type=int, default=5)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    print("=" * 64)
    print("  Effective Action & Minisuperspace Comparison")
    print("  Reproduces Figs 11-13, Ambjorn, Jurkiewicz, Loll (2005)")
    print("  Parameters: k0=2.2, Delta=0.6")
    print(f"  N4={args.n_simplices}, therm={args.n_therm}, "
          f"meas={args.n_meas}, interval={args.meas_interval}")
    print("=" * 64)

    print("\nCollecting configurations...")
    t0 = time.time()
    t_total = time.time()
    profiles = collect_profiles(
        args.n_simplices, args.n_therm, args.n_meas, args.meas_interval)
    avg_slices = np.mean([len(p) for p in profiles])
    avg_vol = np.mean([np.sum(p) for p in profiles])
    print(f"  {len(profiles)} configurations in {time.time()-t0:.1f}s")
    print(f"  Avg slices: {avg_slices:.0f}, avg total volume: {avg_vol:,.0f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ---- Fig 11: D_2 from finite-size scaling ----
    # Measure distribution of volume differences at different N_3 values
    ax_d2 = axes[0, 0]

    D_2_range = np.linspace(1.0, 3.0, 30)
    errors = []

    all_diffs_by_n3 = {}
    for p in profiles:
        for tau in range(len(p) - 1):
            n3 = p[tau] + p[tau + 1]
            if n3 <= 0:
                continue
            diff = abs(p[tau + 1] - p[tau])
            bucket = int(n3 / 50) * 50
            if bucket not in all_diffs_by_n3:
                all_diffs_by_n3[bucket] = []
            all_diffs_by_n3[bucket].append(diff)

    # For each D_2, compute rescaled distributions and measure overlap
    for D_2 in D_2_range:
        all_z = []
        for bucket, diffs in all_diffs_by_n3.items():
            n3 = max(bucket, 1)
            z = np.array(diffs) / (n3 ** (1.0 / D_2))
            all_z.extend(z.tolist())
        if len(all_z) < 10:
            errors.append(1e10)
            continue
        all_z = np.array(all_z)
        errors.append(np.std(all_z))

    errors = np.array(errors)
    if np.any(errors < 1e9):
        errors_norm = errors / errors[errors < 1e9].max()
        ax_d2.plot(D_2_range, errors_norm, "k-", linewidth=1.5)
    ax_d2.axvline(x=2.0, color="gray", linestyle=":", label=r"$D_2=2$")
    ax_d2.set_xlabel(r"$D_2$", fontsize=12)
    ax_d2.set_ylabel("Overlap error (normalized)", fontsize=12)
    ax_d2.set_title(r"Scaling dimension $D_2$ from finite-size scaling"
                    "\n(cf. Fig 11, Ambjorn et al. 2005)")
    ax_d2.legend(fontsize=10)
    ax_d2.grid(True, alpha=0.3)

    # ---- Fig 12: Volume difference distribution ----
    ax_dist = axes[0, 1]

    for p in profiles:
        for tau in range(len(p) - 1):
            n3 = p[tau] + p[tau + 1]
            if n3 <= 0:
                continue

    all_z = []
    for p in profiles:
        for tau in range(len(p) - 1):
            n3_total = p[tau] + p[tau + 1]
            if n3_total <= 0:
                continue
            diff = abs(p[tau + 1] - p[tau])
            z = diff / max(n3_total ** 0.5, 1e-10)  # D_2 = 2
            all_z.append(z)

    if len(all_z) > 10:
        all_z = np.array(all_z)
        hist, edges = np.histogram(all_z, bins=25, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax_dist.plot(centers, hist, "o", markersize=4, color="blue",
                     label="Measured")

        try:
            def gauss(x, A, c):
                return A * np.exp(-c * x**2)
            popt, _ = curve_fit(gauss, centers[hist > 0], hist[hist > 0],
                                p0=[hist.max(), 1.0], maxfev=5000)
            x_fit = np.linspace(0, centers.max(), 100)
            ax_dist.plot(x_fit, gauss(x_fit, *popt), "r-", linewidth=1.5,
                         label=fr"$e^{{-{popt[1]:.1f} z^2}}$")
        except Exception:
            pass

    ax_dist.set_xlabel(r"$z$", fontsize=12)
    ax_dist.set_ylabel(r"$P_{N_3}(z)$", fontsize=12)
    ax_dist.set_title("Volume difference distribution\n"
                      "(cf. Fig 12, Ambjorn et al. 2005)")
    ax_dist.legend(fontsize=10)
    ax_dist.grid(True, alpha=0.3)

    # ---- Fig 13: Volume-volume correlator vs minisuperspace ----
    ax_mss = axes[1, 0]

    # Average profile
    max_len = max(len(p) for p in profiles)
    avg_profile = np.zeros(max_len)
    counts = np.zeros(max_len)
    for p in profiles:
        avg_profile[:len(p)] += p
        counts[:len(p)] += 1
    counts[counts == 0] = 1
    avg_profile /= counts

    T = len(avg_profile)
    tau = np.arange(T)

    # The minisuperspace prediction: N_3(tau) ~ cos^4(pi*tau/T)
    # with the effective action S_eff from Eq. 40
    cos4 = np.cos(np.pi * (tau - T / 2.0) / T) ** 4
    if cos4.max() > 0:
        cos4 *= avg_profile.max() / cos4.max()

    ax_mss.plot(tau, avg_profile, "ko", markersize=4, label="Monte Carlo")
    ax_mss.plot(tau, cos4, "r-", linewidth=1.5,
                label=r"Minisuperspace: $\cos^4(\pi\tau/T)$")
    ax_mss.set_xlabel(r"$\tau$", fontsize=12)
    ax_mss.set_ylabel(r"$N_3(\tau)$", fontsize=12)
    ax_mss.set_title("Volume profile vs minisuperspace prediction\n"
                     "(cf. Fig 13, Ambjorn et al. 2005)")
    ax_mss.legend(fontsize=10)
    ax_mss.grid(True, alpha=0.3)

    # ---- Action tracking over simulation ----
    ax_action = axes[1, 1]

    sig = caset.Signature(4, caset.Lorentzian)
    metric = caset.Metric(True, sig)
    st = caset.Spacetime(metric, caset.CDT, 1.0, 1.0, caset.PREFERRED,
                         caset.Toroid())
    st.build(args.n_simplices)
    target = st.getSimplexCount()
    cdt = caset.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, target)

    actions = []
    volumes = []
    for sweep_num in tqdm(range(100), desc="  Action tracking",
                          unit="sweep", leave=False):
        cdt.sweep()
        actions.append(cdt.computeAction())
        volumes.append(st.getSimplexCount())

    ax_action.plot(actions, "b-", linewidth=0.8, alpha=0.7)
    ax_action.set_xlabel("Sweep", fontsize=12)
    ax_action.set_ylabel(r"$S_{\mathrm{Regge}}$", fontsize=12)
    ax_action.set_title("Regge action evolution")
    ax_action.grid(True, alpha=0.3)

    # Inset: volume evolution
    ax_vol = ax_action.twinx()
    ax_vol.plot(volumes, "r-", linewidth=0.8, alpha=0.5)
    ax_vol.set_ylabel(r"$N_4$", color="red", fontsize=12)
    ax_vol.tick_params(axis="y", labelcolor="red")

    fig.suptitle(r"Effective action analysis ($\kappa_0=2.2$, $\Delta=0.6$)",
                 fontsize=14, y=1.01)
    fig.tight_layout()

    print(f"\nAction range: [{min(actions):.2f}, {max(actions):.2f}], "
          f"final N4={volumes[-1]:,}")
    print(f"Total elapsed: {time.time()-t_total:.1f}s")

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
