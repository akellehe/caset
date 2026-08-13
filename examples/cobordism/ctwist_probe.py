# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""C-twist probe: where matter/antimatter lives (and doesn't) in the readouts.

EXPERIMENT (untracked). Three demonstrations on one dumped frame:

1. MAGNETIC FLUX SWEEP (k=0). Thread a U(1) phase theta through one edge
   (Edge.setPhase) and sweep theta in [0, 2pi]: the Hermitian magnetic
   Laplacian's bands flow, the zero mode lifts at theta != 0 (magnetic
   frustration, the HarmonicDimension story), and spec(theta) == spec(2pi -
   theta) to machine precision -- charge conjugation: the antiparticle sees
   the reversed flux with an IDENTICAL spectrum ("same mass, opposite
   charge"), while the eigenVECTORS at +/-theta are complex conjugates.

2. r_state IS CHARGE-CONJUGATION-BLIND, twice over.
   (a) S3 absorption: MultiCobordism::residualOfTargetStateAgainstHarmonic
       minimizes over ALL permutations of the target components
       (MultiCobordism.cpp:220-239). conj{1,w,w^2} = {1,w^2,w} is itself a
       permutation, so r_state(singlet) == r_state(conjugate) on EVERY
       background, phases or not.
   (b) Real-span blindness: the periods are metric harmonics
       (EigenstateSynthesis.cpp:911), REAL vectors for any geometry; a
       least-squares residual against a real span cannot distinguish t from
       conj(t) -- so even a cyclic-only (Z3) relabeling stays blind here.
   Verified numerically on the engine's own periods.

3. THE FIX, DEMONSTRATED SYNTHETICALLY: with a COMPLEX period matrix and
   relabeling restricted to the cyclic group Z3 (the physical 'which quark is
   first' gauge freedom; the leftover S3/Z3 = Z2 transposition IS charge
   conjugation), the singlet and conjugate residuals split. C-honest readout
   = complex periods (U(1) holonomy entering k>=1) + Z3-only relabeling.

Sakharov reading: today the background (Im l^2 = 0 in every dump) AND the
objective (S3 quotient x real periods) are exactly C-symmetric, so no baryon
asymmetry can condense -- C violation must be added by hand or by phases.

Usage (repo root):
    python examples/cobordism/ctwist_probe.py [dump.json] [--out out.png]
"""
import argparse
import itertools
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tessera

cob = tessera.cobordism

DEFAULT_DUMP = "laplacian_dumps/run-20260811-121826/iter_0042.json"


def load_dump(path):
    with open(path) as f:
        d = json.load(f)
    st = tessera.observables.LiveComplex.load(
        d["cells"],
        {(u, v): complex(re, im) for u, v, re, im in d["squared_lengths"]},
        {int(v): t for v, t in d["vertex_times"].items()},
        d["dimensions"])
    return st, d


# ---------------------------------------------------------------------------
# residual replicas: S3 (engine behaviour) and Z3 (cyclic-only relabeling)
# ---------------------------------------------------------------------------

def _residual(Pt, t):
    c, *_ = np.linalg.lstsq(Pt, t, rcond=None)
    return float(np.linalg.norm(Pt @ c - t) ** 2)


def residual_min(Pt, target, perms):
    t = np.asarray(target, dtype=complex)
    return min(_residual(Pt, t[list(p)]) for p in perms)


def s3_perms(d):
    return list(itertools.permutations(range(d)))


def z3_perms(d):
    return [tuple((i + s) % d for i in range(d)) for s in range(d)]


def period_matrix_transposed(st, degree, target_dim):
    """The engine's periodMatrixTransposed (d x rank), zero-filled beyond the
    usable hole columns, built from EigenstateSynthesis.cyclePeriods."""
    holes = [list(h) for h in cob.MultiCobordism.emergent_holes(st, degree)]
    if not holes:
        return None, 0
    holes = holes[:target_dim]
    m = len(holes)
    P = np.asarray(cob.EigenstateSynthesis(st, degree).cyclePeriods(holes),
                   dtype=complex)
    dim = P.size // m
    P = P.reshape(dim, m)
    betti = list(cob.MultiCobordism.betti(st))
    rank = min(betti[degree], dim)
    Pt = np.zeros((target_dim, rank), dtype=complex)
    Pt[:m, :] = P[:rank, :].T
    return Pt, m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump", nargs="?", default=DEFAULT_DUMP)
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=81)
    args = ap.parse_args()
    out_png = args.out or os.path.join(os.path.dirname(args.dump),
                                       "ctwist_probe.png")

    st, meta = load_dump(args.dump)
    edges = st.getEdgeList().toVector()
    edge0 = edges[0]
    eid = (edge0.getSource().getId(), edge0.getTarget().getId())
    print(f"frame {args.dump}: {len(edges)} edges; threading flux through "
          f"edge {eid} (l^2 = {complex(edge0.getSquaredLength()):.3f})")

    # --- 1. flux sweep ----------------------------------------------------
    thetas = np.linspace(0.0, 2 * np.pi, args.steps)
    spectra = []
    for th in thetas:
        edge0.setPhase(float(th))
        H = cob.HodgeLaplacian(st)  # fresh: spectra cache per instance
        spectra.append(np.asarray(H.eigenvalues(0), dtype=float))
    spectra = np.array(spectra)               # (steps, N)
    zero_modes = (np.abs(spectra) < 1e-9).sum(axis=1)
    gaps = spectra[:, 1] - spectra[:, 0]

    # C-isospectrality: spec(theta) == spec(2pi - theta)
    iso_dev = float(np.abs(spectra - spectra[::-1]).max())

    # eigenvector conjugation at a sample flux: psi(2pi-th) ~ conj(psi(th))
    th_s = float(thetas[args.steps // 7])
    edge0.setPhase(th_s)
    V1 = np.asarray(cob.HodgeLaplacian(st).eigenvectors(0)).reshape(
        len(spectra[0]), -1)
    edge0.setPhase(2 * np.pi - th_s)
    V2 = np.asarray(cob.HodgeLaplacian(st).eigenvectors(0)).reshape(
        len(spectra[0]), -1)
    edge0.setPhase(0.0)  # restore
    overlaps = np.abs(np.einsum("ij,ij->j", V1.conj(), V2.conj()))
    conj_check = float(np.median(overlaps))

    print(f"flux sweep: iso-spectrality max|spec(th)-spec(2pi-th)| = {iso_dev:.2e}")
    print(f"zero modes over sweep: {zero_modes.min()}..{zero_modes.max()} "
          f"(at theta=0: {zero_modes[0]}; a timelike edge's negative weight is "
          f"already a pi-phase, so dispositions frustrate the constant mode "
          f"before any explicit flux)")
    print(f"eigenvector conjugation |<psi_i(th), conj(psi_i(-th))>| median = "
          f"{conj_check:.6f}  (1 = anti-mode is the conjugate mode)")

    # --- 2. readout blindness on the engine's own periods ------------------
    singlet = [complex(z) for z in cob.Proton.singlet()]
    conjugate = [z.conjugate() for z in singlet]
    r_engine_s = float(cob.MultiCobordism.r_state(st, 3, singlet))
    r_engine_c = float(cob.MultiCobordism.r_state(st, 3, conjugate))
    Pt, m = period_matrix_transposed(st, 3, len(singlet))
    im_P = float(np.abs(Pt.imag).max()) if Pt is not None else 0.0
    rows = []
    if Pt is not None:
        rows = [
            ("engine r_state (S3)", r_engine_s, r_engine_c),
            ("replica S3, engine periods",
             residual_min(Pt, singlet, s3_perms(3)),
             residual_min(Pt, conjugate, s3_perms(3))),
            ("replica Z3, engine periods",
             residual_min(Pt, singlet, z3_perms(3)),
             residual_min(Pt, conjugate, z3_perms(3))),
        ]
    # --- 3. synthetic complex periods: Z3 splits, S3 cannot ---------------
    rng = np.random.default_rng(7)
    P_syn = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    rows += [
        ("S3, synthetic COMPLEX periods",
         residual_min(P_syn, singlet, s3_perms(3)),
         residual_min(P_syn, conjugate, s3_perms(3))),
        ("Z3, synthetic COMPLEX periods",
         residual_min(P_syn, singlet, z3_perms(3)),
         residual_min(P_syn, conjugate, z3_perms(3))),
    ]
    print(f"\nengine periods: {m} hole(s); max|Im P| = {im_P:.2e} (real span)")
    print(f"{'readout':>34} {'singlet':>9} {'conjugate':>10} {'split?':>7}")
    for name, rs, rc in rows:
        print(f"{name:>34} {rs:9.4f} {rc:10.4f} "
              f"{'YES' if abs(rs - rc) > 1e-9 else 'no':>7}")

    # --- figure ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ((ax_b, ax_g), (ax_r, ax_t)) = plt.subplots(2, 2, figsize=(15, 9.5))
    fig.subplots_adjust(hspace=0.36, wspace=0.26, top=0.90)

    n_show = min(14, spectra.shape[1])
    for i in range(n_show):
        ax_b.plot(thetas, spectra[:, i], lw=1.1,
                  color=plt.cm.viridis(i / max(n_show - 1, 1)))
    ax_b.axvline(np.pi, color="0.6", ls=":", lw=0.8)
    ax_b.set_xlabel(r"flux  $\theta$  through edge "
                    f"{eid}")
    ax_b.set_ylabel(r"spec $L_0(\theta)$ (lowest modes)")
    ax_b.set_title(f"magnetic band flow — spec(θ) = spec(2π−θ) to "
                   f"{iso_dev:.1e}\n(C: antiparticle at −θ, same masses; "
                   f"ψ(−θ) = conj ψ(θ), median overlap {conj_check:.4f})",
                   fontsize=10)

    ax_g.plot(thetas, gaps, color="#4c72b0", lw=1.6, label=r"gap $\lambda_1-\lambda_0$")
    ax_g2 = ax_g.twinx()
    ax_g2.plot(thetas, zero_modes, color="#c44e52", lw=1.6, drawstyle="steps-mid",
               label="zero modes (HarmonicDimension)")
    ax_g2.set_ylim(-0.2, max(zero_modes.max(), 1) + 0.6)
    ax_g.axvline(np.pi, color="0.6", ls=":", lw=0.8)
    ax_g.set_xlabel(r"flux  $\theta$")
    ax_g.set_ylabel("spectral gap")
    ax_g2.set_ylabel("zero modes", color="#c44e52")
    h1, l1 = ax_g.get_legend_handles_labels()
    h2, l2 = ax_g2.get_legend_handles_labels()
    ax_g.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax_g.set_title("C-even gap response (min at θ=π); zero modes stay 0: the\n"
                   "timelike (π-phase) edges already frustrate the constant mode",
                   fontsize=10)

    names = [r[0] for r in rows]
    rs = np.array([r[1] for r in rows])
    rc = np.array([r[2] for r in rows])
    y = np.arange(len(rows))
    ax_r.barh(y - 0.18, rs, height=0.36, color="#4c72b0", label="singlet {1,ω,ω²}")
    ax_r.barh(y + 0.18, rc, height=0.36, color="#c44e52", label="conjugate")
    for yy, (a, b) in zip(y, zip(rs, rc)):
        if abs(a - b) > 1e-9:
            ax_r.annotate("SPLIT — C visible", xy=(max(a, b) + 0.05, yy),
                          va="center", fontsize=8, color="#2a7d2a",
                          fontweight="bold")
    ax_r.set_yticks(y, names, fontsize=8)
    ax_r.invert_yaxis()
    ax_r.set_xlabel(r"residual  $\|Pc - t\|^2$  (0 = fully carried, 3 = full leak)")
    ax_r.set_title("charge-conjugation blindness of the register readout\n"
                   "(only Z3 relabeling + COMPLEX periods distinguishes "
                   "matter from antimatter)", fontsize=10)
    ax_r.legend(fontsize=8, loc="lower right")

    ax_t.axis("off")
    ax_t.text(0.0, 0.98, "\n".join([
        "WHY r_state cannot see matter vs antimatter",
        "",
        "1) S3 absorption (MultiCobordism.cpp:220):",
        "   residual is minimized over ALL 3! relabelings of the target;",
        "   conj{1,ω,ω²} = {1,ω²,ω} is itself a relabeling  ⇒  identical",
        "   residuals on every background. The physical gauge freedom is",
        "   only CYCLIC (Z3); the leftover transposition S3/Z3 ≅ Z2 IS",
        "   charge conjugation — currently quotiented away.",
        "",
        "2) Real-span blindness (EigenstateSynthesis.cpp:911):",
        f"   periods = metric harmonics, real for any geometry (here",
        f"   max|Im P| = {im_P:.1e}); |Pc-t| = |conj(Pc-t)| = |Pc*-t*|  =>",
        "   r(t) = r(t*) for ANY target when P is real -- Z3 alone is not",
        "   enough. Complex periods need U(1) holonomy entering k >= 1",
        "   (today phases enter only the k = 0 magnetic Laplacian).",
        "",
        "Bonus: at theta = 0 the k=0 operator ALREADY has no zero mode --",
        "a timelike edge (l^2 < 0) is a pi-phase in disguise, so the",
        "causal dispositions themselves magnetically frustrate the",
        "constant mode. Flux just moves the frustration around.",
        "",
        "Sakharov reading: background (Im l^2 = 0 in every dump) and",
        "objective (S3 x real periods) are exactly C-symmetric -- no",
        "baryon asymmetry can condense until C is broken in one of them.",
    ]), fontsize=9.5, family="monospace", va="top")

    fig.suptitle(f"C-twist probe — {args.dump}", fontsize=13)
    fig.savefig(out_png, dpi=130)
    print(f"\nfigure -> {out_png}")


if __name__ == "__main__":
    main()
