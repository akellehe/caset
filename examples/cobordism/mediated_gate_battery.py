"""Regge-mediated gate-battery re-test of H1-H3 across beta (#249).

Milestone "Bulk Synthesis with Regge Action Mediation v0.1", chain step 5. Re-runs
the spectral gate battery of spectral_gate_realizability.py under the mediated
objective and sweeps the gravitational coupling beta.

The base layer builds a triangulated sphere (icosahedron `_ICO`), punches 3
triangular holes in it by surgery (`removeInteriorCell` of `_CLASS_HOLES`) so it
carries a 2-D space of harmonic 1-forms (ker L_1 = the "register"), and asks, per
gate U, whether U|psi_B> stays in that space (the Hodge residual -> 0). 13 gates do
(the charge-conserving set).

Mediation: opening a hole drives the residual down but RAISES the dual Regge action
|S_Regge| (computed on the holed sphere via ReggeSolver.dualReggeAction, #247:
|S| = 9.07, 10.43, 11.79, 13.15 for k = 0,1,2,3 holes open). So per gate and per
beta we pick the hole count k in {0..3} that minimizes

    F_beta(k) = residual_k(U) + beta * |S_Regge(k)|,

commit it, and record:

  H1  realized      residual at the chosen k < REALIZE (1e-9)
  H2  input fixed   the 9 register-edge geometries + the input periods _CP_IN are
                    byte-identical across every (gate, beta) run -- the synthesis
                    never perturbs the input boundary data
  H3  amplitude     the holonomy-charge leak |Sigma(U|psi_B>)| = |sum(U_block @
                    _CP_IN)|, computed explicitly from the gate matrix (NOT inferred
                    from residual -> 0), cross-checked against the spectral residual

plus r_U, |S_Regge|, and the chosen k.

beta = 0 reproduces the base layer (k = 3, the 13 charge-conserving gates). As beta
grows, the cost of the 3rd hole outweighs its residual benefit, the chosen k drops,
and gates contract out of the realizable set -- the headline realizable-gates-vs-beta
curve, plus the H3 deviation vs beta.

Reuses spectral_gate_realizability.py (Register, _gates, _ICO, _CLASS_HOLES,
_CP_IN, REALIZE, _surface, _betti1) + ReggeSolver.dualReggeAction. Results are
written as JSON under /tmp/cobordism/ (attach to the PR; not committed).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spectral_gate_realizability as base  # noqa: E402

import tessera  # noqa: E402

cob = tessera.cobordism

BETAS_DEFAULT = [0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0]
H3_TOL = 1e-9  # H3 amplitude tolerance (the realized leak is ~1e-15 in practice)
KMAX = len(base._CLASS_HOLES)  # 3 holonomy holes -> k in 0..3


def _regge_magnitude(st):
    """|S_Regge(W*)| on the (holed) sphere: the modulus of the dual Lorentzian Regge
    action (ReggeSolver.dualReggeAction, #247). Built in C++ so the facet/coface
    materialization stays consistent (the Python getFacets bindings copy)."""
    s = tessera.ReggeSolver(st, tessera.MatterConfiguration()).dualReggeAction()
    return float(abs(s))


def _kstage(k):
    """The icosahedron with the first k holonomy holes opened. Returns
    (spacetime, EigenstateSynthesis, cell list, |S_Regge|)."""
    st = base._surface(base._ICO)
    es = cob.EigenstateSynthesis(st, 1)
    for hole in base._CLASS_HOLES[:k]:
        es.removeInteriorCell(list(hole))
    cells = [tuple(int(v) for v in c) for c in es.cellSimplices()]
    return st, es, cells, _regge_magnitude(st)


def _build_stages():
    """Precompute the k = 0..KMAX stages once: each carries its complex, residual
    engine, cell order, |S_Regge|, b_1, and |V|. Reused for every gate."""
    canon = base.Register()  # the k=3 register: harmonic basis + orientation signs
    stages = []
    for k in range(KMAX + 1):
        st, es, cells, S = _kstage(k)
        stages.append({"k": k, "st": st, "es": es, "cells": cells, "S": S,
                       "b1": int(base._betti1(st)),
                       "nV": int(st.getVertexList().size())})
    return canon, stages


def _residual_at(canon, stage, U):
    """The gate's spectral residual on the k-hole sphere: build U|psi_B>'s output
    harmonic (in the k=3 register's basis), restrict it to this stage's cells, and
    apply the genuine metric Hodge L_1 residual -- exactly base.identity_anchor's
    per-k evaluation, generalized to any gate."""
    u_reg = np.asarray(U, dtype=complex)[1:4, 1:4]
    cp_out = u_reg @ base._CP_IN.astype(complex)
    psi_full = canon.harmonic_form(canon.sign * cp_out)
    by_cell = {canon.cells[i]: psi_full[i] for i in range(len(canon.cells))}
    psi = np.array([by_cell.get(c, 0.0) for c in stage["cells"]], dtype=complex)
    return float(stage["es"].residual([complex(z) for z in psi]))


def _charge_leak(U):
    """H3 amplitude, explicit: the holonomy-charge leakage |Sigma(U|psi_B>)| =
    |sum(U_block @ _CP_IN)|. 0 iff U conserves total holonomy charge (the all-ones
    covector is preserved) -- the algebraic statement of Z = <psi_A|U|psi_B>,
    independent of the residual."""
    u_reg = np.asarray(U, dtype=complex)[1:4, 1:4]
    return float(abs((u_reg @ base._CP_IN.astype(complex)).sum()))


def _register_edge_fingerprint(stage):
    """H2 witness: the (squared-length, phase) of every edge of the opened holes,
    plus the input periods _CP_IN. Held byte-identical across the whole sweep."""
    emap = {}
    for e in stage["st"].getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        emap[(min(a, b), max(a, b))] = (e.getSquaredLength().real, e.getPhase())
    reg = {str(k): emap[k] for k in base._REG_EDGES if k in emap}
    return reg, [float(x) for x in base._CP_IN]


def sweep(betas, gates=None):
    """Run the mediated battery sweep. Returns (rows, h2). rows: one record per
    (gate, beta). h2: the H2 fixed-input check (the same for all runs)."""
    canon, stages = _build_stages()
    gate_list = gates if gates is not None else base._gates()
    S_by_k = {st["k"]: st["S"] for st in stages}
    b1_by_k = {st["k"]: st["b1"] for st in stages}

    # H2: the input register-edge geometry + _CP_IN are identical across all stages
    # (the synthesis only opens interior holes; it never rewrites the input). Compare
    # every stage's register-edge fingerprint to k=0's.
    base_fp, base_cp = _register_edge_fingerprint(stages[0])
    h2_ok = True
    for st in stages:
        fp, cp = _register_edge_fingerprint(st)
        if fp != base_fp or cp != base_cp:
            h2_ok = False
    h2 = {"input_boundary_byte_fixed": bool(h2_ok),
          "register_edges": len(base_fp), "cp_in": base_cp}

    rows = []
    for name, U, fam in gate_list:
        res_by_k = {st["k"]: _residual_at(canon, st, U) for st in stages}
        leak = _charge_leak(U)
        for beta in betas:
            # F_beta(k) = residual_k + beta * |S_Regge(k)|; commit the minimizing k.
            kstar = min(res_by_k, key=lambda k: res_by_k[k] + beta * S_by_k[k])
            r = res_by_k[kstar]
            realized = bool(r < base.REALIZE)
            rows.append({
                "gate": name, "family": fam, "beta": float(beta),
                "k_star": int(kstar), "b1": int(b1_by_k[kstar]),
                "r_U": r, "S_regge": float(S_by_k[kstar]),
                "F_beta": float(r + beta * S_by_k[kstar]),
                "realizable": realized,            # H1
                "charge_leak": leak,               # H3 (explicit amplitude deviation)
                # H3: for a REALIZED gate the explicit amplitude must match (leak ~ 0);
                # vacuous for a gate mediation floored (we make no H3 claim about it).
                "h3_holds": bool((not realized) or (leak < H3_TOL)),
            })
    return rows, h2


def summarize(rows, betas):
    """Per-beta: realizable count, the realized-set identity, max H3 leak among
    realized gates, and whether every H3 cross-check held."""
    out = []
    for beta in betas:
        br = [r for r in rows if r["beta"] == beta]
        realized = sorted(r["gate"] for r in br if r["realizable"])
        max_leak_realized = max((r["charge_leak"] for r in br if r["realizable"]),
                                default=0.0)
        out.append({
            "beta": float(beta),
            "n_realizable": len(realized),
            "realized": realized,
            "max_charge_leak_realized": max_leak_realized,  # H3 vs beta
            "h3_holds": all(r["h3_holds"] for r in br),
        })
    return out


def make_plots(summary, rows, outdir):
    """The two #250 deliverable plots: (a) realizable-gates-vs-beta, (b) the H3
    amplitude error of each realized gate vs beta. matplotlib is imported lazily so
    a plain sweep never pulls it in. Returns the two file paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    betas = [s["beta"] for s in summary]
    x = list(range(len(betas)))
    labels = ["0" if b == 0 else f"{b:g}" for b in betas]

    # (a) realizable gate count vs beta -- the contraction curve.
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(x, [s["n_realizable"] for s in summary], marker="o", color="#1f77b4")
    ax.axhline(13, ls="--", color="grey", alpha=0.6, label="base layer (13 gates)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"$\beta$  (gravitational coupling)")
    ax.set_ylabel("# realizable gates")
    ax.set_title(r"Realizable gates vs $\beta$  ($F_\beta = r_U + \beta\,|S_{Regge}|$)")
    ax.set_ylim(-0.5, 14.0)
    ax.legend()
    fig.tight_layout()
    pa = os.path.join(outdir, "regge_mediated_realizable_vs_beta.png")
    fig.savefig(pa, dpi=140)
    plt.close(fig)

    # (b) H3 amplitude error per realized gate vs beta -- fidelity under mediation.
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for i, b in enumerate(betas):
        leaks = [r["charge_leak"] for r in rows if r["beta"] == b and r["realizable"]]
        if leaks:
            ax.scatter([i] * len(leaks), [max(v, 1e-18) for v in leaks],
                       s=20, color="#2ca02c", alpha=0.6,
                       label="realized gate" if i == 0 else None)
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 1e-8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(1e-15, ls="--", color="grey", alpha=0.6, label=r"$\sim 10^{-15}$ target")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$|Z - \langle\psi_A|U|\psi_B\rangle|$  (per realized gate)")
    ax.set_title(r"H3 amplitude error vs $\beta$")
    ax.legend()
    fig.tight_layout()
    pb = os.path.join(outdir, "regge_mediated_h3_vs_beta.png")
    fig.savefig(pb, dpi=140)
    plt.close(fig)
    return pa, pb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--betas", type=float, nargs="+", default=BETAS_DEFAULT)
    ap.add_argument("--canonical-only", action="store_true",
                    help="restrict to the 13-gate CANONICAL_SET (a fast slice)")
    ap.add_argument("--plots", metavar="DIR", default=None,
                    help="also write the two deliverable plots (#250) to DIR")
    ap.add_argument("--out", default="/tmp/cobordism/mediated_gate_battery.json")
    args = ap.parse_args()

    gates = None
    if args.canonical_only:
        keep = set(base.CANONICAL_SET)
        gates = [g for g in base._gates() if g[0] in keep]

    rows, h2 = sweep(args.betas, gates=gates)
    summary = summarize(rows, args.betas)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"betas": args.betas, "h2": h2, "summary": summary, "rows": rows},
                  f, indent=2)

    print(f"H2 input-boundary byte-fixed: {h2['input_boundary_byte_fixed']}")
    print(f"{'beta':>8}  {'#realizable':>11}  {'max H3 leak (realized)':>22}  "
          f"{'H3 holds':>9}")
    for s in summary:
        print(f"{s['beta']:>8}  {s['n_realizable']:>11}  "
              f"{s['max_charge_leak_realized']:>22.3e}  "
              f"{str(s['h3_holds']):>9}")
    base_set = next(s for s in summary if s["beta"] == 0.0)["realized"] \
        if 0.0 in args.betas else None
    if base_set is not None:
        print(f"\nbeta=0 realized set ({len(base_set)} gates): {base_set}")
    print(f"\nwrote {args.out}")

    if args.plots:
        pa, pb = make_plots(summary, rows, args.plots)
        print(f"wrote plots:\n  {pa}\n  {pb}")


if __name__ == "__main__":
    main()
