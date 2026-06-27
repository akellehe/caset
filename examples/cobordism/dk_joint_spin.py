# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Entangled joint 3-fermion spin read — proton ½ vs Δ 3/2 (#489, part of #410).

#485 (`dk_composite_spin.py`) showed the **composite** total spin can't be read from a product
of independently-extracted per-hole spinors: a product floors at the mixture, and the proton's
`J²=¾` lives in an **entangled, mixed-symmetry** state. A first attempt (#488) to read it from a
*static* color singlet found the per-hole **flavor** index not independent — but that was an
artifact of representing the proton with too little structure: the three-quark entanglement
Fermi statistics ties to spin only arises when the three quarks are genuinely **interacted into
being**.

So the proton here is **built**, not read off a hand-made singlet, using the emergent
`MultiCobordism` engine (#491/#492): a **single co-optimized** pair-creation event

    3 neutral q-q̄ pairs  ⟶  [ proton , antiproton ]

with the diquark/antidiquark forming as emergent interior structure. The legs are co-optimized
in **one** system (interacted simultaneously); charge is conserved end to end (each input pair
is neutral, `Σ = 0`). The proton's three quarks are the **three emergent holes of the proton
output block** — carved out via `MultiCobordism.outputs[0].verts`, with the relaxed metric
copied over.

What this delivers, validated (`tests/cobordism/test_dk_joint_spin.py`):
  * the proton output block **carries the color singlet** (`r_state → 0`);
  * the per-hole **flavor** (DK charge) is **independent / distinguishable** across the three
    holes — the structure #488's two-quark read lacked.

What it does NOT deliver, and why (the honest, validated finding):
  * the composite `J²` sits at the **indefinite mixture (~9/4)**, NOT the proton ¾. This is not a
    bug — the `J²` operator here is validated exact (`proton-eigenstate → ¾`, `Δ → 15/4`,
    `product |uud⟩ → 7/4`). Every available readout reduces each hole to a single-qubit Bloch
    vector, so the three-quark spin state is a **product** — and a product provably cannot
    represent the proton's **entangled** mixed-symmetry ¾. The build creates the entanglement;
    the per-hole readout discards it.
  * Resolving ¾ therefore needs a **total-space spin operator** acting on the whole carried
    representative at once (the entangled three-quark state as one object), rather than a product
    of per-hole spinors — exactly the open item scoped as **#477**.

`J²` is read post-hoc, never a loop condition. Honest negative reported as-is.
"""
import importlib.util
import math
import os
import sys

import cmath
import numpy as np

import tessera

cob = tessera.cobordism
_W = cmath.exp(2j * math.pi / 3)

# The pair-creation initial state and the baryon/antibaryon targets.
_PAIRS = [[1, -1, 0], [1, 0, -1], [0, 1, -1]]   # three neutral q-q̄ pairs (Σ = 0)
_PROTON = [1, _W, _W * _W]                       # color singlet
_ANTIPROTON = [1, _W * _W, _W]                   # conjugate singlet

# Clean spin-½ Pauli operators (the validated J² works in (C²)³, not the 4-component
# Dirac space, whose spin sector is two degenerate doublets).
_PAULI = [np.array([[0, 1], [1, 0]], complex),
          np.array([[0, -1j], [1j, 0]]),
          np.array([[1, 0], [0, -1]], complex)]
_S = [p / 2 for p in _PAULI]
_I2 = np.eye(2, dtype=complex)

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    sys.path.insert(0, _HERE)
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


eo = _load("emergent_optimizer")
cs = _load("dk_composite_spin")
_TT = eo._top_tuple
_KDIM = 4


# ----- build -----
def build_pair_creation(seed, n_refine=20, rounds=24, stage1_steps=80,
                        stage1_patience=12, stage2_iters=30):
    """One co-optimized `MultiCobordism`: 3 neutral pairs ⟶ [proton, antiproton],
    interacted simultaneously. `opt.st` is the live geometry; `opt.outputs[0]`/`[1]`
    are the proton/antiproton blocks."""
    host = eo.build_closed_s4(n_refine=n_refine, seed=seed)
    opt = cob.MultiCobordism(host, _PAIRS, [_PROTON, _ANTIPROTON], degrees=[3],
                             gamma=1.0, seed=seed)
    sv = [v.getId() for v in host.getVertexList().toVector()]
    opt.construct_inputs(sv[:3], rounds=rounds)
    opt.construct_outputs(sv[3:5], rounds=rounds)
    opt.run_stage1(max_steps=stage1_steps, n_candidates=10, patience=stage1_patience)
    opt.relax_stage2(beta=1.0, max_iters=stage2_iters)
    return opt


def block_subcomplex(st, verts):
    """The boundary block's own sub-complex (cells entirely inside `verts`), with the
    parent's **relaxed** edge metric copied over (a unit-metric rebuild would make the
    spinor frames degenerate). None if the block has < 2 cells."""
    vs = set(verts)
    cells = [list(_TT(s)) for s in st.getTopSimplices() if set(_TT(s)) <= vs]
    if len(cells) < 2:
        return None
    sub = tessera.Spacetime.fromCells(_KDIM, cells, 1.0, 0.0)
    parent = {tuple(sorted((e.getSource().getId(), e.getTarget().getId()))):
              e.getSquaredLength() for e in st.getEdgeList().toVector()}
    for e in sub.getEdgeList().toVector():
        key = tuple(sorted((e.getSource().getId(), e.getTarget().getId())))
        if key in parent:
            e.setSquaredLength(parent[key])
    sub.materializeFacets()
    return sub


# ----- flavor (independent per-hole index — the structure #488 lacked) -----
def per_hole_flavor(sub, holes):
    """Per-hole Dirac–Kähler flavor charge over the proton block's quark holes."""
    dk = cob.DiracKahler(sub)
    es = cob.EigenstateSynthesis(sub, 3)
    return np.array([dk.charge(dk.lift(3, list(es.carriedRepresentative([list(h)], [1.0]))))
                     for h in holes])


def flavor_spread(charges):
    """Relative spread of the per-hole flavor charges (0 ⇒ degenerate / not independent)."""
    charges = np.asarray(charges, float)
    return float((charges.max() - charges.min()) / (abs(charges.mean()) + 1e-12))


# ----- composite spin J² (validated operator on clean spin-½ qubits) -----
def spinor_to_qubit(s4):
    """Reduce a 4-component Dirac spinor to a clean spin-½ qubit via its Bloch vector
    `n_a = ⟨s|S_a|s⟩` (the spin direction), then the C² state polarized along `n`."""
    s4 = np.asarray(s4, complex)
    nrm = (s4.conj() @ s4).real
    if nrm < 1e-30:
        return np.array([1, 0], complex)
    n = np.array([(s4.conj() @ cs._SG[a] @ s4).real / nrm for a in range(3)])
    if np.linalg.norm(n) < 1e-9:
        return np.array([1, 0], complex)
    m = sum((n / np.linalg.norm(n))[a] * _PAULI[a] for a in range(3))
    w, v = np.linalg.eigh(m)
    return v[:, int(np.argmax(w.real))]


def j2_three_qubit(psi8):
    """`⟨J²⟩` of a three-spin-½ state in (C²)³. Validated: proton eigenstate → ¾,
    Δ → 15/4, product |uud⟩ → 7/4."""
    st = psi8 / np.linalg.norm(psi8)

    def si(a, i):
        ops = [_I2, _I2, _I2]
        ops[i] = _S[a]
        return np.kron(np.kron(ops[0], ops[1]), ops[2])

    sa = [sum(si(a, i) for i in range(3)) for a in range(3)]
    j2 = sum(sa[a] @ sa[a] for a in range(3))
    return float((st.conj() @ j2 @ st).real)


def _kron(*a):
    out = a[0]
    for x in a[1:]:
        out = np.kron(out, x)
    return out


def composite_j2(sub, holes, joint=True):
    """Composite spin `J²` of the proton block's three quark holes, as a **product** of
    the per-hole qubits (`joint=True` uses the color-correlated joint spinors). Sits at
    the mixture by construction — see the module docstring and #477."""
    spinors = (cs.joint_spinors(sub, holes, _TT) if joint
               else cs.emergent_spinors(sub, holes, _TT))
    qubits = [spinor_to_qubit(s) for s in spinors]
    return j2_three_qubit(_kron(*qubits))


# ----- read a (anti)proton leg -----
def read_proton(opt, block_index=0):
    """Read the (anti)proton leg off its output block. Returns a dict with the block
    residual (singlet carried?), the quark holes, the per-hole flavor + its spread, and
    the composite `J²` (joint & per-hole product reads). None if no 3-hole register."""
    sub = block_subcomplex(opt.st, list(opt.outputs[block_index].verts))
    if sub is None:
        return None
    holes = cob.MultiCobordism.emergent_holes(sub, 3)
    target = _PROTON if block_index == 0 else _ANTIPROTON
    out = {"n_holes": len(holes),
           "block_residual": cob.MultiCobordism.r_state(sub, 3, target)}
    if len(holes) < 3:
        return out
    h3 = holes[:3]
    out["flavor_charges"] = per_hole_flavor(sub, h3)
    out["flavor_spread"] = flavor_spread(out["flavor_charges"])
    out["j2_joint"] = composite_j2(sub, h3, joint=True)
    out["j2_product"] = composite_j2(sub, h3, joint=False)
    out["_sub"], out["_holes"] = sub, h3
    return out


def build_converged_proton(seeds=range(3, 60), max_residual=1.0, **build_kw):
    """Scan `seeds`, returning the first (opt, read, seed) whose proton block carries the
    singlet (≥ 3 holes and `block_residual ≤ max_residual`). None if none converge."""
    for s in seeds:
        opt = build_pair_creation(s, **build_kw)
        rd = read_proton(opt, 0)
        if rd and rd["n_holes"] >= 3 and rd["block_residual"] <= max_residual:
            return opt, rd, s
    return None


def main():
    print("Proton built by simultaneous pair creation — composite spin read (#489)\n")
    found = build_converged_proton()
    if not found:
        print("No converged 3-hole proton block in the seed range (honest negative).")
        return
    _opt, rd, seed = found
    print(f"converged proton (seed {seed}): holes={rd['n_holes']} "
          f"block_residual={rd['block_residual']:.3e}  (singlet carried)")
    print(f"  flavor charges = {np.round(rd['flavor_charges'], 4)}  "
          f"spread = {rd['flavor_spread']:.3f}  (>0 ⇒ independent per-hole flavor ✓)")
    print(f"  composite J² (joint)   = {rd['j2_joint']:.4f}")
    print(f"  composite J² (product) = {rd['j2_product']:.4f}")
    print("  proton ¾=0.75, Δ 15/4=3.75 — the product readout sits at the mixture; the")
    print("  entangled ¾ needs the total-space spin operator (#477).")


if __name__ == "__main__":
    main()
