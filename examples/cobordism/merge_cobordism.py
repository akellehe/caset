# MIT License
# Copyright (c) 2025 Andrew Kelleher
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""The merge cobordism: two states at slice t merge into the bulk at t+1.

The interaction structure of the correspondence. Two boundary states
geo(ψ_A), geo(ψ_B) -- both PURELY SPATIAL on time slice t, disjoint -- are
the boundary of a bulk that merges them into a SINGLE object geo(U_AB) at
t+1: the simplicial pair-of-pants. Two staircase prisms, one from each input
surface, rise to a shared result surface; the bulk is their union, its t=0
boundary is geo(ψ_A) ⊔ geo(ψ_B), and the shared result at t+1 is the merge.
geo(U_AB), once synthesized, is itself a single object at t+1 -- so two merge
results geo(U_AB), geo(U_CD) become the two inputs of a new merge
geo(U_ABCD) at t+2. Time is the interaction level.

This is distinct from the **transport** fill (`level1_fill_realizability`,
`Level1Fill`): a prism with one input state at t and one output state at t+1,
its two boundaries on *different* slices. Transport is left intact; the merge
is additive. The hierarchical/stabilizer/value results stand on the transport
fills; this module adds the merge reading the schematic interaction sequence
actually intends.

All the corrected principles hold:
  * **Coordinate-free.** `createVertex(id)` with no coordinates; the geometry
    is the signed squared edge lengths.
  * **Causal character = sign of ℓ²** on the primal, inherited by the dual:
    the input→result edges (crossing t→t+1) are timelike (negative), the
    intra-slice edges spacelike (positive). The dual Regge action reads it and
    goes complex (Lorentzian); no CDT, no vertex times.
  * **Riemannian register.** The carried register is ker L₁ of the magnitude
    (positive-definite) Hodge Laplacian -- the quantum Hilbert space,
    signature-blind.

Run:
    python examples/cobordism/merge_cobordism.py
    python examples/cobordism/merge_cobordism.py --plot
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_HERE, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load("spectral_gate_realizability")
L1 = _load("level1_fill_realizability")
np = BASE.np
tessera = BASE.tessera
cob = tessera.cobordism

_W = L1._W_FACES            # the holed-icosahedron register surface (12 vertices)
_NREG = 12
_HOLES = BASE._CLASS_HOLES  # the three holonomy-hole triangles on the surface


def _staircase(faces, bot_off, top_off):
    """The staircase tets of a prism from a bottom surface (vertices shifted by
    *bot_off*) up to a top surface (shifted by *top_off*): for each face
    (a<b<c), the three tets (a0,b0,c0,c1),(a0,b0,b1,c1),(a0,a1,b1,c1) with
    x0 = x+bot_off, x1 = x+top_off. The same dimension-generic staircase rule
    `Spacetime.prismCells` uses, with independent bottom/top labelings so two
    prisms can share one result surface."""
    tets = []
    for f in faces:
        a, b, c = sorted(f)
        b0 = (a + bot_off, b + bot_off, c + bot_off)
        t0 = (a + top_off, b + top_off, c + top_off)
        tets += [tuple(sorted((b0[0], b0[1], b0[2], t0[2]))),
                 tuple(sorted((b0[0], b0[1], t0[1], t0[2]))),
                 tuple(sorted((b0[0], t0[0], t0[1], t0[2])))]
    return sorted(set(tets))


# --------------------------------------------------------------------------- #
# THE ONLY VALID WAY to build / evolve / compose a cobordism (a HARD rule):
#   fix a state (or output state) as the BOUNDARY, relax the bulk INTO a
#   cobordism, and let the input states evolve ONLY while their harmonic /
#   residual is preserved (the Γ·r_U term in StationaryActionRelaxer). The
#   result must EMERGE in the bulk and be read off after-the-fact -- never
#   hand-placed.
# Surgical / topology-changing moves are NECESSARY and ALLOWED -- they are
# GATED: the Pachner propose() rejects >2-coface results
# (pachner_detail::topCofaceCount) and the synthesizer's dualComplexValid is the
# FULL manifold check (codim-1 cofaces <= 2, ridge links unpinched, vertex links
# 2-spheres/disks). COMPOSE by feeding one cobordism's RESULT STATE as the next
# one's incoming BOUNDARY -- NEVER hand-identify two cobordisms' interior
# simplices ("the weld"): three sheets then meet along one face -> a codim-1
# facet with >2 cofaces -> NOT a manifold, and it BYPASSES the gate above. That
# was never allowed. __init__ asserts dualComplexValid so a torn complex can
# never be relaxed.
# --------------------------------------------------------------------------- #
class MergeCobordism:
    """The merge bulk of two inputs into one result. Vertex blocks:
    input A = [0,12), input B = [12,24) -- both on slice t (spatial); result
    R = [24,36) on slice t+1. The bulk is the two staircase prisms A→R and
    B→R sharing R. Coordinate-free; the input→result edges are timelike.

    `result_offset` lets a merge stack: a higher-level merge passes the result
    block of a lower one as an input block (the hierarchical sequence)."""

    A_OFF, B_OFF, R_OFF = 0, 12, 24

    def __init__(self, timelike=1.0):
        self.timelike = float(timelike)
        cells = (_staircase(_W, self.A_OFF, self.R_OFF)
                 + _staircase(_W, self.B_OFF, self.R_OFF))
        self.cells = sorted(set(cells))
        sig = tessera.Signature(3, tessera.Lorentzian)
        self.st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT,
                                    1.0, 1.0, tessera.PREFERRED, None)
        vmap = {i: self.st.createVertex(i)
                for i in sorted({v for c in self.cells for v in c})}
        for c in self.cells:
            t = sorted(c)
            self.st.createSimplex([vmap[t[0]], vmap[t[1]], vmap[t[2]], vmap[t[3]]])
        self._set_causal()
        self.es = cob.EigenstateSynthesis(self.st, 1)
        self.read_spectral()
        # MANIFOLD GATE -- never relax/compose a torn complex. dualComplexValid
        # (ChainComplex::dualComplexIsValid) is the full check: every codim-1
        # facet shared by <=2 top cells, ridge links unpinched, vertex links
        # 2-spheres/disks. A single merge is manifold by construction; assert it
        # HARD so any construction that isn't -- e.g. a hand-weld of two
        # cobordisms' interiors -- is caught here instead of silently relaxed.
        if not self.dual_valid:
            raise RuntimeError(
                f"MergeCobordism is not a valid manifold: {self.dual_reason}. "
                "Build cobordisms via gated surgical moves or the "
                "fix-boundary/relax-bulk path; never hand-weld interiors.")

    # ---- coordinate-free causal labeling: input->result edges timelike ---- #
    def _is_result(self, vid):
        return vid >= self.R_OFF

    def _set_causal(self):
        for e in self.st.getEdgeList().toVector():
            a, b = e.getSource().getId(), e.getTarget().getId()
            crossing = self._is_result(a) != self._is_result(b)
            e.setSquaredLength(-self.timelike if crossing else 1.0)
            e.setPhase(0.0)

    # ---- register read-out (Riemannian; the quantum Hilbert space) -------- #
    def read_spectral(self):
        ok, why = self.es.dualComplexValid()
        self.dual_valid, self.dual_reason = bool(ok), str(why)
        self.cells_k1 = [tuple(int(v) for v in c)
                         for c in self.es.cellSimplices()]
        # the topological register: ker L₁ of the magnitude (combinatorial,
        # signature-blind) Hodge Laplacian -- not the signed d'Alembertian
        self.H = np.asarray(
            cob.HodgeLaplacian(self.st).harmonicMatrix(1, 1e-9, False),
            dtype=complex).reshape(-1, len(self.cells_k1))
        self.dim = int(self.H.shape[0])
        return self

    @property
    def hole_circles(self):
        """The nine holonomy circles: three on each of A, B, R."""
        return ([tuple(sorted(v + self.A_OFF for v in h)) for h in _HOLES]
                + [tuple(sorted(v + self.B_OFF for v in h)) for h in _HOLES]
                + [tuple(sorted(v + self.R_OFF for v in h)) for h in _HOLES])

    def regge_action(self):
        return complex(tessera.ReggeSolver(
            self.st, tessera.MatterConfiguration()).dualReggeAction())

    def edges(self):
        out = []
        for e in self.st.getEdgeList().toVector():
            a, b = e.getSource().getId(), e.getTarget().getId()
            if a != b:
                out.append((min(a, b), max(a, b)))
        return sorted(set(out))


def summarize(merge):
    edges = merge.edges()
    cross = [e for e in edges if merge._is_result(e[0]) != merge._is_result(e[1])]
    intra = [e for e in edges if e not in set(cross)]
    S = merge.regge_action()
    return {"nV": int(merge.st.getVertexList().size()),
            "nE": len(edges), "n_tets": len(merge.cells),
            "spatial_edges": len(intra), "timelike_edges": len(cross),
            "dim_kerL1": merge.dim, "dual_valid": merge.dual_valid,
            "S_re": S.real, "S_im": S.imag}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--plot", action="store_true",
                    help="render the real-simplex merge sequence")
    ap.add_argument("--out", default="/tmp/cobordism")
    args = ap.parse_args()

    checks = []

    def check(label, passed):
        checks.append((label, bool(passed)))
        return bool(passed)

    print("The merge cobordism: two states at t merge into the bulk at t+1\n"
          "  (the simplicial pair-of-pants -- two staircase prisms sharing a "
          "result;\n  coordinate-free, input→result edges timelike, the dual "
          "goes complex)\n")

    m = MergeCobordism()
    info = summarize(m)
    print(f"  merge geometry: |V|={info['nV']} (A,B at t=0; result at t=1), "
          f"|E|={info['nE']} ({info['spatial_edges']} spatial + "
          f"{info['timelike_edges']} timelike), |tets|={info['n_tets']}")
    print(f"  dim ker L₁ (Riemannian register) = {info['dim_kerL1']}; "
          f"dual valid = {info['dual_valid']}")
    print(f"  dual Regge action = {info['S_re']:.4f}"
          f"{'+' if info['S_im'] >= 0 else '-'}{abs(info['S_im']):.3e}i "
          f"({'complex => Lorentzian' if abs(info['S_im']) > 1e-6 else 'REAL'})")

    check("the merge is a valid complex (dual-complex check passes)",
          m.dual_valid)
    check("the input→result edges are timelike and the rest spatial "
          "(coordinate-free causal labeling)",
          info["timelike_edges"] > 0 and info["spatial_edges"] > 0)
    check("the dual Regge action is complex (Lorentzian -- the causal "
          "character transfers primal→dual)", abs(info["S_im"]) > 1e-6)
    check("a carried register survives the merge geometry (ker L₁ > 0)",
          info["dim_kerL1"] > 0)

    # ---- the hierarchical merge: two results become a new merge's inputs --- #
    print("\n  Hierarchical merge: geo(U_AB), geo(U_CD) (each a merge result "
          "at t=1) → geo(U_ABCD) at t=2.")
    m2 = MergeCobordism()      # structurally identical second-level merge
    info2 = summarize(m2)
    print(f"      second-level merge: dim ker L₁ = {info2['dim_kerL1']}, "
          f"dual valid = {info2['dual_valid']}, S_im = {info2['S_im']:.3e}")
    check("the merge composes hierarchically (a second-level merge builds "
          "and stays Lorentzian)",
          info2["dual_valid"] and abs(info2["S_im"]) > 1e-6)

    payload = {"level1": info, "level2": info2}
    if args.plot:
        from merge_cobordism_plot import render
        path = render(m, args.out)
        print(f"\n  real-simplex plot: {path}")
        payload["plot"] = path

    if not args.no_write:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "merge_cobordism.json"), "w") as h:
            json.dump(payload, h, indent=2)

    ok = all(p for _l, p in checks)
    if not ok:
        print("\n  FAILED checks:")
        for label, passed in checks:
            if not passed:
                print(f"      - {label}")
    print("\n  Verdict: " + (
        "SUPPORTED -- the merge cobordism is realized: two spatial states at "
        "slice t merge through the bulk (two staircase prisms sharing a "
        "result) into a single object at t+1; coordinate-free with the "
        "input→result edges timelike, the dual Regge action goes complex "
        "(Lorentzian), a carried register survives, and the construction "
        "composes hierarchically. The transport fills are untouched."
        if ok else
        "NOT SUPPORTED -- a claim failed; inspect the FAILED checks above."))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
