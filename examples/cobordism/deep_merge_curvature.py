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

"""Deep-merge curvature profile: the charge is localized, and so is the curvature
it induces.

A **deep merge** over a geodesically-subdivided holed register surface. Where the
canonical merge (`merge_cobordism`) is two distance-shells deep, the subdivided
surface puts *many* graph-distance shells between the holonomy worldtubes -- where
the carried register (the matter) sits -- and the far bulk. Level 2 gives seven
shells (d = 0..6), enough to read a falloff. Two emergent, **measured** (not
imposed) results:

  * **The matter source falls off.** The carried register's density |h|^2 -- the
    charge's stress-energy -- is concentrated at the worldtube and decays
    monotonically with distance from it (~65x over six shells at level 2). The
    charge is localized, and emergently so: we open a holonomy worldtube and read
    how the register concentrates around it; no profile is put in by hand.

  * **The curvature it induces is ultra-local.** Source the worldtube (compress
    its timelike edges -- the matter's pull, the #312 emergent-dual selection
    resolved spatially) and the deficit responds only in the worldtube's immediate
    neighborhood (d <= 1); it is *exactly zero* beyond. This is correct Regge
    physics -- the deficit at a hinge depends only on the tets sharing that hinge,
    so a static sourced perturbation cannot propagate. The framework curves the
    geometry **right at the charge**.

What this is NOT: a long-range (e.g. 1/r) potential. Weak-field bending is a
linear-response phenomenon of the *field equation* -- the Regge gradient is the
deficit (Schlafli identity), and the linearized sourced solve is the correct tool
-- not a property of this static geometry. That boundary is stated, not crossed:
the static deep merge shows the charge and its curvature are co-localized, no more.

All the merge principles hold: coordinate-free (`createVertex(id)`, no vertex
times); causal character = sign of l^2 (input->result edges timelike), inherited
by the complex Sorkin dual; the carried register is ker L_1 of the magnitude
(signature-blind) Hodge Laplacian. The subdivision tracks each holonomy hole onto
the **central child** of its hole face, preserving the three-class register on the
finer surface (the equivariant re-triangulation idiom).

Run:
    python examples/cobordism/deep_merge_curvature.py
    python examples/cobordism/deep_merge_curvature.py --level 2 --plot
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict, deque

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


MC = _load("merge_cobordism")          # the merge machinery (_staircase, causal)
np = MC.np
tessera = MC.tessera
cob = MC.cob
BASE = MC.BASE
_ICO = BASE._ICO
_CLASS_HOLES = BASE._CLASS_HOLES


# ---- the subdivided register surface, tracking the holonomy holes ------------ #
def _subdivide_tracked(faces, holes):
    """One geodesic (1->4) subdivision -- matching `spectral_gate_realizability.
    _subdivide`'s midpoint id scheme -- that also follows each tracked hole onto
    the CENTRAL CHILD of its face (the triangle of the three edge midpoints). The
    central-child chain keeps the three holonomy holes vertex-disjoint on the
    finer surface, so the three-class register survives subdivision."""
    nxt = [max(v for f in faces for v in f) + 1]
    mid = {}

    def m(a, b):
        key = (min(a, b), max(a, b))
        if key not in mid:
            mid[key] = nxt[0]
            nxt[0] += 1
        return mid[key]

    out, central = [], {}
    for (a, b, c) in faces:
        ab, bc, ca = m(a, b), m(b, c), m(c, a)
        out += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        central[tuple(sorted((a, b, c)))] = tuple(sorted((ab, bc, ca)))
    return out, [central[tuple(sorted(h))] for h in holes]


def deep_surface(level):
    """The icosahedron subdivided `level` times, with the three holonomy holes
    carried onto their central children. Returns (holed_faces, holes, n_reg):
    the holed register surface, the three hole triangles, and the vertex count
    (12, 42, 162 at levels 0, 1, 2)."""
    faces = [tuple(f) for f in _ICO]
    holes = [tuple(sorted(h)) for h in _CLASS_HOLES]
    for _ in range(level):
        faces, holes = _subdivide_tracked(faces, holes)
    holeset = set(tuple(sorted(h)) for h in holes)
    holed = [tuple(sorted(f)) for f in faces if tuple(sorted(f)) not in holeset]
    n_reg = max(v for f in faces for v in f) + 1
    return holed, sorted(holeset), n_reg


class DeepMerge:
    """A merge cobordism over a subdivided holed register. Vertex blocks: input
    A = [0, n_reg), input B = [n_reg, 2*n_reg) on slice t; result R = [2*n_reg,
    3*n_reg) on slice t+1. The bulk is the two staircase prisms A->R and B->R
    sharing R (the simplicial pair-of-pants). Coordinate-free; the input->result
    edges are timelike. `level` sets the subdivision depth -- and thus the number
    of graph-distance shells from the holonomy worldtubes to the far bulk."""

    def __init__(self, level=2, timelike=1.0):
        self.level = int(level)
        self.timelike = float(timelike)
        self.holed, self.holes, self.n_reg = deep_surface(self.level)
        self.A, self.B, self.R = 0, self.n_reg, 2 * self.n_reg
        cells = (MC._staircase(self.holed, self.A, self.R)
                 + MC._staircase(self.holed, self.B, self.R))
        self.cells = sorted(set(cells))
        sig = tessera.Signature(3, tessera.Lorentzian)
        self.st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT,
                                    1.0, 1.0, tessera.PREFERRED, None)
        vmap = {i: self.st.createVertex(i)
                for i in sorted({v for c in self.cells for v in c})}
        for c in self.cells:
            t = sorted(c)
            self.st.createSimplex([vmap[t[0]], vmap[t[1]], vmap[t[2]], vmap[t[3]]])
        # the nine holonomy circles: three holes on each of A, B, R
        self.circles = [tuple(sorted(v + off for v in h))
                        for off in (self.A, self.B, self.R) for h in self.holes]
        self.hole_vs = {v for c in self.circles for v in c}
        self._emap = {}
        for e in self.st.getEdgeList().toVector():
            a, b = e.getSource().getId(), e.getTarget().getId()
            self._emap[(min(a, b), max(a, b))] = e
        self.set_worldtube(self.timelike)            # uniform reference geometry
        self.es = cob.EigenstateSynthesis(self.st, 1)
        self.dist = self._shells()

    # ---- coordinate-free causal labeling --------------------------------- #
    def _is_result(self, vid):
        return vid >= self.R

    def _is_worldtube_edge(self, a, b):
        """A timelike (result-crossing) edge incident to a holonomy cycle -- the
        worldtube, where the carried register sits."""
        return ((a in self.hole_vs) or (b in self.hole_vs))

    def set_worldtube(self, s_wt, s_bulk=None):
        """Set the timelike edge scales: worldtube edges to `s_wt`, the remaining
        bulk timelike edges to `s_bulk` (default = the reference scale). Spatial
        edges stay at +1. Coordinate-free: only signed squared lengths are set."""
        s_bulk = self.timelike if s_bulk is None else float(s_bulk)
        for (a, b), e in self._emap.items():
            e.setPhase(0.0)
            if self._is_result(a) != self._is_result(b):
                wt = self._is_worldtube_edge(a, b)
                e.setSquaredLength(-(float(s_wt) if wt else s_bulk))
            else:
                e.setSquaredLength(1.0)
        self.st.materializeFacets()

    # ---- distance shells from the charge --------------------------------- #
    def _shells(self):
        """BFS graph distance from the holonomy-cycle (worldtube) vertices over
        the merge 1-skeleton: d = 0 at the charge, growing into the bulk."""
        adj = defaultdict(set)
        for (a, b) in self._emap:
            if a != b:
                adj[a].add(b)
                adj[b].add(a)
        dist = {v: 0 for v in self.hole_vs}
        dq = deque(self.hole_vs)
        while dq:
            u = dq.popleft()
            for w in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1
                    dq.append(w)
        return dist

    def _edge_shell(self, a, b):
        return min(self.dist.get(a, 10**9), self.dist.get(b, 10**9))

    @property
    def n_shells(self):
        return max(self.dist.values()) + 1

    # ---- the carried register (Riemannian; the matter) ------------------- #
    def harmonic_matrix(self):
        """The dim x |C_1| harmonic 1-forms of the magnitude (signature-blind)
        Hodge L_1 -- the carried register basis, in cell order."""
        cells_k1 = self.es.cellSimplices()
        H = np.asarray(cob.HodgeLaplacian(self.st).harmonicMatrix(1, 1e-9, False),
                       dtype=complex).reshape(-1, len(cells_k1))
        return H, [tuple(int(v) for v in c) for c in cells_k1]

    def dim_register(self):
        return int(self.harmonic_matrix()[0].shape[0])

    def source_profile(self):
        """The matter-density source profile: the carried register's weight
        |h|^2 = sum_i |H[i, e]|^2 per 1-cell, averaged over the cells in each
        distance shell. The charge's stress-energy by distance from the worldtube
        -- emergent (read off the register), not imposed."""
        H, cells_k1 = self.harmonic_matrix()
        tot = defaultdict(float)
        cnt = defaultdict(int)
        for j, c in enumerate(cells_k1):
            d = self._edge_shell(c[0], c[1])
            tot[d] += float(np.sum(np.abs(H[:, j]) ** 2))
            cnt[d] += 1
        return {d: tot[d] / max(cnt[d], 1) for d in sorted(tot)}

    # ---- curvature (deficit) by shell ------------------------------------ #
    def curvature_profile(self):
        """Mean Lorentzian deficit (real part -- timelike-hinge curvature is
        strictly real) per distance shell on the current geometry."""
        self.st.materializeFacets()
        bins = defaultdict(list)
        for s in self.st.getSimplices():
            vs = [v.getId() for v in s.getVertices()]
            if len(vs) == 2:
                bins[self._edge_shell(vs[0], vs[1])].append(
                    s.lorentzianDeficitAngle().real)
        return {d: float(np.mean(v)) for d, v in sorted(bins.items())}

    def curvature_response(self, s_wt):
        """The matter's curvature response by shell: deficit(worldtube sourced to
        `s_wt`) - deficit(reference), per distance shell. Confined to d <= 1 by
        the ultra-locality of Regge curvature."""
        self.set_worldtube(self.timelike)
        ref = self.curvature_profile()
        self.set_worldtube(s_wt)
        mat = self.curvature_profile()
        self.set_worldtube(self.timelike)            # restore the reference
        return {d: mat[d] - ref[d] for d in sorted(ref)}

    def regge_action(self):
        return complex(tessera.ReggeSolver(
            self.st, tessera.MatterConfiguration()).dualReggeAction())


def summarize(dm, s_wt=0.6):
    H, _ = dm.harmonic_matrix()
    src = dm.source_profile()
    resp = dm.curvature_response(s_wt)
    S = dm.regge_action()
    ds = sorted(src)
    local = [d for d in resp if abs(resp[d]) > 1e-9]
    return {
        "level": dm.level,
        "n_reg": dm.n_reg,
        "nV": int(dm.st.getVertexList().size()),
        "n_tets": len(dm.cells),
        "n_shells": dm.n_shells,
        "dim_register": int(H.shape[0]),
        "S_re": S.real, "S_im": S.imag,
        "source": src,
        "source_falloff": (src[ds[0]] / src[ds[-1]]) if src[ds[-1]] else float("inf"),
        "response": resp,
        "response_support": max(local) if local else -1,
        "s_wt": s_wt,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--level", type=int, default=2,
                    help="subdivision depth (0=12v/2 shells, 1=42v, 2=162v/7 shells)")
    ap.add_argument("--s-wt", type=float, default=0.6,
                    help="worldtube timelike scale for the matter response")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default="/tmp/cobordism")
    args = ap.parse_args()

    checks = []

    def check(label, passed):
        checks.append((label, bool(passed)))
        return bool(passed)

    print("Deep-merge curvature profile: the charge is localized, and so is the "
          "curvature it induces\n  (a merge over a subdivided holed register -- "
          "many distance shells from the worldtube to the bulk)\n")

    dm = DeepMerge(level=args.level)
    info = summarize(dm, s_wt=args.s_wt)
    print(f"  deep merge (level {info['level']}): register {info['n_reg']}v -> "
          f"merge {info['nV']}v, {info['n_tets']} tets; "
          f"{info['n_shells']} distance shells")
    print(f"  carried register: dim ker L_1 = {info['dim_register']}; dual Regge "
          f"action = {info['S_re']:.2f}{'+' if info['S_im'] >= 0 else '-'}"
          f"{abs(info['S_im']):.2f}i (Lorentzian)\n")

    print("  matter source |h|^2(d) -- the charge's stress-energy by distance "
          "(emergent, not imposed):")
    for d, v in info["source"].items():
        bar = "#" * max(1, int(round(60 * v / max(info["source"].values()))))
        print(f"      d={d}: {v:.5f}  {bar}")
    print(f"    falloff d=0 -> far: {info['source_falloff']:.1f}x\n")

    print(f"  matter curvature response  d(d) = deficit(s_wt={info['s_wt']}) - "
          f"deficit(ref):")
    for d, v in info["response"].items():
        print(f"      d={d}: {v:+.4f}" + ("   (sourced)" if abs(v) > 1e-9 else ""))
    print(f"    nonzero only out to d = {info['response_support']} "
          f"(ultra-local: Regge curvature is a function of the local tets)\n")

    ds = sorted(info["source"])
    check("the deep merge carries the three-class register (ker L_1 = 2) on the "
          "subdivided surface", info["dim_register"] == 2)
    check("the dual Regge action is complex (Lorentzian -- the causal character "
          "transfers primal->dual)", abs(info["S_im"]) > 1e-6)
    check(f"the substrate is deep (>= 4 distance shells; got {info['n_shells']})",
          info["n_shells"] >= 4)
    check("the matter source is localized at the worldtube (|h|^2 falls off with "
          "distance from the charge)",
          info["source"][ds[0]] > info["source"][ds[-1]] and
          info["source_falloff"] > 2.0)
    check("the matter curves the geometry at the charge (the deficit responds at "
          "the worldtube)", abs(info["response"][ds[0]]) > 1e-9)
    check("the curvature response is ultra-local (no propagating tail: the "
          "deficit is unchanged beyond d = 1)",
          0 <= info["response_support"] <= 1)

    if args.plot:
        from deep_merge_curvature_plot import render
        path = render(info, args.out)
        print(f"  profile plot: {path}\n")
        info["plot"] = path

    if not args.no_write:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "deep_merge_curvature.json"), "w") as h:
            json.dump({k: v for k, v in info.items()
                       if k not in ("source", "response")}
                      | {"source": {str(k): v for k, v in info["source"].items()},
                         "response": {str(k): v for k, v in info["response"].items()}},
                      h, indent=2)

    ok = all(p for _l, p in checks)
    if not ok:
        print("  FAILED checks:")
        for label, passed in checks:
            if not passed:
                print(f"      - {label}")
    print("\n  Verdict: " + (
        "SUPPORTED -- on a deep merge the charge's stress-energy (the carried "
        "register |h|^2) is localized at the worldtube and falls off with "
        "distance, and the curvature it induces is co-localized: the deficit "
        "responds at the charge and is exactly unchanged beyond its immediate "
        "neighborhood. The framework curves the geometry right at the charge. A "
        "propagating long-range potential is a linear-response phenomenon of the "
        "field equation, not of this static geometry -- and is not claimed."
        if ok else
        "NOT SUPPORTED -- a claim failed; inspect the FAILED checks above."))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
