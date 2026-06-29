# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The emergent optimizer loop on a closed S⁴ (T5, #462).

Builds the topology of a cobordism EMERGENTLY. Per the design note (§2):

* **Host** — a bare closed S⁴ (`SimplexBoundarySphere(4)` refined for surgery room),
  with two **constructed interior input states** inserted.
* **Two inputs** — each solved *separately* into its own interior sub-complex whose
  **own `L_k` harmonic** represents that input. Topology emergent (no opened holes, no
  designated register). Tracked, held *effectively* fixed by its residual.
* **Output** — never constructed: the harmonic of the **entire** structure. The loop
  drives the free part until the whole complex's `L_k` harmonic matches the expected
  output.
* **Objective** — `F = ‖∇S_Regge‖² + Γ·r_U`, `r_U` a **three-term** residual:

      r_U = r(sub₁, input₁ | sub₁'s own L_k)
          + r(sub₂, input₂ | sub₂'s own L_k)
          + r(whole, output | whole L_k)

  Each term is a `residualForPeriods` of the expected state against the `L_k` harmonic
  read over the structure's *emergent* register, with the register **zero-filled** to
  the expected state's dimension (un-emerged slots = 0) and matched **relabeling-
  invariantly**. The register is *read off* the structure (`getBoundary`), never placed.
* **Two stages, one functional, in sequence (never together):**
  - **Stage 1** (combinatorial, fixed edge length) — greedy best-ΔF over random single
    Pachner + gated surgical cone-out/in, kept only by ΔF; the two inputs are *really
    held fixed* (moves can't touch their edges), so the free part is *whole minus the
    inputs*. `Δ‖∇S‖²` is the incremental T4 local delta.
  - **Stage 2** (continuous, geometric) — `relaxInterior` on every free edge (the inputs
    held representative, not frozen). Build-plan §9.2; not wired here.

The topology is FULLY EMERGENT, NEVER PRESCRIBED: random moves kept only by ΔF, no
target topology, no `b_k` goal, no recipes. See `docs/design/t5_emergent_optimizer_design.md`.
"""
import itertools
import random

import numpy as np

import tessera as T

cob = T.cobordism

# The register degree is a free parameter, explored k = 2 → 1 → 0 with no topological
# semantics attached (§ design note). The host is a 4-manifold, so d = 4.
_DIM = 4


def build_closed_s4(n_refine=20, seed=0):
    """A closed S⁴ (Betti [1,0,0,0,1]) with plenty of edges/vertices to optimize over:
    the bare `∂Δ⁵` sphere refined by `n_refine` PreGeometric stellar Pachner moves so
    surgery has somewhere to act (the minimal triangulation is too small)."""
    sig = T.Signature(_DIM, T.Lorentzian)
    st = T.Spacetime(T.Metric(True, sig), T.CDT, 1.0, 1.0, T.PREFERRED,
                     T.SimplexBoundarySphere(_DIM))
    st.build()
    for e in st.getEdgeList().toVector():
        e.setSquaredLength(1.0)
    applied = 0
    for s in range(seed, seed + n_refine * 4):
        mv = T.AddMove(st, s, False, T.PachnerMode.PreGeometric, False)
        if mv.propose() and mv.apply():
            applied += 1
        if applied >= n_refine:
            break
    for i, e in enumerate(st.getEdgeList().toVector()):
        e.setSquaredLength(1.0 + 0.01 * (i % 6))
    return st


def _top_tuple(s):
    return tuple(sorted(v.getId() for v in s.getVertices()))


def betti(st):
    return list(cob.ChainComplex.fromSpacetime(st).bettiNumbers())


# ===== the emergent register read + the residual (a function of the L_k harmonic and
# ===== the expected state — nothing imposed) =====
def emergent_holes(st, k):
    """The emergent k-register, **read off** the structure: the `(k+2)`-vertex tuples
    (removed `(k+1)`-simplices) all of whose drop-one facets are boundary facets — the
    cells whose boundary `k`-cycle the surgery exposed. A pure read of `getBoundary`;
    nothing is placed and no register is tracked."""
    bnd = {tuple(sorted(f)) for f in st.getBoundary()}
    if len(next(iter(bnd), ())) != k + 1:               # boundary facets must be k-cells
        return []
    verts = sorted({v for f in bnd for v in f})
    holes = set()
    for f in bnd:
        for v in verts:
            if v in f:
                continue
            tup = tuple(sorted(f + (v,)))
            facets = [tuple(x for j, x in enumerate(tup) if j != i)
                      for i in range(len(tup))]
            if all(ff in bnd for ff in facets):
                holes.add(tup)
    return sorted(holes)


def r_state(st, k, target):
    """The residual of the expected `target` state against the `L_k` harmonic of `st`,
    with the emergent register **zero-filled** to the target dimension and matched
    **relabeling-invariantly**:

    * `b_k = 0` (nothing emerged) → every slot zero → residual = ‖target‖² (full leak),
      so the loop has a gradient toward growing a register rather than `r_U` being
      undefined;
    * as cycles emerge and their periods align with the target, the leak falls → 0 iff
      the structure carries the state.

    Purely a function of `harmonicMatrix(k)` (via `cyclePeriods`) and the expected
    state; the register is read, never placed."""
    d = len(target)
    t = np.asarray(target, dtype=complex)
    bk = betti(st)[k]
    if bk == 0:
        return float(np.vdot(t, t).real)                 # zero-filled: full leak
    holes = emergent_holes(st, k)[:d]                    # up to d emergent slots
    if not holes:
        return float(np.vdot(t, t).real)
    periods = cob.EigenstateSynthesis(st, k).cyclePeriods([list(h) for h in holes])
    pmat = np.asarray(periods, dtype=complex).reshape(bk, len(holes))
    pd = np.zeros((bk, d), dtype=complex)                # zero-fill un-emerged slots
    pd[:, :pmat.shape[1]] = pmat
    best = float("inf")
    for perm in itertools.permutations(range(d)):        # relabeling-invariant match
        ts = t[list(perm)]
        c, *_ = np.linalg.lstsq(pd.T, ts, rcond=None)
        leak = pd.T @ c - ts
        best = min(best, float(np.vdot(leak, leak).real))
    return best


def _grad_norm2(st):
    return sum(abs(z) ** 2
               for z in T.ReggeSolver(st, T.MatterConfiguration()).actionGradientExact())


class EmergentOptimizer:
    """Greedy + random-restart optimizer of `F = ‖∇S_Regge‖² + Γ·r_U` with fully
    emergent topology and the three-term, asymmetric `r_U` of §2. The two inputs are
    constructed in place as interior sub-complexes (their own `L_k` harmonic represents
    each input) and tracked by vertex set; the output is the whole structure's harmonic.

    `input_targets` is a list of two expected-state period vectors; `output_target` the
    expected output. Nothing is frozen and no register is imposed."""

    def __init__(self, host, input_targets, output_target, degrees=(3,), gamma=1.0,
                 seed=0):
        self.st = host
        self.input_targets = [list(t) for t in input_targets]
        self.output_target = list(output_target)
        self.degrees = tuple(degrees)        # the register degrees r_U requires at once
        self._gate_k = max(self.degrees)     # dualComplexValid is the (degree-free) gate
        self.gamma = gamma
        self.rng = random.Random(seed)
        self._tol = 1e-9
        self.inputs = []        # [{'verts': frozenset, 'target': [...]}, ...]

    # ----- the constructed interior inputs -----
    def _sub_of(self, st, verts):
        """An input sub-complex: the cells of `st` entirely within `verts`, as its own
        complex so we can take its *own* `L_k` at each register degree."""
        cells = [list(c) for c in (_top_tuple(s) for s in st.getTopSimplices())
                 if set(c) <= verts]
        return T.Spacetime.fromCells(_DIM, cells, 1.0, 0.0) if cells else None

    def _r_input(self, inp, st):
        """The input's residual: its own-`L_k` representativeness summed over ALL register
        degrees (each degree's harmonic must carry the input)."""
        sub = self._sub_of(st, inp['verts'])
        if sub is None:
            tg = np.asarray(inp['target'])
            return len(self.degrees) * float(np.vdot(tg, tg).real)
        return sum(r_state(sub, k, inp['target']) for k in self.degrees)

    def construct_inputs(self, seeds, rounds=24):
        """Solve each input *separately* into an interior sub-complex whose own `L_k`
        harmonic represents it: a region-restricted move-solve (surgical cone-out/in on
        cells inside the region, kept only by Δ of that input's term) growing whatever
        emergent topology carries the target. `seeds` is the list of seed vertices; the
        region is the seed's neighbourhood. Nothing opened by hand."""
        for seed_v, target in zip(seeds, self.input_targets):
            verts = frozenset(
                v for c in (_top_tuple(s) for s in self.st.getTopSimplices())
                if seed_v in c for v in c)
            inp = {'verts': verts, 'target': target}
            r = self._r_input(inp, self.st)
            for _ in range(rounds):
                cells = [c for c in (_top_tuple(s) for s in self.st.getTopSimplices())
                         if set(c) <= verts]
                if not cells:
                    break
                cell = list(self.rng.choice(cells))
                snap = self._snapshot_of(self.st)
                if not cob.SurgicalCone(self.st).coneOut(cell)[0]:
                    continue
                ok, _why = cob.EigenstateSynthesis(self.st, self._gate_k).dualComplexValid()
                r_new = self._r_input(inp, self.st) if ok else float("inf")
                if ok and r_new < r - self._tol:
                    r = r_new
                else:
                    self._restore(self._build(snap))     # reject: restore exactly
            self.inputs.append(inp)
        return [self._r_input(i, self.st) for i in self.inputs]

    @property
    def _input_verts(self):
        out = set()
        for inp in self.inputs:
            out |= inp['verts']
        return out

    # ----- objective (asymmetric, summed over the register degrees) -----
    def r_u(self, st=None):
        """The three-term residual, summed over ALL register degrees: the whole's `L_k`
        harmonic must carry the output, and each input sub-complex's own `L_k` must carry
        its input, at **every** degree in `self.degrees`. Requiring (say) both `L₂` and
        `L₃` forces both a `b₂` and a `b₃` register to emerge."""
        st = st if st is not None else self.st
        total = sum(r_state(st, k, self.output_target) for k in self.degrees)  # output
        for inp in self.inputs:                                               # inputs
            total += self._r_input(inp, st)
        return total

    def objective(self):
        return _grad_norm2(self.st) + self.gamma * self.r_u(self.st)

    # ----- snapshot / restore (drift-free; never trust move rollback, #365/#371) -----
    def _snapshot_of(self, st):
        cells = [list(_top_tuple(s)) for s in st.getTopSimplices()]
        l2 = {}
        for e in st.getEdgeList().toVector():
            a, b = e.getSource().getId(), e.getTarget().getId()
            l2[(min(a, b), max(a, b))] = e.getSquaredLength()
        return (cells, l2)

    def _snapshot(self):
        return self._snapshot_of(self.st)

    def _build(self, snap):
        cells, l2 = snap
        st = T.Spacetime.fromCells(_DIM, cells, 1.0, 0.0)
        for e in st.getEdgeList().toVector():
            a, b = e.getSource().getId(), e.getTarget().getId()
            v = l2.get((min(a, b), max(a, b)))
            if v is not None:
                e.setSquaredLength(v)
        return st

    def _restore(self, st):
        self.st = st

    # ----- Stage 1: combinatorial moves; inputs kept REPRESENTABLE, not walled off -----
    def _random_spec(self, st):
        """A single RANDOM move on `st`. Moves may freely ADD to (or near) the input
        regions; the only thing held fixed is that no input vertex is removed (enforced
        in `_apply_spec`), so cone moves are proposed on any cell/facet."""
        kind = self.rng.choice(
            ["add", "remove", "flip", "iflip", "cone_out", "cone_in"])
        if kind in ("add", "remove", "flip", "iflip"):
            return (kind, self.rng.randrange(2 ** 31))
        tops = [_top_tuple(s) for s in st.getTopSimplices()]
        if not tops:
            return ("noop", None)
        if kind == "cone_out":
            return ("cone_out", list(self.rng.choice(tops)))
        verts = list(self.rng.choice(tops))
        drop = self.rng.randrange(len(verts))
        return ("cone_in", [v for i, v in enumerate(verts) if i != drop])

    def _apply_spec(self, st, spec):
        """Apply a move to `st`; True iff applied, gated by `dualComplexValid` (§3), and
        it does NOT remove an input vertex. Moves may add to the input regions — only the
        set of vertices representing each input state must persist (the residual keeps it
        representative), so the rejection is the narrow `input vertex disappeared`."""
        kind, p = spec
        if kind == "noop":
            return False
        if kind in ("add", "remove", "flip", "iflip"):
            cls = {"add": T.AddMove, "remove": T.RemoveMove,
                   "flip": T.FlipMove, "iflip": T.IFlipMove}[kind]
            mv = cls(st, p, False, T.PachnerMode.PreGeometric, False) \
                if cls is T.AddMove else cls(st, p, T.PachnerMode.PreGeometric, False)
            applied = bool(mv.propose() and mv.apply())
        elif kind == "cone_out":
            applied = cob.SurgicalCone(st).coneOut(p)[0]
        else:
            applied = cob.SurgicalCone(st).coneIn(p)[0]
        if not applied:
            return False
        live = {v for c in (_top_tuple(s) for s in st.getTopSimplices()) for v in c}
        if not (self._input_verts <= live):              # an input vertex was removed
            return False
        ok, _why = cob.EigenstateSynthesis(st, self._gate_k).dualComplexValid()
        return ok

    def _delta_f(self, base_solver, base_g2_edges, base_ru, base_cells, cand):
        """Incremental ΔF = Δ‖∇S‖²(touched edges) + Γ·Δr_U (T4, #461). The geometry term
        reads over the affected-edge index of the move (union of `affectedEdgesOfCells`
        on both legs); `r_U` (the three-term residual) is an exact before/after recompute
        — a global spectral quantity with no hinge-local delta."""
        cand_cells = {_top_tuple(s) for s in cand.getTopSimplices()}
        touched = [list(c) for c in base_cells ^ cand_cells]
        cand_solver = T.ReggeSolver(cand, T.MatterConfiguration())
        edges = sorted({tuple(p) for p in base_solver.affectedEdgesOfCells(touched)}
                       | {tuple(p) for p in cand_solver.affectedEdgesOfCells(touched)})
        edges = [list(p) for p in edges]
        d_grad = cand_solver.gradientNorm2OverEdges(edges) - base_g2_edges(edges)
        d_ru = self.r_u(cand) - base_ru
        return d_grad + self.gamma * d_ru

    def step(self, candidate_count=12):
        """One greedy step: draw `candidate_count` random single moves on the free part,
        score each by the incremental ΔF against the live base, commit the most-negative
        (if < 0). The winner's resulting complex is committed as-is; we never re-apply a
        spec (Pachner `propose` is non-deterministic across rebuilds). Returns ΔF."""
        snap = self._snapshot()
        base_solver = T.ReggeSolver(self.st, T.MatterConfiguration())
        base_g2_edges = base_solver.gradientNorm2OverEdges
        base_ru = self.r_u(self.st)
        base_cells = {_top_tuple(s) for s in self.st.getTopSimplices()}
        specs = [self._random_spec(self.st) for _ in range(candidate_count)]
        best_dF, best_snap = -self._tol, None
        for spec in specs:
            cand = self._build(snap)
            if not self._apply_spec(cand, spec):
                continue
            dF = self._delta_f(base_solver, base_g2_edges, base_ru, base_cells, cand)
            if dF < best_dF:
                best_dF = dF
                best_snap = self._snapshot_of(cand)
        if best_snap is not None:
            self._restore(self._build(best_snap))
            return best_dF
        return 0.0

    def run_stage1(self, max_iterations=200, candidate_count=12, patience=8):
        """Greedy best-ΔF steps until `patience` consecutive no-ops, re-seeding the
        random stream on each stall (restart). Returns the F trace."""
        trace = [self.objective()]
        stalls = 0
        for _ in range(max_iterations):
            dF = self.step(candidate_count)
            trace.append(trace[-1] + dF)
            if dF >= -self._tol:
                stalls += 1
                self.rng = random.Random(self.rng.randrange(2 ** 31))
                if stalls >= patience:
                    break
            else:
                stalls = 0
        return trace

    # ----- Stage 2: continuous geometric relaxation (every edge free) -----
    def relax_stage2(self, beta=1.0, max_iterations=40, alpha0=0.05):
        """Stage 2 (§6/§7): relax **every** edge squared-length toward a stationary point
        of `β‖∇S‖² + Γ·r_U`, re-opening the scale DOF Stage 1 froze. The inputs are held
        *representable*, not frozen — their residual terms are in the objective, so input
        edges relax like any other while staying representative.

        The squared lengths are **complex** (Lorentzian): each `ℓ²_e` carries a real and
        an imaginary part and both must be relaxed. `‖∇S‖²` is real, so its steepest
        descent over a complex `ℓ²` is the Wirtinger direction
        `∂‖∇S‖²/∂ℓ̄²_f = 2·conj(H)·g` (`actionGradientExact` `g` + `actionHessianExact`
        `H` — exact analytic, no finite differences), which reduces to the real gradient
        when the metric is Euclidean. A backtracking line search accepts only a decrease
        of the **full three-term** objective; `r_U` gates the step (its general-k gradient
        is not folded — the §9.2 precompute, matching `CobordismRelaxer::relaxInterior`'s
        k≥2 semantics). The conformal/scale runaway is diagnosed by this restoring force,
        never pinned to a boundary. Returns the F trace."""
        edges = self.st.getEdgeList().toVector()

        def full_f():
            return beta * _grad_norm2(self.st) + self.gamma * self.r_u(self.st)

        trace = [full_f()]
        alpha = alpha0
        for _ in range(max_iterations):
            rs = T.ReggeSolver(self.st, T.MatterConfiguration())
            g = np.asarray(rs.actionGradientExact(), dtype=complex)
            hmat = np.asarray(rs.actionHessianExact(), dtype=complex).reshape(len(g), len(g))
            grad = beta * 2.0 * (np.conj(hmat) @ g)          # complex ∂‖∇S‖²/∂ℓ̄²
            l2 = np.asarray([e.getSquaredLength() for e in edges], dtype=complex)
            f0, step, improved = trace[-1], alpha, False
            for _ls in range(24):
                l2n = l2 - step * grad
                re = np.clip(l2n.real, 0.05, 20.0)           # keep the real part bounded;
                for e, v in zip(edges, re + 1j * l2n.imag):  # carry the imaginary part
                    e.setSquaredLength(complex(v))
                f1 = full_f()
                if f1 < f0 - self._tol:
                    trace.append(f1)
                    alpha, improved = min(alpha * 1.3, 1.0), True
                    break
                step *= 0.5
            if not improved:
                for e, v in zip(edges, l2):                  # restore the best point
                    e.setSquaredLength(complex(v))
                break
        return trace
