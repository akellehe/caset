# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The emergent optimizer loop on a closed S⁴ (T5, #462).

Builds the topology of a cobordism EMERGENTLY: start from a closed S⁴ with plenty
of edges/vertices, hold the start/end states representative through the single
combined residual ``r_U``, and minimize the objective

    F = ‖∇S_Regge‖²  +  Γ · r_U          (extremize the action, δS = 0)

by **random, single** moves — Pachner (refine/flip) and gated surgical cone-out/in —
each kept **only** by ΔF (gated by the dual-lattice manifold check). The topology is
FULLY EMERGENT, NEVER PRESCRIBED: the optimizer carries no target topology, no `b_k`
goal, and no move recipes; the objective is the only thing that ever guides the
lattice, move-by-move (greedy + random restarts to escape local minima).

Composes the merged primitives: `ReggeSolver` (‖∇S‖² via `actionGradientExact`),
`EigenstateSynthesis` (`r_U = residualForPeriods` at register degree `k`, and the
`dualComplexValid` gate), the Pachner moves, and `SurgicalCone` (#469). See
`docs/design/t5_emergent_optimizer_design.md`.
"""
import random

import tessera as T

cob = T.cobordism

# The register degree is a free parameter, explored k = 2 → 1 → 0 with no topological
# semantics attached (§ design note). The host is a 4-manifold, so d = 4.
_DIM = 4


def build_closed_s4(n_refine=20, seed=0):
    """A closed S⁴ (Betti [1,0,0,0,1]) with plenty of edges/vertices to optimize over:
    the minimal `∂Δ⁵` sphere refined by `n_refine` PreGeometric stellar Pachner moves."""
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


class EmergentOptimizer:
    """Greedy + random-restart optimizer of `F = ‖∇S_Regge‖² + Γ·r_U` over the move
    set, with fully emergent topology. The fixed states are held representative by the
    single combined `r_U` term (`stateHoles`/`stateTargets`) — not frozen, not seeded
    as registers: as the bulk emerges, `r_U` → 0 only if the structure can carry every
    state at once. Nothing else is imposed."""

    def __init__(self, host, state_holes, state_targets, k=2, gamma=1.0, seed=0):
        self.st = host
        self.holes = [list(h) for h in state_holes]
        self.targets = [complex(t) for t in state_targets]
        self.k = k
        self.gamma = gamma
        self.rng = random.Random(seed)
        self._tol = 1e-9          # only accept a meaningfully-improving move

    # ----- objective -----
    def grad_norm2(self):
        rs = T.ReggeSolver(self.st, T.MatterConfiguration())
        return sum(abs(z) ** 2 for z in rs.actionGradientExact())

    def r_u(self):
        if not self.holes:
            return 0.0
        return cob.EigenstateSynthesis(self.st, self.k).residualForPeriods(
            self.holes, self.targets)

    def objective(self):
        return self.grad_norm2() + self.gamma * self.r_u()

    # ----- snapshot / restore (drift-free base for candidate evaluation) -----
    # Move rollback is not bit-exact for every move (RemoveMove drifts O(1), #365/#371),
    # so we never rely on it: each candidate is evaluated on a freshly REBUILT copy of
    # the base complex (fromCells is deterministic ⇒ bit-identical), and only the winner
    # is committed. No drift can accumulate.
    def _snapshot(self):
        cells = [list(_top_tuple(s)) for s in self.st.getTopSimplices()]
        l2 = {}
        for e in self.st.getEdgeList().toVector():
            a, b = e.getSource().getId(), e.getTarget().getId()
            l2[(min(a, b), max(a, b))] = e.getSquaredLength()
        return (cells, l2)

    def _restore(self, snap):
        cells, l2 = snap
        st = T.Spacetime.fromCells(_DIM, cells, 1.0, 0.0)
        for e in st.getEdgeList().toVector():
            a, b = e.getSource().getId(), e.getTarget().getId()
            v = l2.get((min(a, b), max(a, b)))
            if v is not None:
                e.setSquaredLength(v)
        self.st = st

    # ----- random single-move spec (the ONLY guidance is ΔF, applied below) -----
    def _random_spec(self):
        """A single RANDOM move, as a deterministic spec — `(kind, param)`. The move is
        chosen at random; nothing about target topology enters here."""
        kind = self.rng.choice(
            ["add", "remove", "flip", "iflip", "cone_out", "cone_in"])
        if kind in ("add", "remove", "flip", "iflip"):
            return (kind, self.rng.randrange(2 ** 31))
        tops = [_top_tuple(s) for s in self.st.getTopSimplices()]
        if not tops:
            return ("noop", None)
        if kind == "cone_out":
            return ("cone_out", list(self.rng.choice(tops)))   # a random top pentatope
        verts = list(self.rng.choice(tops))                     # random d-facet (cone-in)
        drop = self.rng.randrange(len(verts))
        return ("cone_in", [v for i, v in enumerate(verts) if i != drop])

    def _apply_spec(self, spec):
        """Apply a move spec to `self.st` and return True iff it applied AND passes the
        authoritative manifold gate `dualComplexValid` (spec §3) — the only structural
        condition imposed on the loop. Surgery is also gated internally (#469); this is
        the single uniform check over every move type."""
        kind, p = spec
        if kind == "noop":
            return False
        if kind in ("add", "remove", "flip", "iflip"):
            cls = {"add": T.AddMove, "remove": T.RemoveMove,
                   "flip": T.FlipMove, "iflip": T.IFlipMove}[kind]
            mv = cls(self.st, p, False, T.PachnerMode.PreGeometric, False) \
                if cls is T.AddMove else cls(self.st, p, T.PachnerMode.PreGeometric, False)
            applied = bool(mv.propose() and mv.apply())
        elif kind == "cone_out":
            applied = cob.SurgicalCone(self.st).coneOut(p)[0]
        else:
            applied = cob.SurgicalCone(self.st).coneIn(p)[0]
        if not applied:
            return False
        ok, _reason = cob.EigenstateSynthesis(self.st, self.k).dualComplexValid()
        return ok

    # ----- one greedy step over a random batch -----
    def step(self, n_candidates=12):
        """Draw `n_candidates` RANDOM single moves, evaluate ΔF for each on a fresh
        rebuilt copy of the base, and commit the single move with the most-negative ΔF
        (if any < 0). Returns the committed ΔF (0.0 = no improving candidate → no-op).

        The winning candidate's *resulting* complex is snapshotted and restored directly
        — we never re-apply a move spec (Pachner `propose` is not deterministic across
        rebuilds), so the committed state IS exactly the evaluated state."""
        snap = self._snapshot()
        f0 = self.objective()
        specs = [self._random_spec() for _ in range(n_candidates)]
        best_dF, best_snap = -self._tol, None
        for spec in specs:
            self._restore(snap)                          # fresh, bit-identical base
            if not self._apply_spec(spec):
                continue
            dF = self.objective() - f0
            if dF < best_dF:
                best_dF = dF
                best_snap = self._snapshot()             # the WINNING result, committed as-is
        if best_snap is not None:
            self._restore(best_snap)
            return best_dF
        self._restore(snap)
        return 0.0

    # ----- Stage 1: greedy combinatorial moves at fixed edge length + restarts -----
    def run_stage1(self, max_steps=200, n_candidates=12, patience=8):
        """Greedy best-ΔF steps until `patience` consecutive no-ops, restarting the
        random stream on each stall. Returns the F trace."""
        trace = [self.objective()]
        stalls = 0
        for _ in range(max_steps):
            dF = self.step(n_candidates)
            trace.append(trace[-1] + dF)
            if dF >= -self._tol:
                stalls += 1
                # random restart: re-seed the proposal stream (NOT the topology)
                self.rng = random.Random(self.rng.randrange(2 ** 31))
                if stalls >= patience:
                    break
            else:
                stalls = 0
        return trace
