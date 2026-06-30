# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""A Gym-style environment over the `MultiCobordism` objective search (#537).

The emergent cobordism optimizer drives a single boundary-seeded node toward a
stationary point of the **four-term** objective

    F = ‖∇S_Regge‖²  +  Γ · r_U          (extremize δS = 0; F ≥ 0 floors at δS = 0)

by alternating two engine stages — `run_stage1` (gated combinatorial surgery: greedy
best-of-n random `add/remove/flip/iflip/cone_out/cone_in` moves + a trap-door grow) and
`run_stage2` (Wirtinger geometric relaxation of the edge ℓ²). Today `Proton.build()`
orchestrates those stages with a FIXED schedule (init → evolve → relax). This env exposes
that orchestration as an RL problem: the agent learns WHICH macro-action to take and with
what parameters, while the engine keeps full ownership of the moves, the gating
(`dualComplexValid`), and the objective.

Nothing about the physics changes. The env only *drives* the existing public bindings:

  * **reset(seed)** seeds a fresh node on a single Δ⁴ simplex
    (`Proton.formation_node` / `recombination_node`).
  * **step(action)** applies ONE macro-action and returns ``(obs, reward, done, info)``.
  * Hybrid action = (discrete macro-move ∈ {GROW, EVOLVE, RELAX}, continuous params).
    GROW/EVOLVE call `run_stage1` with a *modest* `max_steps` (≈ 4–30 — one engine step
    does not grow topology, the grow-burst recovery + `patience` only act within a single
    `run_stage1` call). RELAX calls `run_stage2`.
  * **reward** = a numerically-stable, monotone transform of −ΔF (the per-step drop in the
    true objective toward its floor), plus a one-time terminal bonus when the node carries
    its target color state (`r_state → 0` over ≥ `target_holes` emergent holes).
  * **obs** = a feature vector (log F / ‖∇S‖² / r_U / r_state, hole count, Betti numbers,
    cell/edge/vertex counts, remaining budget, last move).

This module depends only on `tessera` + `numpy` (no PyTorch), so the env and its tests
run without the `rl` extra.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

import tessera

cob = tessera.cobordism

# Discrete macro-moves (the policy's categorical head). GROW/EVOLVE are stage-1 surgery
# passes (boundary-growing vs frozen-boundary evolution); RELAX is a stage-2 geometric
# relaxation pass.
GROW, EVOLVE, RELAX = 0, 1, 2
MOVE_NAMES = {GROW: "grow", EVOLVE: "evolve", RELAX: "relax"}
N_MOVES = 3

# Continuous parameters per action (the policy's Gaussian head), each interpreted in
# [0, 1] (the env squashes/clips): param[0] = intensity (how many engine steps/iters),
# param[1] = knob (the relaxation β; unused by GROW/EVOLVE).
PARAM_DIM = 2

# Observation layout (see `_observation`): 4 log-magnitude scalars + 5 Betti slots +
# hole count + 3 size counts + budget fraction + 3 last-move one-hot.
_BETTI_SLOTS = 5
OBS_DIM = 4 + _BETTI_SLOTS + 1 + 3 + 1 + N_MOVES


@dataclass
class _Box:
    """A minimal Gym-style continuous space descriptor (no gymnasium dependency)."""

    low: np.ndarray
    high: np.ndarray
    shape: tuple
    dtype: type = np.float32


@dataclass
class _HybridActionSpace:
    """A parameterized-action space: one discrete move + a continuous parameter vector,
    mirroring Gym's `Tuple((Discrete(n), Box(...)))` without requiring gymnasium."""

    n_moves: int = N_MOVES
    param_dim: int = PARAM_DIM
    param_box: _Box = field(
        default_factory=lambda: _Box(
            low=np.zeros(PARAM_DIM, np.float32),
            high=np.ones(PARAM_DIM, np.float32),
            shape=(PARAM_DIM,),
        )
    )


def _lerp(lo: float, hi: float, t: float) -> float:
    return lo + (hi - lo) * float(np.clip(t, 0.0, 1.0))


def _slog(x: float) -> float:
    """Signed-magnitude log feature: ``sign(x)·log1p(|x|)`` — compresses the wide
    dynamic range of F / ‖∇S‖² (thousands → ~0) into a network-friendly scale while
    keeping the (rare) sign of a drifted quantity."""
    return math.copysign(math.log1p(abs(float(x))), float(x))


def formation_node_factory(register_degree: int = 3, gamma: float = 50.0,
                           input_weight: float = 20.0) -> Callable[[int], "cob.MultiCobordism"]:
    """A ``seed -> MultiCobordism`` factory for Step B (formation, a small 2→1 merge):
    the diquark ``{1, ω}`` + the third quark ``{ω²}`` → the proton color singlet
    ``{1, ω, ω²}``, on a single Δ⁴ seed. The singlet is carried by the WHOLE cobordism
    (it is the natural `target_state` for the env), so `r_u` (the term in F) holds the two
    INPUTS while the singlet EMERGES on the bulk — exactly the emergent-target story the
    terminal carry bonus rewards."""
    def make(seed: int):
        proton = cob.Proton(seed=int(seed), register_degree=register_degree,
                            gamma=gamma, input_weight=input_weight)
        return proton.formation_node(int(seed))
    return make


def recombination_node_factory(register_degree: int = 3, gamma: float = 50.0,
                               input_weight: float = 20.0) -> Callable[[int], "cob.MultiCobordism"]:
    """A ``seed -> MultiCobordism`` factory for Step A (recombination, 2→2): two neutral
    q-q̄ pairs → a colored diquark ``{1, ω}`` ⊔ antidiquark ``{1, ω²}``. The diquark/
    antidiquark are LOCALIZED output blocks, so their residuals are already inside `r_u`
    (hence F); there is no single whole-cobordism target, so the matching env uses
    ``target_state=None`` (success = realizability `r_u` driven down)."""
    def make(seed: int):
        proton = cob.Proton(seed=int(seed), register_degree=register_degree,
                            gamma=gamma, input_weight=input_weight)
        return proton.recombination_node(int(seed))
    return make


class CobordismObjectiveEnv:
    """A Gym-style RL environment over one `MultiCobordism` node's objective search.

    The agent's job is to learn the SEARCH POLICY (which surgery/relaxation macro-action,
    with what parameters) that drives the true objective ``F = ‖∇S‖² + Γ·r_U`` to its
    stationary floor, and — for a node with a whole-cobordism target — to make that target
    color state emerge. The engine is unchanged: every topology move is drawn and gated
    (`dualComplexValid`, no pinned boundary vertex removed) inside `run_stage1`, and the
    objective is the engine's own `objective()` (reconstructed here from its two published
    components, `regge_action_gradient` + Γ·`r_u`, so the env reads F without a third
    redundant eigensolve).

    Action (parameterized / hybrid):
      ``(move, params)`` — ``move ∈ {GROW, EVOLVE, RELAX}`` and ``params ∈ ℝ^PARAM_DIM``
      (clipped to [0, 1]). The env may also be handed a flat array ``[move, p0, p1, …]``.
        * GROW   → ``run_stage1(max_steps, n_candidate_moves, patience, grow_boundaries=True)``
        * EVOLVE → ``run_stage1(max_steps, …, grow_boundaries=False)``
        * RELAX  → ``run_stage2(beta, max_iters, alpha0)``
      where ``max_steps``/``max_iters`` come from ``params[0]`` (intensity) and ``beta``
      from ``params[1]`` (knob), each mapped through the configured ranges.

    Reward: ``reward_scale · (slog(F_before) − slog(F_after))`` (a monotone, bounded
    surrogate for −ΔF whose episode sum telescopes to the total log-reduction of F), plus
    a one-time ``carry_bonus`` the step the target is first carried, plus two OPTIONAL dense
    proton-shaping terms (off by default): ``hole_reward_weight`` for progress toward
    ``target_holes`` (capped) and ``rstate_reward_weight`` for driving the singlet residual
    ``r_state`` down. The shaping is a *training signal* — it never changes F, the engine, or
    the carry verdict; with both weights 0 the reward is exactly the foundation's −ΔF drop.

    Episode ends (``done``) when the action budget ``max_actions`` is spent (truncation) or
    the target is carried (termination).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        node_factory: Optional[Callable[[int], "cob.MultiCobordism"]] = None,
        target_state: Optional[Sequence[complex]] = None,
        register_degree: int = 3,
        gamma: float = 50.0,
        max_actions: int = 8,
        grow_steps: tuple = (2, 8),
        evolve_steps: tuple = (2, 8),
        relax_iters: tuple = (1, 4),
        beta_range: tuple = (0.25, 2.0),
        alpha_range: tuple = (0.02, 0.2),
        n_candidate_moves: int = 6,
        patience: int = 15,
        carry_tol: float = 0.5,
        target_holes: int = 3,
        carry_bonus: float = 3.0,
        reward_scale: float = 1.0,
        hole_reward_weight: float = 0.0,
        rstate_reward_weight: float = 0.0,
        terminate_on_carry: bool = True,
        directed_grow: bool = False,
        cone_strategy: str = "greedy",
        cone_max_candidates: int = 80,
        cone_overshoot: int = 2,
        cone_probe_openers: int = 6,
    ):
        # Default target = the proton singlet on the formation (2→1) node. `gamma` MUST
        # match the factory's node so F = gradN2 + gamma·r_u is reconstructed exactly; the
        # default factory uses Proton's default gamma=50.0, so the defaults agree.
        self.node_factory = node_factory or formation_node_factory(
            register_degree=register_degree, gamma=gamma)
        self.target_state = (None if target_state is None
                             else [complex(z) for z in target_state])
        self.k = int(register_degree)
        self.gamma = float(gamma)
        self.max_actions = int(max_actions)
        self.grow_steps = grow_steps
        self.evolve_steps = evolve_steps
        self.relax_iters = relax_iters
        self.beta_range = beta_range
        self.alpha_range = alpha_range
        self.n_candidate_moves = int(n_candidate_moves)
        self.patience = int(patience)
        self.carry_tol = float(carry_tol)
        self.target_holes = int(target_holes)
        self.carry_bonus = float(carry_bonus)
        self.reward_scale = float(reward_scale)
        # Dense proton-shaping weights (a TRAINING SIGNAL only — they never touch F, the
        # engine, or the carry verdict). `hole_reward_weight` rewards each unit of progress
        # toward `target_holes` (capped, so over-growing past the target earns nothing);
        # `rstate_reward_weight` rewards driving the singlet residual `r_state` down. Both
        # default to 0, so the base env's reward stays exactly `−ΔF` (+ carry bonus) and the
        # foundation's reward contract (telescoping log-reduction of F) is unchanged.
        self.hole_reward_weight = float(hole_reward_weight)
        self.rstate_reward_weight = float(rstate_reward_weight)
        # When True (the #539 default), carrying the target ENDS the episode. The proton arc
        # is grow → evolve → RELAX, though, and the carry happens at the GROW stage — so the
        # carry profile sets this False to let the policy keep going (relax the geometry,
        # lower F) after forming the register, learning the full arc rather than stopping the
        # instant the holes appear. The carry verdict (read off the final state, as
        # `Proton.build()` does) is unaffected: once carried, GROW/EVOLVE early-break and
        # RELAX preserves the register, so the register persists to the episode's end.
        self.terminate_on_carry = bool(terminate_on_carry)
        # Directed cone-out probe (#546): when on, a GROW first runs the random-draw
        # `run_stage1` to build the bulk, then DELIBERATELY opens the register holes the
        # random draws left short of `target_holes` — choosing each cone-out cell by a gated
        # `SurgicalCone` probe (`directed_cone_out`) instead of hoping a random draw hits a
        # disjoint cell. Off by default (pure #539 random growth); the carry profile turns it
        # on. `cone_strategy` ∈ {"greedy", "bfs"}.
        self.directed_grow = bool(directed_grow)
        self.cone_strategy = str(cone_strategy)
        self.cone_max_candidates = int(cone_max_candidates)
        # GROW (directed) may OVER-open up to `target_holes + cone_overshoot` register holes;
        # EVOLVE (directed) then SELECTS the best `target_holes` by capping the worst (the
        # cone-in that leaves the lowest singlet r_state) — register selection, not just
        # growth (so the policy can drop a sub-optimal hole and keep a better one).
        self.cone_overshoot = int(cone_overshoot)
        # The cone-out probe stops scanning candidates once this many hole-openers are found
        # (interior-first ordering surfaces them up front), so the probe stays affordable.
        self.cone_probe_openers = int(cone_probe_openers)

        # Gym-style space descriptors (no gymnasium dependency).
        self.observation_space = _Box(
            low=np.full(OBS_DIM, -np.inf, np.float32),
            high=np.full(OBS_DIM, np.inf, np.float32),
            shape=(OBS_DIM,),
        )
        self.action_space = _HybridActionSpace()
        self.obs_dim = OBS_DIM
        self.n_moves = N_MOVES
        self.param_dim = PARAM_DIM

        # Episode state (populated by reset()).
        self.node = None
        self._seed = None
        self._steps_taken = 0
        self._F = math.nan
        self._last_move = -1
        self._carried = False
        self._last_metrics: dict = {}

    # ------------------------------------------------------------------ measurement
    def _metrics(self) -> dict:
        """The engine quantities for one state, read through the public bindings. F is
        reconstructed from its two published components (the SAME `objective()` value, but
        without the extra eigensolve a third `objective()` call would cost). `r_state` is
        only evaluated when a whole-cobordism target is set."""
        st = self.node.st
        gradN2 = float(cob.MultiCobordism.regge_action_gradient(st))
        rU = float(self.node.r_u(st))
        F = gradN2 + self.gamma * rU
        betti = list(cob.MultiCobordism.betti(st))
        holes = len(cob.MultiCobordism.emergent_holes(st, self.k))
        rstate = (float(cob.MultiCobordism.r_state(st, self.k, self.target_state))
                  if self.target_state is not None else math.nan)
        edges = st.getEdgeList().toVector()
        return {
            "F": F, "gradN2": gradN2, "rU": rU, "rstate": rstate, "holes": holes,
            "betti": betti, "n_vertices": len(st.getVertexList().toVector()),
            "n_edges": len(edges), "n_top_cells": len(st.getTopSimplices()),
        }

    def _is_carried(self, metrics: dict) -> bool:
        """Carried = the target color state is an L_k harmonic over enough emergent holes.
        With no whole-cobordism target (recombination), 'carried' instead means the
        realizability residual `r_u` has been driven essentially to zero."""
        if self.target_state is None:
            return metrics["rU"] < 1e-3
        return (metrics["holes"] >= self.target_holes
                and metrics["rstate"] < self.carry_tol)

    def _observation(self, metrics: dict) -> np.ndarray:
        betti = (metrics["betti"] + [0] * _BETTI_SLOTS)[:_BETTI_SLOTS]
        last_one_hot = [1.0 if self._last_move == m else 0.0 for m in range(N_MOVES)]
        rstate = metrics["rstate"]
        obs = [
            _slog(metrics["F"]),
            _slog(metrics["gradN2"]),
            _slog(metrics["rU"]),
            _slog(0.0 if math.isnan(rstate) else rstate),
            *[float(b) for b in betti],
            float(metrics["holes"]),
            metrics["n_vertices"] / 50.0,
            metrics["n_edges"] / 100.0,
            metrics["n_top_cells"] / 50.0,
            self._steps_taken / max(1, self.max_actions),
            *last_one_hot,
        ]
        return np.asarray(obs, dtype=np.float32)

    # ------------------------------------------------------------------ gym API
    def reset(self, seed: int = 0):
        """Seed a fresh node on a single Δ⁴ simplex and return the initial observation.
        Deterministic in the seed: the node is built from the fixed seed simplex with a
        seeded RNG, so the same `seed` yields the same starting state and observation."""
        self._seed = int(seed)
        self.node = self.node_factory(self._seed)
        self._steps_taken = 0
        self._last_move = -1
        self._carried = False
        self._last_metrics = self._metrics()
        self._F = self._last_metrics["F"]
        return self._observation(self._last_metrics)

    @staticmethod
    def _split_action(action):
        """Accept ``(move, params)``, a dict ``{"move", "params"}``, or a flat array
        ``[move, p0, p1, …]`` and return ``(int move, np.ndarray params)``."""
        if isinstance(action, dict):
            return int(action["move"]), np.asarray(action["params"], np.float32)
        if isinstance(action, (tuple, list)) and len(action) == 2 \
                and np.ndim(action[1]) >= 1:
            return int(action[0]), np.asarray(action[1], np.float32)
        flat = np.asarray(action, np.float32).ravel()
        return int(round(float(flat[0]))), flat[1:1 + PARAM_DIM]

    def step(self, action):
        """Apply ONE macro-action; return ``(obs, reward, done, info)``."""
        if self.node is None:
            raise RuntimeError("step() called before reset()")
        move, params = self._split_action(action)
        move = int(np.clip(move, 0, N_MOVES - 1))
        params = np.clip(np.asarray(params, np.float32).ravel(), 0.0, 1.0)
        intensity = float(params[0]) if params.size > 0 else 0.5
        knob = float(params[1]) if params.size > 1 else 0.5

        prev_metrics = self._last_metrics  # the state BEFORE this macro-action
        F_before = self._F
        engine_error = None
        try:
            if move == GROW:
                max_steps = int(round(_lerp(*self.grow_steps, intensity)))
                self.node.run_stage1(max_steps=max(1, max_steps),
                                     n_candidate_moves=self.n_candidate_moves,
                                     patience=self.patience, grow_boundaries=True)
                if self.directed_grow:
                    # Finish the register the random draws left short: deliberately open the
                    # remaining holes with the gated directed cone-out probe (#546).
                    self.directed_cone_out(self.cone_strategy)
            elif move == EVOLVE:
                max_steps = int(round(_lerp(*self.evolve_steps, intensity)))
                self.node.run_stage1(max_steps=max(1, max_steps),
                                     n_candidate_moves=self.n_candidate_moves,
                                     patience=self.patience, grow_boundaries=False)
                if self.directed_grow:
                    # Select the register: cap the worst hole(s) back toward target_holes,
                    # dropping a sub-optimal hole GROW may have over-opened (#546).
                    self.directed_cone_in(self.cone_strategy)
            else:  # RELAX
                max_iters = int(round(_lerp(*self.relax_iters, intensity)))
                beta = _lerp(*self.beta_range, knob)
                alpha0 = _lerp(*self.alpha_range, intensity)
                self.node.run_stage2(beta=beta, max_iters=max(1, max_iters), alpha0=alpha0)
        except Exception as exc:  # an engine stage failed: no-op this action, small penalty
            engine_error = repr(exc)

        self._last_move = move
        self._steps_taken += 1
        metrics = self._metrics()
        self._last_metrics = metrics
        F_after = metrics["F"]
        self._F = F_after

        # Reward = the dense −ΔF term (a monotone, bounded surrogate for the drop in the
        # true objective; its per-step values telescope to the episode's total log-reduction
        # of F) + the dense PROTON-SHAPING terms (a training signal that points the policy at
        # the carry outcome F alone is too flat to find — see __init__) + the one-time carry
        # bonus. The shaping weights default to 0, so the base contract (reward == −ΔF drop)
        # is recovered exactly when they are off.
        dF_term = self.reward_scale * (_slog(F_before) - _slog(F_after))

        # Hole-progress: reward closing the gap to `target_holes`, capped so over-growing
        # past 3 holes earns nothing (the proton wants exactly the 3-quark register). The
        # capped differences telescope to `w·(min(final,T) − min(initial,T))` over an episode.
        hole_term = 0.0
        if self.hole_reward_weight != 0.0:
            prev_h = min(int(prev_metrics.get("holes", 0)), self.target_holes)
            cur_h = min(int(metrics["holes"]), self.target_holes)
            hole_term = self.hole_reward_weight * float(cur_h - prev_h)

        # Singlet-residual descent: reward driving `r_state` toward 0 (only meaningful for a
        # whole-cobordism target). slog compresses the wide range (full leak ≈ ‖t‖² = 3 down
        # to ~0 when carried); the per-step values telescope to the total slog-drop of r_state.
        rstate_term = 0.0
        if self.rstate_reward_weight != 0.0 and self.target_state is not None:
            prev_r = prev_metrics.get("rstate", math.nan)
            cur_r = metrics["rstate"]
            if math.isfinite(prev_r) and math.isfinite(cur_r):
                rstate_term = self.rstate_reward_weight * (_slog(prev_r) - _slog(cur_r))

        error_term = -0.1 if engine_error is not None else 0.0  # discourage faulting moves

        carried_now = self._is_carried(metrics)
        # One-time bonus the step the target is FIRST carried (the proton criterion).
        carry_term = self.carry_bonus if (carried_now and not self._carried) else 0.0

        reward = dF_term + hole_term + rstate_term + error_term + carry_term

        terminated = carried_now and self.terminate_on_carry
        self._carried = carried_now
        truncated = self._steps_taken >= self.max_actions
        done = bool(terminated or truncated)

        info = {
            "move": move, "move_name": MOVE_NAMES[move],
            "F": F_after, "F_before": F_before, "delta_F": F_after - F_before,
            "gradN2": metrics["gradN2"], "rU": metrics["rU"], "rstate": metrics["rstate"],
            "holes": metrics["holes"], "betti": metrics["betti"],
            "n_top_cells": metrics["n_top_cells"], "n_edges": metrics["n_edges"],
            "carried": carried_now, "terminated": terminated, "truncated": truncated,
            "steps_taken": self._steps_taken, "engine_error": engine_error,
            "reward_terms": {"dF": dF_term, "hole": hole_term, "rstate": rstate_term,
                             "error": error_term, "carry": carry_term},
        }
        return self._observation(metrics), float(reward), done, info

    # ------------------------------------------------------------------ directed probe
    def _pinned_vertex_ids(self) -> set:
        """Every pinned input/output block vertex — a directed cone-out must never strand
        one (the engine's own move gate also forbids removing a pinned boundary vertex)."""
        pinned = set()
        for block in list(self.node.inputs) + list(self.node.outputs):
            pinned.update(int(v) for v in block.vertices)
        return pinned

    @staticmethod
    def _top_cell_tuples(st) -> list:
        """The current top cells as sorted vertex-id tuples — the `coneOut` input form."""
        return [sorted(int(v.getId()) for v in s.getVertices())
                for s in st.getTopSimplices()]

    def directed_cone_out(self, strategy: str = "greedy", max_open: int = 6) -> int:
        """Open register holes by a DIRECTED, gated cone-out probe (#546) instead of
        `run_stage1`'s random cone draws. `run_stage1` picks ``cone_out`` only 1/6 of the
        time, on a UNIFORMLY RANDOM cell, and keeps it only if it lowers F — but opening a
        new disjoint hole can transiently RAISE F, so greedy-ΔF growth misses it and the
        register stalls below `target_holes`. This probe instead enumerates candidate top
        cells, tries each with a gated `SurgicalCone.coneOut` (rolled back), and keeps the
        one that best advances the proton: opens a NEW emergent hole and (tie-break) lowers
        the singlet `r_state`.

        Candidates are ordered INTERIOR-FIRST — by how many of the cell's facets are already
        on the boundary, ascending. Coning out a fully-interior cell turns all of its facets
        into boundary at once, which is exactly what forms a new emergent-hole tuple; a
        boundary-adjacent cell does not (and only a few percent of cells are interior
        hole-creators, so a deliberate ordering matters far more than a random draw).
        ``strategy='bfs'`` additionally prefers cells disjoint from the existing holes
        (fewest shared vertices), so the holes grow well-separated. Every `coneOut` is gated
        on `dualComplexValid` and a pinned input/output vertex is never stranded (the
        candidate is rolled back if it would). Returns the number of holes opened; a no-op
        once `target_holes` already exist (the carry wants exactly the 3-quark register)."""
        st = self.node.st
        pinned = self._pinned_vertex_ids()
        cone = cob.SurgicalCone(st)
        opened = 0
        open_target = self.target_holes + max(0, self.cone_overshoot)
        for _ in range(int(max_open)):
            holes = cob.MultiCobordism.emergent_holes(st, self.k)
            if len(holes) >= open_target:  # over-open; EVOLVE's cone-in selects the best 3
                break
            hole_vertices = {v for hole in holes for v in hole}
            boundary = {tuple(sorted(int(v) for v in f)) for f in st.getBoundary()}
            hole_vertices = {int(v) for v in hole_vertices}

            def _order_key(cell):
                cellset = set(cell)
                # facets already on the boundary (interior cells = 0 → hole-creators first)
                n_boundary_facets = sum(
                    1 for i in range(len(cell))
                    if tuple(c for j, c in enumerate(cell) if j != i) in boundary)
                shared = len(cellset & hole_vertices)  # overlap with existing holes
                return (n_boundary_facets, shared) if strategy == "bfs" \
                    else (n_boundary_facets,)

            cells = sorted(self._top_cell_tuples(st), key=_order_key)[:self.cone_max_candidates]
            best = None  # (score, cell)
            openers_found = 0
            for cell in cells:
                ok, _reason = cone.coneOut(cell)
                if not ok:
                    continue
                current = {int(v.getId()) for v in st.getVertexList().toVector()}
                if not pinned.issubset(current):  # would strand a pinned vertex: skip
                    cone.rollback()
                    continue
                delta = len(cob.MultiCobordism.emergent_holes(st, self.k)) - len(holes)
                if delta > 0 and self.target_state is not None:
                    rstate = float(cob.MultiCobordism.r_state(
                        st, self.k, self.target_state))
                    score = (delta, -rstate)
                else:
                    score = (delta, 0.0)
                if best is None or score > best[0]:
                    best = (score, cell)
                cone.rollback()
                # Early stop: the interior-first ordering surfaces hole-creators up front, so
                # once a few openers are in hand the best is almost surely among them — bail
                # rather than scan the (mostly non-opening) tail. Keeps the probe affordable.
                if delta > 0:
                    openers_found += 1
                    if openers_found >= self.cone_probe_openers:
                        break
            if best is None or best[0][0] <= 0:  # nothing opens a new hole
                break
            ok, _reason = cone.coneOut(best[1])
            if not ok:
                break
            opened += 1
        return opened

    def directed_cone_in(self, strategy: str = "greedy", max_close: int = 6) -> int:
        """Refine the register by DIRECTED, gated cone-in (#546, register SELECTION): cap an
        existing emergent hole — `coneIn` on one of its boundary facets adds a cell that
        covers the facet, so that hole's tuple is no longer all-boundary and the hole closes
        (lowers `b_{d-1}`). The candidates surfaced FIRST are exactly the boundary facets of
        the current holes (the hole-CAPPING cone-ins), as first-class moves: this lets the
        policy DROP a sub-optimal register hole and keep a better one, rather than only ever
        growing structure. While more than `target_holes` holes exist (e.g. after GROW
        over-opened), the cap chosen is the one leaving the LOWEST singlet `r_state` — i.e.
        the WORST hole is dropped, selecting the best `target_holes` that carry the singlet.
        Gated on `dualComplexValid`; cone-in only adds a fresh vertex, so nothing pinned is
        stranded. Returns the number of holes capped (a no-op at/below `target_holes`)."""
        st = self.node.st
        cone = cob.SurgicalCone(st)
        closed = 0
        for _ in range(int(max_close)):
            holes = cob.MultiCobordism.emergent_holes(st, self.k)
            if len(holes) <= self.target_holes:  # keep exactly the target register
                break
            boundary = {tuple(sorted(int(v) for v in f)) for f in st.getBoundary()}
            # Hole-capping cone-ins, surfaced first: every drop-one facet of a current hole
            # that is on the boundary (capping it closes that hole).
            cap_facets = []
            seen = set()
            for hole in holes:
                for i in range(len(hole)):
                    facet = tuple(c for j, c in enumerate(hole) if j != i)
                    if facet in boundary and facet not in seen:
                        seen.add(facet)
                        cap_facets.append(list(facet))
            best = None  # (score, facet) — score = (-r_state) of the result; drop the worst
            for facet in cap_facets[:self.cone_max_candidates]:
                ok, _reason = cone.coneIn(facet)
                if not ok:
                    continue
                if len(cob.MultiCobordism.emergent_holes(st, self.k)) >= len(holes):
                    cone.rollback()  # did not actually close a hole
                    continue
                rstate = (float(cob.MultiCobordism.r_state(st, self.k, self.target_state))
                          if self.target_state is not None else 0.0)
                score = (-rstate,)
                if best is None or score > best[0]:
                    best = (score, facet)
                cone.rollback()
            if best is None:
                break
            ok, _reason = cone.coneIn(best[1])
            if not ok:
                break
            closed += 1
        return closed

    # ------------------------------------------------------------------ helpers
    def dual_complex_valid(self):
        """``(ok, reason)`` from `EigenstateSynthesis.dualComplexValid` on the CURRENT
        complex — the same manifold-with-boundary gate the engine applies to accept a
        topology move. Used by the tests to confirm the env never leaves an invalid
        complex behind (the gating the engine promises)."""
        return cob.EigenstateSynthesis(self.node.st, self.k).dualComplexValid()

    @property
    def metrics(self) -> dict:
        return dict(self._last_metrics)


def make_formation_env(**kwargs) -> CobordismObjectiveEnv:
    """The default small target: the formation (2→1) node carrying the proton singlet
    ``{1, ω, ω²}``. `target_holes`/`carry_tol` default to the proton thresholds."""
    register_degree = kwargs.pop("register_degree", 3)
    gamma = kwargs.pop("gamma", 50.0)
    input_weight = kwargs.pop("input_weight", 20.0)
    kwargs.setdefault("target_state", cob.Proton.singlet())
    return CobordismObjectiveEnv(
        node_factory=formation_node_factory(register_degree, gamma, input_weight),
        register_degree=register_degree, gamma=gamma, **kwargs)


def make_recombination_env(**kwargs) -> CobordismObjectiveEnv:
    """The recombination (2→2) node → a colored diquark ⊔ antidiquark. The output blocks
    are localized (their residuals are already in `r_u`/F), so there is no whole-cobordism
    target: success = `r_u` driven to ~0."""
    register_degree = kwargs.pop("register_degree", 3)
    gamma = kwargs.pop("gamma", 50.0)
    input_weight = kwargs.pop("input_weight", 20.0)
    kwargs.setdefault("target_state", None)
    return CobordismObjectiveEnv(
        node_factory=recombination_node_factory(register_degree, gamma, input_weight),
        register_degree=register_degree, gamma=gamma, **kwargs)
