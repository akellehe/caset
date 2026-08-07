# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Real-time animation of the **ProtonIngredients** build — the emergent arm (#555, #585).

`Proton.cpp` is the canonical line in the sand; the emergent arm prepares **only the
initial state** and reads everything else off the relaxed whole. The default drive is the
**joint three-pair node** (`ProtonIngredients.joint_node`): ONE `MultiCobordism` whose
inputs are the three Z₃-symmetric neutral q-q̄ pairs `{1,−1,0} ⊔ {0,1,−1} ⊔ {−1,0,1}`
(each Σ = 0 — the only prepared content, fixed for the whole build) and whose
output-target list is EMPTY, so the objective is `F = ‖∇S‖² + Γ·Σᵢ r_U(inputᵢ)` and the
proton — the pre-registered expectation is a baryon with a conjugate partner — must
emerge from just those pairs. No diquark, no bare quark, no intermediate is imposed.

`--two-step` drives the reference oracle instead (kept for A-vs-B comparison):

  * **Step A — recombination** (`ProtonIngredients.recombination_node`, the composed
    canonical `Proton`'s node verbatim): two neutral q-q̄ pairs ⟶ a colored diquark
    `{1, ω}` ⊔ anti-diquark `{1, ω²}`;
  * **Step B — formation, nothing pinned** (`ProtonIngredients.formation_node`): the
    ideal diquark `{1, ω}` + third quark `{ω²}` inputs on a fresh single-Δ⁴ seed — the
    imposed intermediate is exactly why this is an oracle and not the arm.

The charts are **identical to `multicobordism_animation.py`'s** — the same
[traces | primal | dual-spatial (Re ε) | dual-temporal (Im ε)] row per node (the grid is
node-count-generic: one row for the joint drive, two for the oracle) — because the
drawing code is reused verbatim (`ProtonAnimator` is subclassed; only the drive labels
and the verdict wording change).

The **verdict is observational, never a gate**: the title reports whether the driven node
ended STATIONARY (its `run_stage2` stopped on the stationarity test rather than the
budget) and what emerged — the register (hole) count, `b_k`, and the **diagnostics**
`r_state({1, ω, ω²})` and its conjugate `r_state({1, ω̄, ω̄²})` (the expected partner
sector), read after the fact purely for comparison with the canonical build's carried
level (≈ 0 there). There is no color tolerance and no minimum hole count. The fast
batched path additionally runs `ProtonIngredients.build()`'s persistence check
(continued evolve+relax passes must leave holes, `b_k`, and `F` stable).

Visualization is **off by default** — the default run takes the fast batched path and
prints the observable summary. Opt in with `--live`/`--save` to animate.

    # default: the joint three-pair build, fast; print what emerged
    python emergent_proton.py
    # live (interactive backend):
    python emergent_proton.py --live
    # headless: write a GIF (no display needed):
    python emergent_proton.py --save emergent_proton.gif
    # the two-step reference oracle instead of the joint node:
    python emergent_proton.py --two-step
    # pre-grow the single-Δ⁴ seed by 12 gated cone-ins before optimizing:
    python emergent_proton.py --precone 12
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import multicobordism_animation as mca  # noqa: E402  (the reused chart machinery)

import tessera  # noqa: E402

cob = tessera.cobordism

# ProtonIngredients.build()'s persistence check, mirrored for the fast batched path:
# up to this many continued evolve+relax passes; the LAST one must leave the
# answer-agnostic summary (holes, b_k, F) stable within the relative tolerance.
_PERSIST_PASSES = 3
_PERSIST_REL_TOL = 0.05


def build_joint_node(seed=3, precone=0):
    """The joint inputs-only node — the campaign arm: ONE `MultiCobordism` whose inputs
    are the three Z₃-symmetric neutral q-q̄ pairs `{1,−1,0} ⊔ {0,1,−1} ⊔ {−1,0,1}` (each
    Σ = 0, the only prepared content, fixed for the whole build) and whose output-target
    list is EMPTY. No diquark, no bare quark, no intermediate imposed — whatever the
    relaxed whole carries is read afterwards."""
    ingredients = cob.ProtonIngredients(seed=seed, precone=precone)
    return [(ingredients.joint_node(seed),
             "Joint — 3 neutral q-q̄ pairs (nothing else prepared)")]


def build_ingredients_nodes(seed=3, precone=0):
    """The two-step reference-oracle nodes, in build order: Step A recombination (the
    canonical node, targets intact) then Step B formation with NOTHING pinned (empty
    output-target list), each on its own single-Δ⁴ seed, with `ProtonIngredients.build`'s
    attempt-0 seeds (A = `seed`, B = `seed + 1`). Kept for A-vs-B comparison against the
    joint drive — the ideal-diquark intermediate makes it an oracle, not the arm."""
    ingredients = cob.ProtonIngredients(seed=seed, precone=precone)
    return [
        (ingredients.recombination_node(seed),
         "Step A — recombination (→ diquark {1, ω})"),
        (ingredients.formation_node(seed + 1),
         "Step B — formation (nothing pinned — final state emerges)"),
    ]


def _conjugate_singlet():
    """The antibaryon diagnostic target `{1, ω̄, ω̄²}` — the conjugate of
    `Proton.singlet()`, read (never driven) because the pre-registered expectation of
    the joint drive is a baryon WITH a conjugate partner."""
    return [complex(z).conjugate() for z in cob.Proton.singlet()]


class EmergentProtonAnimator(mca.ProtonAnimator):
    """The canonical animator with ONLY the verdict semantics changed, via the
    announce/paint text hooks — so `--save` AND the responsive `--live` driver both
    report the emergent-arm wording. The panels, the schedule, and the drive are
    inherited verbatim (node-count-generic, so the joint single-node drive and the
    two-step oracle share the machinery). The verdict reports what EMERGED —
    stationarity plus the observable summary — instead of gating on the singlet."""

    _TITLE_PREFIX = "ProtonIngredients build (final state unpinned)"

    def verdict(self):
        """(stationary, singlet_diagnostic, holes) read off the last node's current
        whole cobordism. `singlet_diagnostic` is `r_state({1, ω, ω²})` — reported ONLY
        for comparison with the canonical build's carried level, never as a pass/fail."""
        node = self.nodes[-1][0]
        st = node.st
        res = float(cob.MultiCobordism.r_state(st, self.k, cob.Proton.singlet()))
        holes = len(cob.MultiCobordism.emergent_holes(st, self.k))
        return bool((node.last_stage2_outcome == "CONVERGED")), res, holes

    def _verdict_tag(self, stationary, res, holes):
        st = self.nodes[-1][0].st
        b_k = int(cob.MultiCobordism.betti(st)[self.k])
        conj = float(cob.MultiCobordism.r_state(st, self.k, _conjugate_singlet()))
        return (f"{'stationary ✓' if stationary else 'hit budget (not stationary)'} — "
                f"{holes} register{'s' if holes != 1 else ''} emerged, b{self.k}={b_k}, "
                f"diagnostics r_state(singlet)={res:.2g} / r_state(conjugate)={conj:.2g}")


def _persistence_check(node, evolve_steps, stage1_candidates, stage1_patience,
                       stage2_beta, stage2_iters, degree):
    """`ProtonIngredients.build()`'s persistence check, on an already-driven node: up to
    `_PERSIST_PASSES` continued evolve+relax passes; persistent once one pass leaves the
    answer-agnostic summary — hole count, `b_k`, and `F` — stable. Returns (persistent,
    passes_used)."""
    def betti_k(st):
        betti = cob.MultiCobordism.betti(st)
        return int(betti[degree]) if degree < len(betti) else 0

    for passes in range(1, _PERSIST_PASSES + 1):
        st = node.st
        before = (len(cob.MultiCobordism.emergent_holes(st, degree)), betti_k(st),
                  float(node.objective()))
        node.run_stage1(max_steps=evolve_steps, n_candidate_moves=stage1_candidates,
                        patience=stage1_patience, grow_boundaries=False)
        node.run_stage2(beta=stage2_beta, max_iters=stage2_iters)
        st = node.st
        after = (len(cob.MultiCobordism.emergent_holes(st, degree)), betti_k(st),
                 float(node.objective()))
        stable_f = abs(after[2] - before[2]) <= _PERSIST_REL_TOL * max(abs(before[2]), 1.0)
        if after[0] == before[0] and after[1] == before[1] and stable_f:
            return True, passes
    return False, _PERSIST_PASSES


def run_build(nodes, visualize=False, save=None, degree=3,
              init_steps=mca._INIT_STEPS, evolve_steps=mca._EVOLVE_STEPS,
              stage2_iters=mca._STAGE2_ITERS, stage1_candidates=mca._STAGE1_CANDIDATES,
              stage1_patience=mca._STAGE1_PATIENCE, stage2_beta=1.0, interval=200):
    """Run the emergent-arm two-step build over `nodes`, driving each node the way
    `ProtonIngredients.build()` does (init pass → evolution pass → `run_stage2`).

    With ``visualize=False`` (and no ``save``) this takes the fast batched path and
    returns each node's final metrics plus the OBSERVATIONAL verdict — stationarity,
    the `ProtonIngredients.build()` persistence check on step B, the emergent register
    count, `b_k`, and the singlet diagnostic. `--live`/`--save` animate the identical
    `multicobordism_animation.py` panels instead (persistence is a build-time check and
    is reported by the fast path)."""
    if not visualize and not save:
        out = []
        for node, label in nodes:
            node.run_stage1(max_steps=init_steps, n_candidate_moves=stage1_candidates,
                            patience=stage1_patience, grow_boundaries=True)
            node.run_stage1(max_steps=evolve_steps, n_candidate_moves=stage1_candidates,
                            patience=stage1_patience, grow_boundaries=False)
            node.run_stage2(beta=stage2_beta, max_iters=stage2_iters)
            st = node.st
            out.append((label, {
                "F": float(node.objective()),
                "gradN2": float(cob.MultiCobordism.regge_action_gradient(st)),
                "rU": float(node.r_u(st)),
                "b3": int(cob.MultiCobordism.betti(st)[degree]),
                "holes": len(cob.MultiCobordism.emergent_holes(st, degree))}))
        step_b = nodes[-1][0]
        persistent, passes = _persistence_check(step_b, evolve_steps, stage1_candidates,
                                                stage1_patience, stage2_beta,
                                                stage2_iters, degree)
        st = step_b.st
        res = float(cob.MultiCobordism.r_state(st, degree, cob.Proton.singlet()))
        conj = float(cob.MultiCobordism.r_state(st, degree, _conjugate_singlet()))
        holes = len(cob.MultiCobordism.emergent_holes(st, degree))
        stationary = bool((step_b.last_stage2_outcome == "CONVERGED"))
        out.append(("what emerged", {
            "stationary": stationary,
            "persistent": persistent,
            "persistence_passes": passes,
            "registers": holes,
            "b3": int(cob.MultiCobordism.betti(st)[degree]),
            "singlet_diagnostic": res,
            "conjugate_diagnostic": conj}))
        return out
    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    anim = EmergentProtonAnimator(nodes, degree=degree, init_steps=init_steps,
                                  evolve_steps=evolve_steps, stage2_iters=stage2_iters,
                                  stage1_candidates=stage1_candidates,
                                  stage1_patience=stage1_patience, stage2_beta=stage2_beta)
    anim._setup(plt)
    anim.fig.suptitle(f"{anim._TITLE_PREFIX} — live")
    if save:                                  # off-screen Agg: synchronous per-frame render
        fa = FuncAnimation(anim.fig, anim.update, frames=anim._frames, interval=interval,
                           repeat=False, blit=False)
        fa.save(save, writer="pillow" if save.endswith(".gif") else "ffmpeg", dpi=90)
        print(f"saved animation -> {save}")
        anim._anim = fa  # keep a ref so it isn't GC'd
    else:                                     # responsive live window (compute off GUI thread)
        anim._run_live(plt, interval)
    return anim.hist


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true",
                    help="show the live animation window (slower than the default)")
    ap.add_argument("--save", help="write a GIF/MP4 of the animation (slower)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--init", type=int, default=mca._INIT_STEPS,
                    help="init-pass (grow_boundaries=True) steps per node")
    ap.add_argument("--evolve", type=int, default=mca._EVOLVE_STEPS,
                    help="evolution-pass (grow_boundaries=False) steps per node")
    ap.add_argument("--stage2", type=int, default=mca._STAGE2_ITERS,
                    help="geometric-relaxation iterations per node")
    ap.add_argument("--precone", type=int, default=0,
                    help="pre-grow each node's single-Δ⁴ seed by this many gated "
                         "cone-in moves before optimization (0 = bare seed)")
    ap.add_argument("--two-step", action="store_true", dest="two_step",
                    help="drive the two-step reference oracle (Step A recombination + "
                         "Step B formation) instead of the joint three-pair node")
    args = ap.parse_args()
    if args.two_step:
        nodes = build_ingredients_nodes(seed=args.seed, precone=args.precone)
    else:
        nodes = build_joint_node(seed=args.seed, precone=args.precone)
    result = run_build(nodes, visualize=args.live, save=args.save, init_steps=args.init,
                       evolve_steps=args.evolve, stage2_iters=args.stage2)
    if not args.live and not args.save:
        arm = "two-step reference-oracle" if args.two_step else "joint three-pair"
        print(f"emergent-arm {arm} build finished (final state unpinned; "
              "pass --live or --save to watch it):")
        for label, metrics in result:
            print(f"  {label}:  " + "  ".join(f"{k}={v}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
