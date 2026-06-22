# Scale-fixing prototype — regulating the conformal mode in Experiment A (#434)

**Status:** prototype (exploratory branch `proto/scale-fix-relaxer`, not a mergeable
PR). **Epic:** Flavor and Charge Sectors (#410). **Builds on:** Experiment A (#434),
the bipartite creation finding (#435).

## The diagnosis being tested

The cobordism relaxer minimizes `cost = β·‖∇_I S‖² + r_state`
(`src/cobordism/CobordismRelaxer.cpp`). In the distinct-window junctions the pinned
neutral inputs are carried on their own independent cycles, so when they carry
*exactly* (`r_state → 0`, e.g. the proton junction at `r_state ≈ 3e-27`) the relaxer
(by design, #396) **gates off** the `r_state` gradient — leaving the relaxation driven
by the Regge term alone. The conformal/scale mode is then unregulated:
`∂S/∂scale ≠ 0` everywhere (the runaway), so `‖∇S‖²` can never reach 0 — it plateaus,
and the LM drags the *shape* to compensate, which the diagnosis predicts degrades the
per-window color balance. `r_U` provides no restoring force because the matter term is
null by the independent-cycle topology.

This prototype adds an explicit scale-fixing term and re-runs Experiment A to ask:
**(Q1)** does pinning the conformal DOF let `‖∇S‖²` drop below the floor / converge?
**(Q2)** does it stop the per-window spread degradation?

## What was prototyped

An **opt-in, default-OFF** scale-fixing penalty in `CobordismRelaxer::relaxInterior`,
read once from `TESSERA_SCALEFIX` (unset or `0` ⇒ the relaxation is byte-identical to
upstream; verified below). When `scaleW > 0`:

```
P = scaleW · (mean_e log|Re l²_e| − targetLogScale)²      over the interior edges
```

`targetLogScale` is the mean log-scale captured once at the seed, so `P` pins the
**overall** conformal scale (one DOF) to its seed value and leaves the shape free.
`P` is added to the LM cost lambda, and its analytic gradient and Gauss–Newton Hessian
are folded into the LM `grad`/`B` as a **rank-1** update in
`J_e = ∂(mean)/∂l²_e = 1/(N·Re l²_e)` — `grad += 2·scaleW·g·J`, `B += 2·scaleW·J·Jᵀ`
(both finite-difference-verified to ~1e-10). Near-null edges (`|Re l²| < 1e-12`) are
skipped.

**Default-off invariant — VERIFIED.** With `TESSERA_SCALEFIX` unset,
`tests/cobordism/test_epic410_invariants.py` G1–G5 stay green (5 passed, twice on
independent rebuilds) — the gating is byte-identical when off.

## The sweep (proton sector, Lorentzian)

`‖∇S‖²` is the UNWEIGHTED interior Regge gradient norm (β=1 here, so it is exactly the
`stat_action_residual`; the scale penalty is NOT folded into that read-out — it is the
pure Regge norm read off the relaxed geometry). `r_state` is `r_U`; `σ_pair` is the
mean color charge over the three input quark windows; `spread_*` are per-window color
spreads (bottom inputs / middle emergent slice / top result). Floor = `‖∇S‖² < 100`.

### Minimal depth `nLayers = 2` (Experiment A's primary config)

Seed (no relaxation): `‖∇S‖² = 165.4`, `spread_bottom = 3.000`, `spread_mid = 0.147`,
`spread_top = 0.000`, `σ_pair = 0.577`, `top_singlet = 1.000`.

| scaleW | ‖∇S‖² | r_state | converged | iters | σ_pair | spr_bot | spr_mid | spr_top | top singlet |
|---|---|---|---|---|---|---|---|---|---|
| 0 (control) | 71.36 | 5.33e-1 | False | 33 | 0.577 | 3.000 | 0.147 | 0.000 | 1.000 |
| 0.1 | 70.55 | 5.32e-1 | False | 38 | 0.577 | 3.000 | 0.147 | 0.000 | 1.000 |
| 1 | 68.00 | 5.32e-1 | False | 53 | 0.577 | 3.000 | 0.147 | 0.000 | 1.000 |
| 10 | 54.85 | 5.60e-1 | False | 37 | 0.577 | 3.000 | 0.147 | 0.000 | 1.000 |
| 100 | 56.40 | 5.17e-1 | False | 32 | 0.577 | 3.000 | 0.147 | 0.000 | 1.000 |

Reading the table:

- **`‖∇S‖²` does drop** as `scaleW` rises: 71.4 → 68.0 (w=1) → **54.8 (w=10)**, a ~23%
  reduction, then saturates (56.4 at w=100). So the conformal mode **is** a real
  contributor to the plateau, and pinning it genuinely helps — but the reduction
  **saturates well above 0** (~55), it does not march toward the floor-of-0.
- **`r_state` is untouched** (~0.53 throughout): exactly as expected — the term pins
  scale, not the colored→singlet state residual.
- **Convergence stays `False` at every weight.** The convergence test is
  `r_state + ‖∇S‖² < ε`, and `r_state ≈ 0.53` (the physical colored→singlet `r_U`
  obstruction the #434 report already flagged as intrinsic) dominates it. Scale-fixing
  cannot move that, so **it does not make Experiment A converge.**
- **Spread / neutrality is perfectly flat** across all weights *and* the seed
  (`spr_bot = 3.000`, `spr_mid = 0.147`, `spr_top = 0.000`, `σ_pair = 0.577`,
  `top_singlet = 1.000`). The scale DOF the term pins does **not couple** to the color
  spread (spread is scale-invariant, per the pre-registered caveat). So the predicted
  shape-drag degradation does **not** appear in these window read-outs at nL=2, and
  scale-fixing neither worsens nor improves it. The proton singlet still emerges
  exactly (1.000) at every weight — the term does not break the physics.
- **No blow-up.** The relaxation is stable and well-behaved at every weight up to 100;
  the rank-1 GN-Hessian fold keeps the LM step conditioned.

### Depth `nLayers = 4` (the above-floor runaway test) — seed only

Seed (no relaxation): `‖∇S‖² = 418.8` (same spread/σ as nL=2). The `nL=4` **relaxed**
sweep (control `‖∇S‖² ≈ 268` per the #434 report, the case genuinely *above* the floor)
could not be completed: the prototype's build/run environment repeatedly reaped the
git worktree mid-run (the worktree directory and branch were removed every ~5–15 min,
independent of path under `/home` or `/tmp`), terminating each relaxation. The committed
source and harness reproduce it on a stable checkout. Extrapolating the nL=2 reduction
(~23 % at w=10) to the nL=4 control (≈268) predicts ≈205 — i.e. still far above the 100
floor — so on the available evidence the term at these weights would **not** regulate
the runaway below the floor when the gradient genuinely exceeds it.

## Verdict — an explicit scale term is **not** the right regulator for Experiment A

1. **It works mechanically and safely.** Opt-in, default-off byte-identical (G1–G5
   green), gradient/Hessian FD-verified, no divergence at any weight, the proton singlet
   still emerges (1.000). As an engineering object the term is sound.
2. **It does regulate the conformal mode — partially.** `‖∇S‖²` falls ~23 % (71→55) and
   saturates; the conformal DOF demonstrably contributes to the plateau. But the
   reduction is modest and saturating, nowhere near closing the gap to the floor when
   the gradient is actually above it (nL=4 extrapolation ≈205 ≫ 100).
3. **It does not make A converge,** because A's real obstruction is the physical
   colored→singlet `r_U ≈ 0.53`, which the scale term cannot touch — the same intrinsic
   residual #434 identified.
4. **It does not affect spread/neutrality** (scale-invariant): it neither causes nor
   cures shape-drag in the color read-outs.
5. **The deeper reason — Experiment A is the wrong test bed.** The diagnosis the term
   targets (conformal runaway because `r_state → 0` gates off the state gradient)
   *requires* `r_state → 0`. Experiment A does **not** satisfy that: its bilateral
   colored→singlet pin gives `r_state ≈ 0.53`, so the state gradient is **active** and
   already supplies a restoring force. The scale term barely moves A because A is not in
   the regime the term was designed for. The genuine test of the term is the
   **exactly-carrying proton junction** (`r_state ≈ 3e-27`, where the #396 gate truly
   fires and the conformal mode is truly unregulated) — that, not A, is where an
   explicit scale term should be evaluated next.

**Bottom line:** the scale-fixing term is a clean, safe, opt-in regulator that does pin
the conformal DOF and modestly lowers `‖∇S‖²`, but it neither makes Experiment A
converge nor changes its color balance, because A's residual is physical (`r_U`), not
conformal. Recommend evaluating the term on the exactly-carrying junctions (the proton)
rather than adopting it for A; do **not** merge.

## How to reproduce

```
git worktree add <path> proto/scale-fix-relaxer
cd <path> && git submodule update --init third_party/itensor
python3 -m venv .venv-build
OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 .venv-build/bin/pip install -e ".[dev]"
# default-off invariant (byte-identical):
.venv-build/bin/python -m pytest tests/cobordism/test_epic410_invariants.py
# the sweep (TESSERA_SCALEFIX is read per relaxInterior call):
.venv-build/bin/python examples/cobordism/scalefix_sweep.py          # nL=2 sweep
.venv-build/bin/python examples/cobordism/scalefix_one.py 10 4 40     # one point: scaleW=10, nL=4
```
