# Deep-merge dynamics baseline (#313)

Exploratory probe scripts and one relaxation log, captured to **establish a
reproducible baseline of the deep-merge dynamics on the pre-merge base** (commit
491e11b) so we can re-run after integrating `main` and confirm the model's
outputs are unchanged. These are development scratch — the **canonical runnable
artifact is `../deep_merge_curvature.py`**. The probes reference `/tmp/` paths
from the dev session (records, not necessarily runnable as-is from here); they'll
be productized into the module + a regression test, or relocated to the
`issue-attachments` release, before this branch merges.

## The substrate
- `probe_deep.py` — the deep merge over a geodesically-subdivided holed register
  (central-child hole tracking). Level 2 → 162-vertex register → 486-vertex
  merge, **7 distance shells** (d = 0..6) from the holonomy worldtubes. ker L₁=2
  survives; complex Sorkin action. Also the matter-source profile |h|²(d) (the
  carried register density, falls ~65× over six shells — the charge is localized,
  emergent) and the static curvature (deficit) per shell.
- `probe_resp.py` — the static curvature *response* to a worldtube perturbation:
  confined to d ≤ 1, exactly zero beyond (Regge curvature is ultra-local in the
  edge lengths). This is why a *static* sourced perturbation does not propagate.

## The matter-regulated emergent dual (the real physics)
The mass must regulate the conformal runaway to a **convergent** geometry — not
be sidestepped. Energy/action/free-energy/selection are **exactly #312**
(combinatorial harmonics `harmonicMatrix(1,1e-9,False)`, `cyclePeriods`
recomputed every geometry, `c=lstsq(P,target)`, `E=⟨h,w₁·h⟩`, `dualReggeAction`,
`G=ReS+κE+λ|ImS|`, argmin). The only broadening is the carrier family: a per-shell
radial profile `s(d)` instead of #312's two collective scales.
- `diag.py` — **where the restoring force comes from**: the register span (ker L₁)
  is metric-independent, but `cyclePeriods` are metric-dependent, so the carried
  representative re-fits (`c=lstsq`) as the geometry distorts → `E` gains an
  interior minimum. Freezing `h` deletes it. (Confirms the self-consistent energy
  is mandatory.)
- `probe_exact.py` — exact-#312 energy; conformal scan showing the runaway, and
  that the mass makes the minimum **interior at κ≈1e4**.
- `probe_exact2.py` + `relaxation_kappa1e4.log` — the full **7-DOF radial
  relaxation** at κ=1e4 (≈91 min, 320 evals). Result: convergent/interior (the
  mass regulates), curvature **peaked at the charge** (d=0 ≈ +1.7 excess vs ~+1.0
  bulk). Caveats: G* was still descending at the 320-eval cap (not fully
  converged), and the profile is "peak at charge + irregular elevated bulk," not a
  clean monotonic falloff; the outer low-matter shell drifted toward the bound.

## Rejected approaches (kept as records of what changes the physics)
- `probe_radial2.py` — a **fixed-`h`** energy (compute the harmonic once, vary only
  `w₁`). Faster, but its restoring force is gone → ran to the bounds. Sidesteps the
  mass's regulating role; **do not use**.
- `probe_sparse.py`, `probe_real.py` — a sparse **metric-harmonic** null-space
  energy. Fast and has *a* restoring force, but ~40× weaker than #312's → a
  different functional = **changed physics**. Kept only to document why it was
  rejected.
- `probe_time.py`, `probe_radial.py` — timing / first-cut radial probes.
