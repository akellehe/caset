# Economic register — kill-experiment spike

> **Exploratory status.** This directory is research scaffolding for
> [akellehe/tessera#602](https://github.com/akellehe/tessera/issues/602). It is
> not the official development track of either tessera or the TVL platform. Do
> not treat it as canon or build on it until the go/no-go decision recorded in
> `report.md` lands. A clean negative result is an acceptable terminal outcome.

## The claim under test

Read the US input-output accounts as an oriented, weighted simplicial complex:
industries (plus closure sectors — households, government, investment, rest of
world) are vertices, dollar flows are directed weighted edges, and triads whose
mutual obligations can be netted by bookkeeping alone are filled triangles. The
harmonic flows of that complex — the null space of the degree-1 Hodge Laplacian
— are the economy's irreducible circuits of money, and their periods are the
gauge-invariant register carried by the network (the Wilson loops of an
additive-group lattice gauge theory whose Gauss law is the flow-of-funds
identity).

The **period leak** of a year-over-year transition is the component of change
that no re-weighting of the existing network can absorb: a topological
obstruction, in the sense of the register theorems of
`papers/cobordism-residual/main.tex`.

**Hypothesis:** the normalized period leak separates known structural breaks
(2001, 2008–09, 2020) from calm years in a way that magnitude and
size-composition baselines do not.

## Gates (in order; later steps are uninterpretable if earlier gates fail)

1. **Divergence gate.** On the balanced, closed table, divergence residuals
   (net accumulation at each vertex) vanish up to rounding, every year. If not,
   the closure design is wrong.
2. **Planted-break gate.** Surgically deleting a known sector's relationships
   in one year must fire the leak, and harmonic-circuit attribution must
   localize to that sector. A machine that cannot find a break we planted has
   no business interpreting 2008.

## Decision experiment

For every consecutive year-pair 1997→latest: normalized period leak versus
three baselines —

- **IPF/RAS null**: later-year margins fitted on the earlier-year prior. Under
  the gauge reading this is the minimum-energy relaxation with pinned Gauss-law
  data, so the comparison is principled: the leak is what relaxation cannot
  remove.
- **Frobenius distance** between consecutive tables (naive magnitude).
- **Leontief-inverse distance** (propagated-requirements magnitude).

The netting threshold and the value→metric map (length ∝ value vs.
length ∝ 1/value, i.e. conductance ∝ flow) are scanned knobs, not fixed
choices. Sector attribution uses the projection proxy (rank harmonic circuits
by leak contribution), not surgery search — surgery localization is post-signal
work.

## Contents

| file | purpose |
|---|---|
| `fetch_bea_io.py` | one-off export of BEA summary Make (262) + Use (259) tables from the TVL platform to parquet (documented, not automated) |
| `econ_register.py` | construction + register: IxI flows, closure, complex, b₁, harmonic circuits, periods, Gram matrix, divergence residuals |
| `leak_experiment.py` | planted-break gate, year-pair leak vs. baselines, threshold/metric scan, decision plot |
| `gauge_dictionary.md` | theory note: the additive lattice-gauge reading of the accounts |
| `report.md` | go/no-go report (written last) |

## Data

The spike reads the TVL database directly (read-only) and records ingest
metadata as a vintage caveat. Pinned-vintage snapshot tooling is deliberately
out of scope until a go decision.
