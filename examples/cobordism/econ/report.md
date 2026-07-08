# Kill-experiment report — economic register on BEA 1997–2024

> Go/no-go report for tessera#602. **Status: DRAFT — decision section
> pending the historical run.** Exploratory research; not the official
> development track of tessera or the TVL platform.

## 1. What was tested

The claim: reading the US input-output accounts as an oriented weighted
complex, the **period leak** of a year-over-year transition — the
harmonic, gauge-invariant component of the change, which no re-weighting
of the fixed year-t geometry can absorb — separates known structural
breaks (2001, 2008–09, 2020) from calm years in a way that magnitude
baselines (Frobenius, Leontief-inverse) and the size-recomposition null
(IPF/RAS) do not. Protocol and definitions: `gauge_dictionary.md`;
construction: `econ_register.py`; experiment: `leak_experiment.py`.

## 2. Gates (both passed)

**Gate 1 — the accounting anchor.** The BEA summary Supply–Use identities
were pinned empirically and hold to rounding (≈2×10⁻⁶ relative):
T013 = T007 + MCIF + MADJ; T014 = Trade + Trans (each margin column is an
exact zero-sum redistribution); T015 = MDTY + TOP + SUB;
T016 = T013 + T014 + T015 = T019; T018 = T005 + VABAS with
VABAS = V001 + V003 + T00OTOP (+ T00OSUB from 2018). The assembled
money-flow network (71 industries + HH/CAP/GOV/ROW, purchaser cells split
into basic/margin/tax parts, margins rebooked through the data's own
redistribution weights, net-lending closure into CAP) balances with max
divergence 1.3×10⁻⁷ of total flow; the capital-account discrepancy is
−4.8 M$ on a 59 T$ economy. Sanity anchor: ROW net lending 2017 comes out
at +0.54 T$ — the US current-account deficit.

**Gate 2 — the control pair (frozen 2005 geometry, b₁ = 77).**

- Negative control: re-sourcing half of every buyer's purchases away from
  credit intermediation (521CI), rebalanced to the original margins,
  displaces the flow cochain by 0.84 of its R-norm and leaks 1.2×10⁻⁸.
  The certificate is silent on changes the geometry can carry.
- Positive control: an injected irreducible circulation of ε = 0.05
  (harmonic: margins unchanged, un-nettable) is measured as 0.05000; the
  IPF null absorbs essentially none of it (excess 0.045); sector
  attribution recovers the injected mode's sectors 5/5.

The statistic is calibrated, exact on positive controls, silent on
negative controls, and localizes.

## 3. MRIO consistency control (held-fixed, single vintage)

Production MRIO holds one vintage: 2018, county grain (~92.7 M
A-coefficients). Aggregated to the national industry level and tested on
the frozen national 2018 register: raw cell distance is enormous
(‖A_mrio − A_nat‖_F / ‖A_nat‖_F = 1.54 — a naive Frobenius criterion
would declare a massive structural difference, partly construction-basis,
partly the unweighted detail→summary aggregation), yet the certificate is
correctly silent (leak ≈ 0, off-complex mass 0): the MRIO layer
regionalizes the same accounts, and the machinery classifies the
difference as absorbable. Discrimination runs in both directions.

## 4. The historical decision experiment

**PENDING** — 27 consecutive year-pairs, each on the frozen earlier-year
geometry: observed leak vs IPF-null leak vs Frobenius vs Leontief-inverse
distances; per-pair sector attribution. Results: `out/leak_history.csv`,
decision plot `out/decision.png`.

Decision criteria (from the issue):

1. Recession pairs (2000–01, 2007–08, 2008–09, 2019–20) separated from
   calm pairs in leak excess (observed − IPF null).
2. Sector attribution consistent with the known episodes (finance and
   construction 2008–09; transportation, accommodation, health 2020).
3. Signal not reproduced by Frobenius, Leontief-inverse, or the IPF null.

## 5. Caveats and known limitations

- **Single-vintage caveat**: the annual series was exported from the TVL
  database in one pass (`data/manifest.json` records it); the BEA API
  serves one current revision, but the underlying ingest history is
  gap-tracked, so cross-vintage mixing cannot be fully excluded without
  the pinned-snapshot tooling (deliberately out of scope pre-decision).
  Code-set consistency across years holds except the T00OSUB VA row
  (2018+), which is handled explicitly.
- **Complete graph at industry grain**: off-complex mass is structurally
  ≈0 at 71 industries; the "new relationship" signal only becomes
  available at firm grain (Stage 2 territory).
- **Aggregation**: the MRIO detail→summary comparison uses unweighted
  means over detail codes (no detail output weights at spike level).
- **Margin/tax attribution**: per-commodity uniform rates (the standard
  proportionality assumption); the TVL de-marginalization pipeline does
  this properly at detail level and is the upgrade path.
- **tessera integration**: `ChainComplex.fromSpacetime` reads top cells
  only, so a complex mixing filled triangles and bare edges cannot be
  represented directly; the spike computes boundary operators in numpy
  and uses tessera for cross-checks on representable cases. A mixed-cell
  `ChainComplex` entry point is the natural upstream improvement if this
  program continues.

## 6. Decision

**PENDING the historical run.**
