// q-qbar quench: prepare a flux-tube initial state from a Schwinger-model
// vacuum-like MPS by flipping two spins separated by `d` sites (PLAN.md
// §5 Phase 4 / Buyens et al. 2014 string-breaking construction).
//
// ─── What this does ───────────────────────────────────────────────────────
//
// In the staggered + Jordan-Wigner + Gauss-eliminated Schwinger
// formulation, the electric field on link n is
//
//     L_n  =  L_0  +  Σ_{k=1..n} [ (1 - σ^z_k)/2  -  (1 - (-1)^k)/2 ]
//          =  c_n  -  ½ Σ_{k=1..n} σ^z_k
//
// (see include/quantum/schwinger_model.hpp). So flipping σ^z at a site k
// shifts L_n on every link n ≥ k. Flipping σ^z at TWO sites (k1, k2) with
// k1 < k2 in opposite directions produces a "flux tube" — L_n shifts by
// +1 (or −1) on links in [k1, k2 − 1] and goes back to vacuum elsewhere.
//
// To create a +1-flux tube on links [i0, i0+d-1], we need to flip σ^z by
// −2 at site i0 (Up → Dn) and by +2 at site i0+d (Dn → Up):
//
//     U_qq̄(i0, d)  =  σ⁻_{i0}  ·  σ⁺_{i0+d}
//
// (PLAN.md writes "σ⁺ σ⁻"; the conjugate σ⁻ σ⁺ is what actually flips the
// vacuum's Up→Dn at i0 and Dn→Up at i0+d. Either form creates a flux
// tube; the sign of the tube is opposite. We use σ⁻ σ⁺ for the +1 tube.)
//
// ─── Parity constraint ────────────────────────────────────────────────────
//
// Both σ⁻ and σ⁺ kill on-shell sites in the wrong direction. For our
// heavy-quark vacuum |↑↓↑↓ … ⟩ (Up at odd 1-based, Dn at even 1-based):
//
//   * σ⁻_{i0} acts non-trivially  ⇔  i0 is odd  (Up sublattice)
//   * σ⁺_{i0+d} acts non-trivially ⇔ i0+d is even (Dn sublattice)
//
// So d must be ODD. PLAN.md mentions d=4; we use d=5 in the heavy-quark
// acceptance test for the same physical reason (5-site separation flux
// tube on 5 links).
//
// At finite m the GS isn't a perfect Néel and the operator never strictly
// vanishes — only a small amplitude is killed by the off-diagonal piece —
// but for clean testing we still want the parity to align.
//
// ─── Sz preservation ──────────────────────────────────────────────────────
//
// σ⁻_{i0} · σ⁺_{i0+d} commutes with total Sz (one raise + one lower
// = net zero charge), so the resulting state stays in the same U(1)
// charge sector as the input. ConserveQNs=true MPSes are fine.

#pragma once

#include <itensor/all.h>

namespace caset::quantum {

// Apply the q-qbar quench operator σ⁻_{i0} σ⁺_{i0+d} to a normalized
// MPS, return a fresh normalized MPS in the same SiteSet.
//
// Throws std::invalid_argument when (i0, d) are out of range or violate
// the parity constraint described in this header. Pass `enforce_parity =
// false` to skip the parity check (e.g. for the broader-mass regime
// where the GS pattern shifts).
itensor::MPS applyQqbarQuench(itensor::MPS const& psi,
                                itensor::SpinHalf const& sites,
                                int i0,
                                int d,
                                bool enforce_parity = true);

} // namespace caset::quantum
