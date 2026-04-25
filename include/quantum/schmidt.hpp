// Schmidt-spectrum extraction for contiguous-interval bipartitions of an
// MPS. PLAN.md §5 Phase 3 calls out exactly this: given an MPS and a
// contiguous interval [i, j], return the Schmidt spectrum of the
// bipartition A = [i, j] vs. its complement.
//
// ─── What we compute ──────────────────────────────────────────────────────
//
// For an MPS |ψ⟩ = T_1 T_2 … T_N |s_1 … s_N⟩ in canonical form with
// orthogonality center brought into A, contracting sites i..j gives a
// tensor M with three index groups: a left bond α (the bond between sites
// i-1 and i), site indices (s_i, …, s_j), and a right bond β (between
// sites j and j+1). With sites outside A in canonical form, the reduced
// density matrix is
//
//     ρ_A^{ s, s' }  =  Σ_{α, β}  M_{αβ}^{s} (M^*)_{αβ}^{s'}
//
// and so the entries of the *Schmidt spectrum* — the eigenvalues of ρ_A —
// are the squared singular values of M reshaped as
// (sites = rows) × (bonds = cols). Equivalently they are the squares of
// the Schmidt coefficients in the decomposition |ψ⟩ = Σ_α λ_α |α⟩_A ⊗
// |α⟩_{Ā} (which is why we call them "values" interchangeably below; the
// methodology page docs/source/quantum-methodology.md fixes the squared
// convention with normalization Σ_α λ_α = 1).
//
// We use the convention λ_α = σ_α^2 throughout (the eigenvalues of ρ_A,
// summing to 1 for a normalized state) so that downstream majorizes()
// calls behave as the methodology page specifies.

#pragma once

#include <itensor/all.h>

#include <vector>

namespace caset::quantum {

// 1-based contiguous interval [i, j] with i ≤ j on a chain of N sites.
struct Interval {
    int i{0};  // first site, 1-based
    int j{0};  // last  site, 1-based, j ≥ i
};

// Schmidt spectrum across the bipartition [i, j] | rest of an MPS `psi`.
//
// Returns the eigenvalues of ρ_A (= squared Schmidt coefficients), sorted
// non-increasingly, with no zero-padding (the caller's majorizes() call
// pads as needed).
//
// Special cases:
//   • i == j == 1 or i == 1 && j == N-1 etc.: bipartitions with one
//     contiguous component on the bar side; handled by the same SVD path.
//   • i == 1 && j == N: the whole chain | empty; returns {1.0}, the
//     spectrum of a 1-dimensional reduced density matrix.
//
// Throws std::invalid_argument if the interval is out of range or i > j.
//
// Complexity: an SVD of a tensor whose dimensions are at most
//   ( min(2^|A|, D_left · D_right) ) × ( D_left · D_right )
// where D_* are the MPS bond dimensions adjacent to the interval. For the
// MPSes we work with (bond dim ~10 — 100, |A| ≤ N/2 in practice) this is
// fast; for very large intervals on long chains the memory may dominate
// and a complement-side computation would be preferable, but Phase 3's
// acceptance tests stay well below that regime.
std::vector<double> schmidt_spectrum(itensor::MPS const& psi,
                                     int i, int j);

// All-contiguous-cut Schmidt spectra of `psi`, excluding the trivial
// full-chain bipartition [1, N] | ∅. PLAN.md §5 Phase 3 specifies exactly
// this set as the cut family $\mathcal{F}$.
struct SchmidtSpectra {
    int N{0};                                 // chain length
    std::vector<Interval> intervals;          // labels for each spectrum
    std::vector<std::vector<double>> spectra; // spectra[k] for intervals[k]
};

SchmidtSpectra all_contiguous_spectra(itensor::MPS const& psi);

} // namespace caset::quantum
