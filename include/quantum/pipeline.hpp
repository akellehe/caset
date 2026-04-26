// End-to-end Phase 3 pipeline: ground-state DMRG followed by
// contiguous-cut Schmidt-spectrum extraction and majorization-poset
// construction. Provides the single C++ entry point that pybind11
// surfaces to Python — keeps the language boundary thin (one config in,
// one struct of plain data out).
//
// Splitting this off from computeGroundState(config) (in dmrg_runner.hpp)
// is deliberate: most users only want the energy and don't pay the O(N²)
// SVD cost; callers who do want the poset opt in by calling this function
// instead.

#pragma once

#include "quantum/dmrg_runner.hpp"
#include "quantum/schmidt.hpp"
#include "quantum/majorization.hpp"

namespace tessera::quantum {

// Result struct for computeGroundStateMajorization. Bundles the same
// scalar diagnostics that computeGroundState returns with the Schmidt
// spectra and the majorization poset of those spectra.
struct GroundStateMajorizationResult {
    GroundStateResult groundState;  // energy, bondDim, truncationErr …
    SchmidtSpectra    spectra;       // labelled contiguous-cut spectra
    Poset             poset;         // Hasse cover edges of strict majorization
};

// Run DMRG to the Schwinger ground state, extract all contiguous-cut
// Schmidt spectra, and build the majorization poset on those spectra.
//
// `config` is interpreted exactly as for computeGroundState — the same
// validation rules apply (N ≥ 2, a > 0, …).
//
// `tol` controls the slack in the majorization comparisons used to build
// the poset. 1e-12 is well below DMRG / SVD numerical noise for the
// problem sizes we exercise; tighten only for synthetic spectra where you
// can prove numerically exact equality.
GroundStateMajorizationResult
computeGroundStateMajorization(QuantumConfig const& config,
                                  double tol = 1e-12);

} // namespace tessera::quantum
