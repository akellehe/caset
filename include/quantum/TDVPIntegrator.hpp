// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.
//
// Two-site, real-time TDVP (time-dependent variational principle) integrator
// for matrix-product states, built entirely on Apache-2.0 ITensor *core*
// primitives (applyExp / LocalMPO / MPS::svdBond / sweepnext). It replaces the
// unlicensed ITensor/TDVP add-on (third_party/itensor_tdvp) that tessera
// previously linked, so the distributed binary carries no code of unstated
// license. See THIRD_PARTY_NOTICES.md.
//
// The algorithm is the standard two-site TDVP scheme (Haegeman et al.,
// Phys. Rev. B 94, 165116 (2016)): sweep across the chain forming each
// two-site block, evolve it forward a half time-step with a local Krylov
// exponential, SVD-truncate it back to the chain, then evolve the resulting
// one-site bond tensor backward a half time-step. Only the two-site
// (NumCenter = 2) path is implemented — that is the only configuration
// tessera uses, and the two-site sweep grows bond dimension on its own, so
// the add-on's one-site subspace-expansion (basisExtension) path is not
// needed.

#ifndef TESSERA_QUANTUM_TDVP_INTEGRATOR_HPP
#define TESSERA_QUANTUM_TDVP_INTEGRATOR_HPP

#include <itensor/all.h>

namespace tessera::quantum {

class TDVPIntegrator {
public:
    // Evolve the MPS `psi` under the MPO Hamiltonian `H` by the (complex)
    // time `t`, applying the two-site TDVP sweep(s) prescribed by `sweeps`.
    //
    // Real-time evolution e^{-iHΔt} corresponds to t = -i·Δt (the same
    // convention as the ITensor/TDVP add-on this replaces). Returns the
    // variational energy ⟨ψ|H|ψ⟩ measured at the final bond.
    //
    // Recognised Args (matching the add-on's behaviour): "NumCenter" (must be
    // 2), "Truncate" (default true), "DoNormalize" (default true), "Silent",
    // "Quiet", "RespectDegenerate", plus the per-sweep Cutoff / MinDim /
    // MaxDim / MaxIter taken from `sweeps`, and "ErrGoal" forwarded to the
    // Krylov exponentiation.
    static itensor::Real evolve(itensor::MPS& psi,
                                itensor::MPO const& H,
                                itensor::Cplx t,
                                itensor::Sweeps const& sweeps,
                                itensor::Args args = itensor::Args::global());
};

}  // namespace tessera::quantum

#endif  // TESSERA_QUANTUM_TDVP_INTEGRATOR_HPP
