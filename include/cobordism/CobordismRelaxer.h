// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_COBORDISMRELAXER_H
#define TESSERA_COBORDISM_COBORDISMRELAXER_H

#include <complex>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "cobordism/EigenstateSynthesis.h"

namespace tessera::mesh { class Edge; }
namespace tessera::spacetime { class Spacetime; }

namespace tessera::cobordism {
using ::tessera::spacetime::Spacetime;

/// # CobordismRelaxer
///
/// The shared stationary-action backreaction relaxation for the cobordism
/// primitives. `MergeCobordism` (the operator/final-state merge, scored over
/// edge-loops) and `TransportCobordism` (the carried-rep transport, scored over
/// triangle holes) differ in what they pin and what they read out, but they
/// relax the *same* objective: a bounded Gauss-Newton / Levenberg-Marquardt
/// descent of \f$ \beta\,\lVert\nabla_I S\rVert^2 + r_{\text{state}} \f$ over the
/// interior edge squared-lengths, using the exact analytic gradient and sparse
/// analytic Hessian of the dual Regge action. This util holds that descent so the
/// two primitives share it rather than duplicating it.
class CobordismRelaxer {
  public:
    /// The sorted (min,max) endpoint key of a mesh edge.
    [[nodiscard]] static std::pair<std::uint64_t, std::uint64_t> edgeKey(
        const ::tessera::mesh::Edge *e);

    /// Betti numbers of a complex (combinatorial, geometry-free).
    [[nodiscard]] static std::vector<int> betti(const Spacetime &st);

    /// One bounded GN/LM relaxation of the interior edge squared-lengths to a
    /// stationary point of \f$ \beta\lVert\nabla_I S\rVert^2 + r_{\text{state}} \f$.
    /// dW is held fixed (Dirichlet), so the stationarity is over interior edges.
    /// The state residual is scored over EITHER the EXACT triangle holes
    /// (`stateHoles`, the register/junction's `residualForPeriods` /
    /// `periodGapForPeriods`) OR the SOFT edge-loops (`stateLoops`, the operator's
    /// `residualForLoops` / `periodGapForLoops`) -- whichever is non-empty (holes
    /// take precedence). `periodPin` selects r_psi (period-gap) over r_U
    /// (realizability). The state-residual gradient is folded in ONLY while
    /// r_state exceeds `stateEpsilon` (once it is converged its gradient is
    /// numerical noise that, through the ill-conditioned action Hessian, explodes
    /// the step). Returns the final residual; leaves the interior edges at the
    /// best point found.
    ///
    /// `registerDegree` is the Hodge degree \f$ k \f$ the state residual is read
    /// at (`TopologyBuilder::registerDegree()`): \f$ 1 \f$ (\f$ b_1 \f$, triangle
    /// holes) on an \f$ S^2 \f$ (2+1 D) slice, \f$ 2 \f$ (\f$ b_2 \f$, tetrahedral
    /// holes) on an \f$ S^3 \f$ (3+1 D) slice. The `residualForPeriods` cost is
    /// degree-general (a hole is a \f$ (k+2) \f$-vertex tuple whose facets are
    /// \f$ k \f$-cells), so the line search tracks the carried register at any
    /// degree. The analytic state-residual GRADIENT, however, is the degree-1
    /// \f$ L_1 \f$ port (`periodGradientOverLoops`), so it is folded in ONLY at
    /// \f$ k=1 \f$; at \f$ k \ge 2 \f$ the descent is the action gradient alone
    /// (\f$ \beta\lVert\nabla_I S\rVert^2 \f$) with the degree-\f$ k \f$ state cost
    /// still gating the line search -- on the symmetric uniform seed the carried
    /// register already sits at its `residualForPeriods` floor, so its gradient is
    /// the dropped numerical-noise term either way (#396). The default \f$ 1 \f$
    /// keeps every \f$ S^2 \f$ caller byte-identical.
    [[nodiscard]] static double relaxInterior(
        const std::shared_ptr<Spacetime> &st, double beta,
        const std::vector<EigenstateSynthesis::EdgeLoop> &stateLoops,
        const std::vector<std::complex<double>> &stateTargets,
        const std::vector<std::vector<std::uint64_t>> &stateHoles,
        const std::vector<std::complex<double>> &holeTargets, int maxIters,
        int &iterCounter, bool periodPin, double stateEpsilon,
        bool verbose = false, int registerDegree = 1);
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_COBORDISMRELAXER_H
