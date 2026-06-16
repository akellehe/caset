// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLE_H
#define TESSERA_OBSERVABLE_H

#include <memory>

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
// === cross-subsystem fwd-decls ===
namespace tessera::spacetime {
  class Spacetime;
}
namespace tessera::observables {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;


/// # Observable Base Class
///
/// Interface for physical observables measured on a triangulated spacetime
/// configuration. In lattice quantum gravity, observables are functions
/// \f$ \mathcal{O}[\mathcal{T}] \f$ of the triangulation \f$ \mathcal{T} \f$
/// whose expectation values are estimated by averaging over Monte Carlo samples:
///
/// \f[
///   \langle \mathcal{O} \rangle
///     = \frac{1}{Z} \sum_{\mathcal{T}} \mathcal{O}[\mathcal{T}]\, e^{-S[\mathcal{T}]}
///     \approx \frac{1}{N_{\text{meas}}} \sum_{i=1}^{N_{\text{meas}}} \mathcal{O}[\mathcal{T}_i]
/// \f]
///
/// Subclasses include:
///   - **SpacetimeVolume**: total number of \f$ d \f$-simplices \f$ N_d \f$,
///   - **VolumeProfile**: spatial volume per time slice \f$ N_{d-1}(t) \f$.
///
class Observable {
  public:
    /// Compute the observable from scratch for the given spacetime.
    ///
    /// @param spacetime The spacetime configuration to measure
    /// @return The scalar value of the observable
    virtual double compute(const std::shared_ptr<Spacetime> &spacetime);

    /// Incrementally update the observable after a local move.
    /// Default implementation delegates to compute().
    ///
    /// @param spacetime The spacetime after the most recent move
    /// @return The updated scalar value
    virtual double update(const std::shared_ptr<Spacetime> &spacetime);

    virtual ~Observable() = default;
};

} // namespace tessera::observables

#endif //TESSERA_OBSERVABLE_H
