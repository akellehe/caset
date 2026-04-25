// Implementation of the q-qbar quench operator. See
// include/quantum/quench.hpp for the physics + parity discussion.

#include "quantum/quench.hpp"

#include <itensor/all.h>

#include <stdexcept>

namespace caset::quantum {

itensor::MPS applyQqbarQuench(itensor::MPS const& psi_in,
                                itensor::SpinHalf const& sites,
                                int i0,
                                int d,
                                bool enforce_parity) {
    using namespace itensor;
    const int N = length(psi_in);
    if (i0 < 1 || i0 > N) {
        throw std::invalid_argument(
            "applyQqbarQuench: i0 out of range [1, N]");
    }
    if (d < 1 || i0 + d > N) {
        throw std::invalid_argument(
            "applyQqbarQuench: i0 + d out of range [i0+1, N]");
    }
    if (enforce_parity) {
        // For the heavy-quark vacuum |↑↓↑↓ … ⟩: σ⁻ acts non-trivially only
        // on Up sites (odd 1-based), σ⁺ only on Dn sites (even 1-based).
        // So i0 must be odd and i0 + d must be even ⇒ d must be odd.
        if ((i0 % 2) != 1) {
            throw std::invalid_argument(
                "applyQqbarQuench: i0 must be in the Up sublattice (odd, "
                "1-based) for the heavy-quark vacuum; pass "
                "enforce_parity=false to override");
        }
        if ((d % 2) != 1) {
            throw std::invalid_argument(
                "applyQqbarQuench: d must be odd so i0 + d lands in the "
                "Dn sublattice (even, 1-based); pass enforce_parity=false "
                "to override");
        }
    }

    // ITensor pattern for applying a 1-site operator to an MPS:
    //   1) bring the orthogonality center to the site
    //   2) contract op * tensor and remove the prime introduced by op
    //   3) overwrite the site tensor.
    auto apply_local = [&](MPS& psi, int site, std::string const& op_name) {
        psi.position(site);
        auto Op = op(sites, op_name, site);
        auto new_local = noPrime(Op * psi(site));
        psi.set(site, new_local);
    };

    MPS psi = psi_in;
    apply_local(psi, i0,     "S-");   // σ⁻_{i0}: Up → Dn
    apply_local(psi, i0 + d, "S+");   // σ⁺_{i0+d}: Dn → Up
    psi.normalize();
    return psi;
}

} // namespace caset::quantum
