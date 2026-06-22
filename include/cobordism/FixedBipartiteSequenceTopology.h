// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_FIXEDBIPARTITESEQUENCETOPOLOGY_H
#define TESSERA_COBORDISM_FIXEDBIPARTITESEQUENCETOPOLOGY_H

#include <complex>
#include <cstdint>
#include <vector>

#include "cobordism/EmergentEventTopology.h"

namespace tessera::cobordism {

/// # FixedBipartiteSequenceTopology
///
/// The **fixed-bipartite-sequence event cobordism** (#438, Experiment B): the
/// SAME connected, tube-connected (#378, never welded) `EmergentEventTopology`
/// structure as Experiment A (#434) — the shared `SymmetricWindowSurface` (\f$
/// S^2 \f$ minus the four \f$ A_4 \f$ windows A,B,C,R) stacked over `nLayers`
/// temporal slices by the staircase — reused VERBATIM (this is a subclass; the
/// `build` is inherited unchanged), but with the **intermediate window
/// additionally pinned** to the known bipartite sequence.
///
/// ## What B adds over A (the independent variable)
/// A pins ONLY the endpoints (the three color-indefinite quark inputs A,B,C at
/// the bottom slice, the proton color singlet R at the top slice) and lets the
/// intermediate EMERGE. B keeps those exact endpoints color-emergent and
/// ADDITIONALLY pins the intermediate **result window R at every interior
/// temporal layer** to the explicit bipartite intermediate: the **colored
/// \f$ \bar 3 \f$ diquark** — the antisymmetric (wedge / cross-product)
/// combination \f$ d = q_A \wedge q_B \f$ of the two quark inputs, carried in the
/// conjugate (anti-triplet) rep via the orientation-reversing **#416 twist**
/// (the induced-orientation covector is negated on the diquark window, exactly
/// the within-hole sign reversal `EmergentEventTopology` uses for the U-turn
/// sector — never a re-welded geometry). The spectator quark C is carried
/// through; \f$ \text{diquark} + q_C \to \f$ proton at the top.
///
/// ## The question B answers
/// A produced only a **weak** transient \f$ \bar 3 \f$ (its emergent
/// intermediate was singlet-dominated, \f$ \sigma \approx 0.10 \f$). B PINS the
/// strong colored \f$ \bar 3 \f$ and asks whether the connected bulk **hosts**
/// it — whether the diquark's own realizability residual \f$ r_U \f$ lands near
/// A's connected-bulk value (\f$ \sim 0.3\text{–}0.5 \f$, *hosted*) or far above
/// it (the free-quark-like *floored* residual) — read by
/// `EigenstateSynthesis::residualForPeriods` over the diquark holes ALONE at the
/// relaxed metric. \f$ \|\nabla S\|^2 \f$ stays the realizability check of the
/// whole pinned path (per-depth, extensive in temporal volume — read only at
/// convergence).
///
/// ## Faithfulness (the epic #410 ethos, same quarantine as #434)
/// Emergent-first EVERYWHERE except the deliberately-pinned intermediate diquark
/// (the experiment's independent variable). The **endpoints stay
/// color-emergent** (no painted color at creation). NO parallel registers
/// (charge = the emergent Gauss-law holonomy \f$ Q = \oint_S E \f$, #411); the
/// dynamics are the relaxation's \f$ \delta S = 0 \f$ (NOT a sampler), matter NOT
/// imposed; NO dimension reduction (the full symmetric stack, #429); NEVER welds
/// (one connected manifold, `dualComplexValid`); standard orientation (#412);
/// the COMPLEX action is kept (`Im S`, the Lorentzian worldlines).
class FixedBipartiteSequenceTopology : public EmergentEventTopology {
  public:
    /// BILATERAL endpoints (as #434: A,B,C @ bottom, R @ top) PLUS the
    /// intermediate diquark: window R at every strictly-interior temporal layer
    /// (\f$ 0 < \ell < \text{nLayers} \f$) pinned to the colored \f$ \bar 3 \f$
    /// (the #416-twisted antisymmetric combination of the A,B quark inputs). The
    /// emergent `resultHoles` is window R at the middle slice (the imposed
    /// diquark) so `TransportCobordism::result` mirrors A's intermediate read-out
    /// for the A-vs-B comparison. The supplied `states` are A,B,C,R in order; an
    /// explicit fifth state (or `setDiquarkColor`) overrides the derived diquark.
    void readoutHoles(
        const std::shared_ptr<Spacetime> &cobordism,
        const std::vector<std::vector<std::complex<double>>> &states,
        std::vector<std::vector<std::uint64_t>> &inputHoles,
        std::vector<std::complex<double>> &inputTargets,
        std::vector<std::vector<std::uint64_t>> &resultHoles,
        std::vector<int> &resultSigns) const override;

    [[nodiscard]] std::string name() const override {
      return "fixed bipartite sequence (#434 structure + intermediate diquark "
             "3bar pinned on R at interior layers via the #416 twist)";
    }

    /// Turn the intermediate-diquark pin on/off (default ON). Off reproduces the
    /// pure #434 emergent-intermediate experiment on the same subclass (a
    /// control: the diquark is NOT pinned, the endpoints relax as in A).
    void setPinIntermediate(bool on) { pinIntermediate_ = on; }

    /// Whether the intermediate diquark is pinned (default true).
    [[nodiscard]] bool pinsIntermediate() const { return pinIntermediate_; }

    /// Override the colored \f$ \bar 3 \f$ diquark color (a 3-vector over R's
    /// three color holes). When unset (empty), the diquark is DERIVED as the
    /// antisymmetric (cross-product) combination \f$ q_A \wedge q_B \f$ of the
    /// two pinned quark inputs, normalized to the singlet norm \f$ \sqrt 3 \f$.
    void setDiquarkColor(const std::vector<std::complex<double>> &color) {
      diquarkColor_ = color;
    }

    /// The colored \f$ \bar 3 \f$ diquark color that will be pinned for the given
    /// pinned `states` (A,B,C,R[,diquark]): the explicit override / fifth state
    /// if supplied, else the normalized antisymmetric \f$ q_A \wedge q_B \f$.
    /// Exposed so the read-out can report exactly what was imposed.
    [[nodiscard]] std::vector<std::complex<double>> diquarkColorFor(
        const std::vector<std::vector<std::complex<double>>> &states) const;

    /// Window R's three color holes at the middle (\f$ \ell = \text{nLayers}/2 \f$)
    /// slice — the canonical diquark holes for the per-window \f$ r_U \f$ read-out.
    [[nodiscard]] std::vector<std::vector<std::uint64_t>> diquarkHoles() const;

    /// The orientation-reversed (\f$ \bar 3 \f$, #416-twisted) per-hole signs of
    /// the diquark window at the middle slice — the signs that multiply the
    /// diquark color into its target periods.
    [[nodiscard]] std::vector<int> diquarkSigns() const;

  private:
    bool pinIntermediate_{true};
    std::vector<std::complex<double>> diquarkColor_{};
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_FIXEDBIPARTITESEQUENCETOPOLOGY_H
