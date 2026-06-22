// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_BIPARTITECREATIONTOPOLOGY_H
#define TESSERA_COBORDISM_BIPARTITECREATIONTOPOLOGY_H

#include <cstdint>
#include <vector>

#include "cobordism/TopologyBuilder.h"

namespace tessera::cobordism {

/// # BipartiteCreationTopology
///
/// The **bipartite q/q̄ creation node** (#435): one **seed** window splits into
/// **two separable result windows** — a quark \f$ q \f$ and an antiquark
/// \f$ \bar q \f$ — on ONE connected surface, the elementary building block the
/// proton-assembly experiments (#434/#438) instantiate three times. It is the
/// **time U-turn of a single fermion line** (\f$ \bar q = q \f$ backward in
/// time): the `TemporalOrientation` flip is **localized to the creation vertex**
/// (the apex of a single symmetric reflection), while the propagation slabs stay
/// time-orientation-coherent.
///
/// Like `TripartiteRegisterTopology` the base is ONE connected frequency-N
/// geodesic icosahedron (\f$ S^2 \f$; N=2 → 42 vertices) minus
/// vertex-disjoint hole triangles grouped into windows of three color holes —
/// **distinct holes = independent cycles**, so the windows do not average and the
/// global Stokes relation forces the per-window induced periods to sum to zero.
/// Here there are **three** windows: the **seed** \f$ S \f$ (the pinned, neutral
/// input) and the two emergent results \f$ q, \bar q \f$. Pair-neutrality is then
/// the SAME theorem as confinement:
/// \f$ \sigma_q + \sigma_{\bar q} = -\sigma_S = 0 \f$ for a color-neutral seed.
///
/// **Color is NOT painted.** The two result windows are placed as two
/// \f$ A_4 \f$-equivalent orbits (`TripartiteRegisterTopology`'s symmetric
/// windows), so they are color-**indefinite** at birth — their three color-hole
/// period magnitudes are equal (no preferred color axis, the #414 no-go); a
/// definite color crystallizes only later, from the downstream assembly context.
///
/// **Charge is emergent, never a register.** With `setLorentzianWorldlines()` the
/// cross-time (creation-vertex / U-turn) worldline edges are set timelike
/// (\f$ l^2 < 0 \f$), so the dual Regge action is complex (\f$ \mathrm{Im}\,S
/// \neq 0 \f$) and the electric sector is non-empty. The electric charge of each
/// window is then the Gauss-law holonomy \f$ Q = \oint_S E \f$ read OFF the
/// relaxed connection (`EigenstateSynthesis::curvatureFromConnection` →
/// `fieldStrengthSplit` → `gaussLawCharge`) — never a parallel \f$ \hat Q \f$
/// vertex operator. An all-spacelike (Riemannian) relaxation gives \f$ E \equiv 0
/// \f$ and carries no charge: the **degenerate** case to detect, not the target.
///
/// The orientation-reversing **U-turn twist** (`setUTurnTwist()`, the
/// `RegisterTopology::orientationReversingTwist` mechanism applied to the
/// \f$ \bar q \f$ window) reverses that window's induced orientation, so on the
/// symmetric metric \f$ \bar q \f$ carries the time-reversed (opposite) charge of
/// \f$ q \f$ — the geometric realization of \f$ \bar q = q \f$ backward in time.
///
/// This builder NEVER welds (one connected manifold, `dualComplexValid`), imposes
/// NO matter (`MatterConfiguration()` empty; the dynamics are the relaxation's
/// \f$ \delta S = 0 \f$), and does NOT reduce the dimension (the full symmetric
/// apex interior, #429). The downstream **charge↔color bridge** reads the two
/// windows off the relaxed geometry and hands them to a `TransportCobordism`.
class BipartiteCreationTopology : public TopologyBuilder {
  public:
    [[nodiscard]] std::shared_ptr<Spacetime> build(
        std::size_t stateDim, std::uint64_t seed,
        std::vector<std::vector<std::uint64_t>> &boundaryCells) override;

    /// The EXACT triangle-hole read-out (the #353 period path). Pins the single
    /// supplied seed state on the seed window over `residualForPeriods` with its
    /// induced-orientation covector; returns the quark window \f$ q \f$ as the
    /// emergent `resultHoles`. The antiquark window \f$ \bar q \f$ is read
    /// separately via `antiquarkWindow()` (a creation node emits TWO result
    /// windows, which `TransportCobordism`'s single-result read-out cannot hold).
    void readoutHoles(
        const std::shared_ptr<Spacetime> &cobordism,
        const std::vector<std::vector<std::complex<double>>> &states,
        std::vector<std::vector<std::uint64_t>> &inputHoles,
        std::vector<std::complex<double>> &inputTargets,
        std::vector<std::vector<std::uint64_t>> &resultHoles,
        std::vector<int> &resultSigns) const override;

    /// The two result windows (\f$ q, \bar q \f$) EMERGE from the pinned seed; the
    /// merge pins the seed alone and reads the emergent results off the relax.
    [[nodiscard]] bool emergesResult() const override { return true; }

    [[nodiscard]] std::size_t carriedDim(std::size_t /*stateDim*/) const override {
      return 2;  // the S_3 standard rep (b_1 = 2 on the Sigma=0 hyperplane)
    }

    [[nodiscard]] std::size_t loopsPerState() const override {
      return 3;  // the three color holes per register window
    }

    /// The color register is \f$ d = 3 \f$ (the color triple on \f$ \sum=0 \f$).
    void validateStateDim(std::size_t d) const override;

    [[nodiscard]] std::string name() const override {
      return "bipartite creation ((S^2-9holes) symmetric-stack, seed -> q,qbar)";
    }

    /// Make the creation node LORENTZIAN: the cross-time (creation-vertex / U-turn)
    /// worldline edges are set timelike (\f$ l^2 = \text{worldlineLsq} < 0 \f$), so
    /// the dual Regge action goes complex (\f$ \mathrm{Im}\,S \neq 0 \f$) and the
    /// electric sector is non-empty, letting each window carry a nonzero emergent
    /// Gauss-law charge. Null edges (photons) may EMERGE under the relax as a
    /// worldline's \f$ l^2 \to 0 \f$. Unset (default): all-spacelike (Riemannian),
    /// the degenerate \f$ E \equiv 0 \f$ case that carries NO charge.
    /// @param worldlineLsq the timelike squared length for cross-time edges (<0).
    void setLorentzianWorldlines(double worldlineLsq = -1.0);

    /// Apply the orientation-reversing **U-turn twist** to the antiquark window
    /// (#416, `RegisterTopology::orientationReversingTwist`): it reverses the
    /// induced orientation of each \f$ \bar q \f$ hole (a within-hole transposition
    /// of the two smallest vertices), so \f$ \bar q \f$'s carried period and
    /// emergent electric charge are the time-reversed (sign-flipped) image of
    /// \f$ q \f$'s — the geometric realization of \f$ \bar q = q \f$ backward in
    /// time and the antisymmetric (pair-neutral) channel. On by default; pass
    /// `false` for the untwisted symmetric pair.
    void setUTurnTwist(bool on = true);

    /// Set the geodesic subdivision **frequency** N (the tunable lattice
    /// granularity, #404). N=2 (default) is the #398 base of 42 vertices that hosts
    /// the vertex-disjoint hole triangles; larger N refines the lattice.
    /// @param frequency the subdivision frequency \f$ N \ge 2 \f$.
    void setFrequency(int frequency);

    /// The seed window's three color holes (sorted absolute vertex triples).
    [[nodiscard]] std::vector<std::vector<std::uint64_t>> seedWindow() const;
    /// The quark window's three color holes.
    [[nodiscard]] std::vector<std::vector<std::uint64_t>> quarkWindow() const;
    /// The antiquark window's three color holes (U-turn-twisted when set).
    [[nodiscard]] std::vector<std::vector<std::uint64_t>> antiquarkWindow() const;

    /// The seed window's per-hole induced-orientation signs (`endSignCovector`).
    [[nodiscard]] std::vector<int> seedSigns() const;
    /// The quark window's per-hole induced-orientation signs.
    [[nodiscard]] std::vector<int> quarkSigns() const;
    /// The antiquark window's per-hole induced-orientation signs (sign-reversed by
    /// the U-turn twist relative to the quark window).
    [[nodiscard]] std::vector<int> antiquarkSigns() const;

    /// The number of `TemporalOrientation` flips (apex PT-reflection layers) in the
    /// stack — the count of time U-turns. The creation node is a SINGLE reflection,
    /// so this is **1**: the flip is localized to the one creation slice and the
    /// propagation slabs are time-orientation-coherent (no per-slice PT
    /// alternation, the #429 correction).
    [[nodiscard]] int temporalFlipCount() const { return apexReflections_; }

    /// Whether the orientation-reversing U-turn twist is applied to the antiquark
    /// window (default true). The charge↔color bridge applies the corresponding time
    /// reversal (\f$ Q_{\bar q} = -\oint_{S_{\bar q}} E \f$) so the pair charge
    /// cancels, the geometric realization of \f$ \bar q = q \f$ backward in time.
    [[nodiscard]] bool uTurnTwisted() const { return uTurnTwist_; }

  private:
    // Lorentzian worldlines (set by setLorentzianWorldlines): when on, build()
    // sets the cross-time edges timelike so the electric sector is non-empty.
    bool lorentzian_{false};
    double lorentzWorldlineLsq_{-1.0};

    // The orientation-reversing U-turn twist on the antiquark window (default on).
    bool uTurnTwist_{true};

    // The geodesic subdivision frequency N (set by setFrequency); default 2.
    int frequency_{2};

    // The number of apex PT-reflection layers — the creation node is a single
    // symmetric reflection (one U-turn), so this is 1 (the flip localization).
    int apexReflections_{1};

    // Cached by build() for readoutHoles()/the accessors: the three windows
    // (seed, q, qbar) of three color holes each (sorted absolute vertex triples)
    // and their per-hole induced-orientation signs (endSignCovector of the base
    // surface). Windows live on ONE surface, so holes carry absolute ids.
    std::vector<std::vector<std::vector<std::uint64_t>>> blockHoles_{};
    std::vector<std::vector<int>> signTable_{};
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_BIPARTITECREATIONTOPOLOGY_H
