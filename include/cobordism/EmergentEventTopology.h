// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_EMERGENTEVENTTOPOLOGY_H
#define TESSERA_COBORDISM_EMERGENTEVENTTOPOLOGY_H

#include <cstdint>
#include <vector>

#include "cobordism/TopologyBuilder.h"

namespace tessera::cobordism {

/// # EmergentEventTopology
///
/// The **bilaterally-pinned event cobordism** (#434, Experiment A): the
/// reusable builder that lays out a multi-body color event as ONE connected,
/// tube-connected (#378, never welded) cobordism over several **temporal
/// slices**, pins **only** the initial and final states, and leaves the entire
/// **intermediate interior** free to relax so the intermediate states EMERGE off
/// the relaxed geometry. It is the direct test of the #435 finding: the isolated
/// creation node pinned only ONE boundary (\f$ r_\text{state}\approx 0 \f$), so
/// its relaxation ran into the conformal runaway and never reached the symmetric
/// stationary point; **bilateral** pinning (both endpoints) supplies the
/// constraint the single seed lacked, regulating the runaway.
///
/// ## Construction (one connected manifold, tube-connected)
/// The base is the shared frequency-N symmetric-window geodesic icosahedron
/// (`SymmetricWindowSurface`: \f$ S^2 \f$ minus the four \f$ A_4 \f$-tetrahedral
/// \f$ C_3 \f$-symmetric windows A,B,C,R of three color holes each). Punching the
/// windows opens hole **tubes**; the holed surface is then stacked over
/// `nLayers` temporal layers by the dimension-generic staircase
/// (`Spacetime::prismCells`), giving ONE connected 3-complex whose layers are
/// **tube-connected** through the shared hole-tube walls (the #378 mechanism, not
/// a welded shared block). A window's three color holes at temporal layer
/// \f$ \ell \f$ are the base holes shifted by \f$ \ell\cdot\text{stride} \f$ (and
/// by \f$ \varphi^{\ell} \f$ when the U-turn twist is set), so each window's
/// period is readable at **every** temporal slice.
///
/// ## Bilateral pinning (the experiment)
/// `readoutHoles` pins the three input windows A,B,C at the **bottom** layer
/// (\f$ \ell = 0 \f$: the three color-indefinite neutral-pair quark inputs) AND
/// the result window R at the **top** layer (\f$ \ell = \text{nLayers} \f$: the
/// proton color singlet) — both over the EXACT `residualForPeriods`, signed by
/// the induced-orientation covector (`endSignCovector`). The **middle** layers
/// (\f$ 0 < \ell < \text{nLayers} \f$) are pinned NOWHERE: they are the variable
/// interior whose emergent color content (the transient colored \f$ \bar 3 \f$
/// diquark, the #416 signature) and emergent Gauss-law charge are read off the
/// relax. The canonical emergent intermediate (window R at the middle layer) is
/// returned as `resultHoles` so `TransportCobordism::result` reads it directly;
/// the `windowHolesAtLayer`/`windowSignsAtLayer` accessors expose **any** window
/// at **any** slice for the per-slice crystallization read-out.
///
/// ## Faithfulness (the epic #410 ethos)
/// Emergent-first (read OFF the relaxed geometry, never hand-place an
/// intermediate); NO parallel registers (charge = the emergent Gauss-law
/// holonomy \f$ Q = \oint_S E \f$, #411); the dynamics are the relaxation's
/// \f$ \delta S = 0 \f$ (NOT a sampler), matter NOT imposed
/// (`MatterConfiguration()` empty); NO dimension reduction (the full symmetric
/// stack, #429); NEVER welds (one connected manifold, `dualComplexValid`);
/// standard orientation (#412); the COMPLEX action is kept (`Im S`, the
/// Lorentzian worldlines) so the conformal runaway is regulated by the bilateral
/// constraint, not sidestepped. **Color stays emergent** — no definite color is
/// pinned anywhere (the inputs are color-indefinite, the singlet emerges).
class EmergentEventTopology : public TopologyBuilder {
  public:
    [[nodiscard]] std::shared_ptr<Spacetime> build(
        std::size_t stateDim, std::uint64_t seed,
        std::vector<std::vector<std::uint64_t>> &boundaryCells) override;

    /// The EXACT triangle-hole read-out (#353 period path), BILATERAL: pins the
    /// three input windows A,B,C at the bottom layer and the result window R at
    /// the top layer (signed by their induced-orientation covectors), and returns
    /// the middle-layer R window as the EMERGENT `resultHoles` (read after the
    /// relax). The supplied `states` are the four pinned states in order
    /// A,B,C,R; a state beyond the supplied count is pinned to zero amplitude.
    void readoutHoles(
        const std::shared_ptr<Spacetime> &cobordism,
        const std::vector<std::vector<std::complex<double>>> &states,
        std::vector<std::vector<std::uint64_t>> &inputHoles,
        std::vector<std::complex<double>> &inputTargets,
        std::vector<std::vector<std::uint64_t>> &resultHoles,
        std::vector<int> &resultSigns) const override;

    /// The intermediates EMERGE from the bilaterally pinned endpoints.
    [[nodiscard]] bool emergesResult() const override { return true; }

    [[nodiscard]] std::size_t carriedDim(std::size_t /*stateDim*/) const override {
      return 2;  // the S_3 standard rep (b_1 = 2 per window on the Sigma=0 plane)
    }

    [[nodiscard]] std::size_t loopsPerState() const override {
      return 3;  // the three color holes per register window
    }

    /// The color register is \f$ d = 3 \f$ (the color triple on \f$ \sum=0 \f$).
    void validateStateDim(std::size_t d) const override;

    [[nodiscard]] std::string name() const override {
      return "emergent event ((S^2-12holes) x [0,nLayers] tube-stack, "
             "bilateral pin A,B,C@0 + R@top, intermediates emergent)";
    }

    /// Set the number of **temporal layers** (\f$ \ge 2 \f$): the staircase
    /// stacks the holed surface over `nLayers` product layers, giving `nLayers+1`
    /// vertex copies (slices \f$ 0..\text{nLayers} \f$). At least 2 layers are
    /// needed so a middle (unpinned, emergent) slice exists between the pinned
    /// bottom (\f$ \ell=0 \f$) and top (\f$ \ell=\text{nLayers} \f$). More layers
    /// give the intermediate event stages more room in time.
    /// @param nLayers the number of product layers (\f$ \ge 2 \f$).
    void setLayers(int nLayers);

    /// Make the event cobordism LORENTZIAN: the cross-time (worldline) edges
    /// between adjacent temporal layers are set timelike
    /// (\f$ l^2 = \text{worldlineLsq} < 0 \f$), so the dual Regge action is
    /// complex (\f$ \mathrm{Im}\,S \neq 0 \f$) and the electric sector is
    /// non-empty — the precondition for a non-degenerate emergent Gauss-law
    /// charge. Unset (default): all-spacelike (Riemannian), the degenerate
    /// \f$ E \equiv 0 \f$ case that carries no charge (the control).
    /// @param worldlineLsq the timelike squared length for cross-time edges (<0).
    void setLorentzianWorldlines(double worldlineLsq = -1.0);

    /// Apply the orientation-reversing **U-turn twist** (#416): reverse the
    /// induced-orientation covector of every window, so each carried period (and,
    /// via the bridge, each emergent electric charge) is the sign-flipped image of
    /// the untwisted sector — realized as a readout-level sign reversal (the
    /// within-hole-transposition form `BipartiteCreationTopology` uses for the
    /// antiquark window, never a re-welded geometry). This is the anti-baryon
    /// (anti-proton) sector — the time-reversed (opposite-charge) image of the
    /// untwisted proton sector (\f$ \bar q = q \f$ backward in time), so the two
    /// sectors' charges cancel (total electric charge 0, CPT). Off by default
    /// (the proton sector).
    void setUTurnTwist(bool on = true);

    /// Set the geodesic subdivision **frequency** N (the lattice granularity,
    /// #404). N=2 (default) is the 42-vertex #398 base that hosts the 12
    /// vertex-disjoint holes; larger N refines the lattice.
    /// @param frequency the subdivision frequency \f$ N \ge 2 \f$.
    void setFrequency(int frequency);

    /// Build the genuinely **3+1 D** event over a triangulated \f$ S^3 \f$ spatial
    /// slice (#453) instead of the 2+1 D \f$ S^2 \f$ slice. When set, `build()`
    /// uses `S3WindowSurface` (the join-of-cycles \f$ S^3 \f$ with color windows of
    /// three vertex-disjoint hole **tetrahedra** each), removes the window tetrahedra
    /// (opening the \f$ b_2 \f$ color register --- the \f$ \ker L_2 \f$ degree
    /// \f$ k=2 \f$, the \f$ S^3 \f$ analog of \f$ S^2 \f$'s \f$ b_1 \f$/\f$ k=1 \f$),
    /// and stacks the holed slice over the temporal layers with the **dimension-generic
    /// symmetric apex reflection** (`Spacetime::symmetricStackCells`, #429) into a
    /// genuine **4-manifold** (pentatope top cells), gated by the rigorous \f$ n\ge4 \f$
    /// recursive `dualComplexValid`. The window holes, signs, and bilateral pinning
    /// flow exactly as in the \f$ S^2 \f$ path but with 4-vertex tetrahedral holes
    /// read at `registerDegree()` \f$ =2 \f$. Off by default (the \f$ S^2 \f$ slice).
    /// @param on whether to build the \f$ S^3 \f$ (3+1 D) slice.
    void setS3Slice(bool on = true);

    /// Whether the \f$ S^3 \f$ (3+1 D) spatial slice is selected (default false).
    [[nodiscard]] bool s3Slice() const { return s3Slice_; }

    /// Set the number of color windows on the \f$ S^3 \f$ slice (\f$ \ge 1 \f$;
    /// default 4 = the A,B,C,R structure). The holed slice then carries
    /// \f$ b_2 = 3\cdot\text{windows} - 1 \f$. Fewer windows make a much smaller
    /// 4-complex (the like-resolution four-window event is the 10^3-10^4x cost the
    /// #418 spike budgeted for); one window is the minimal genuinely-4D event. Only
    /// the first four windows are bilaterally pinned by `readoutHoles` (A,B,C,R).
    /// @param windows the number of \f$ \mathbb{Z}_3 \f$ color windows.
    void setS3Windows(int windows);

    /// The number of \f$ S^3 \f$ color windows (default 4).
    [[nodiscard]] int s3Windows() const { return s3Windows_; }

    /// The Hodge degree \f$ k \f$ at which the color register is read: \f$ k=1 \f$
    /// (\f$ b_1 \f$, triangle holes) on the \f$ S^2 \f$ slice; \f$ k=2 \f$
    /// (\f$ b_2 \f$, tetrahedral holes) on the \f$ S^3 \f$ slice. The register is
    /// always \f$ \ker L_{d-1} \f$ for a \f$ d \f$-dimensional spatial slice.
    [[nodiscard]] std::size_t registerDegree() const override {
      return s3Slice_ ? 2u : 1u;
    }

    /// The number of temporal layers (\f$ \ge 2 \f$); slices are \f$ 0..nLayers \f$.
    [[nodiscard]] int nLayers() const { return nLayers_; }

    /// The per-layer vertex stride (the base holed-surface vertex count): a base
    /// vertex \f$ v \f$ at layer \f$ \ell \f$ has cobordism id
    /// \f$ \varphi^{\ell}(v) + \ell\cdot\text{stride} \f$.
    [[nodiscard]] std::uint64_t stride() const { return stride_; }

    /// The number of windows (4: A,B,C,R).
    [[nodiscard]] std::size_t windowCount() const { return blockHoles_.size(); }

    /// The three color holes (sorted absolute cobordism-vertex triples) of window
    /// `w` (0=A, 1=B, 2=C, 3=R) at temporal layer `layer` (\f$ 0..nLayers \f$):
    /// the base window holes shifted by the layer offset
    /// \f$ \ell\cdot\text{stride} \f$. The handle for reading any window's period
    /// at any temporal slice off the relaxed geometry.
    [[nodiscard]] std::vector<std::vector<std::uint64_t>> windowHolesAtLayer(
        std::size_t w, int layer) const;

    /// The per-hole induced-orientation signs of window `w` at `layer`
    /// (`endSignCovector`; sign-reversed by the U-turn twist).
    [[nodiscard]] std::vector<int> windowSignsAtLayer(std::size_t w,
                                                      int layer) const;

    /// Whether the U-turn (anti-baryon) twist is applied (default false).
    [[nodiscard]] bool uTurnTwisted() const { return uTurnTwist_; }

  private:
    int nLayers_{2};            // temporal product layers (>= 2); slices 0..nLayers
    int frequency_{2};          // geodesic subdivision frequency N
    bool lorentzian_{false};    // timelike cross-layer worldlines
    double lorentzWorldlineLsq_{-1.0};
    bool uTurnTwist_{false};    // anti-baryon (orientation-reversing) sector
    bool s3Slice_{false};       // genuinely 3+1 D: S^3 slice (#453) vs the S^2 slice
    int s3Windows_{4};          // number of S^3 color windows (b_2 = 3*windows - 1)

    std::uint64_t stride_{0};   // per-layer vertex stride (base vertex count)

    // The base (layer-0) windows of three color holes each (A,B,C,R) and their
    // per-hole induced-orientation signs (endSignCovector of the base surface).
    std::vector<std::vector<std::vector<std::uint64_t>>> blockHoles_{};
    std::vector<std::vector<int>> signTable_{};
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_EMERGENTEVENTTOPOLOGY_H
