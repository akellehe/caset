// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_TRIPARTITEREGISTERTOPOLOGY_H
#define TESSERA_COBORDISM_TRIPARTITEREGISTERTOPOLOGY_H

#include <cstdint>
#include <vector>

#include "cobordism/TopologyBuilder.h"

namespace tessera::cobordism {

/// # TripartiteRegisterTopology
///
/// The **trivalent W_ABC junction** (#396): three color-neutral q-qbar pair
/// inputs into one bulk, \f$ \partial W = \psi_A \sqcup \psi_B \sqcup \psi_C \f$,
/// whose **emergent** result block reads out the proton's color register.
///
/// The decisive design fact (#382 / the obstruction note): a single shared color
/// register **averages** its inputs (least-squares projection onto one period
/// space → keeps the symmetric/sextet part) and so can never produce the totally
/// antisymmetric singlet \f$ \varepsilon_{ijk} \f$. The fix is **distinct holes =
/// independent cycles**: the three inputs and the result are **four spatially
/// separated windows of three color holes each**, all on ONE connected base
/// surface, a 2-frequency geodesic icosahedron (\f$ S^2 \f$, 42 vertices) minus
/// 12 vertex-disjoint hole triangles, extruded \f$ \times I \f$ (`prismCells`).
/// The four windows are A, B, C (inputs) and R (the emergent result).
///
/// The windows are placed **symmetrically** (#398): one orbit of a tetrahedral
/// subgroup \f$ A_4 \f$ of the icosahedral rotation group, each window a \f$ C_3 \f$
/// orbit of three corner sub-triangles at one of the icosahedron's four tetrahedral
/// vertex-orbits. The windows are \f$ A_4 \f$-equivalent, so the per-window
/// period-transport blocks are cyclically related: the transport intertwines the
/// color \f$ \mathbb{Z}_3 \f$, and a color-symmetric (\f$ \omega \f$-representation)
/// input transports to the EXACT singlet with manifest \f$ S_3 \f$ — a greedy pick,
/// whose windows are geometrically inequivalent, reaches only \f$ \sim 0.74 \f$.
///
/// Confinement is then **conservation at the junction**: on a connected
/// surface-minus-holes the global Stokes relation forces
/// \f$ \sum_{\text{all holes}} (\text{induced period}) = 0 \f$, so
/// \f$ \Sigma_R = -\Sigma_{\text{inputs}} \f$ — three color-neutral inputs
/// (\f$ \Sigma = 0 \f$ each) give a color-neutral result. (Unlike the
/// single-register `RegisterTopology`, a colored input is NOT floored here at the
/// input level — its hole is an independent cycle — so candidate A conserves
/// charge rather than confining at the input; per-input confinement is the
/// heavier tube-connected junction, deferred.)
///
/// Read-out: each input window's three color holes carry that state's three color
/// amplitudes, the target periods pre-multiplied by the window's
/// induced-orientation covector (`ChainComplex::endSignCovector`, the
/// generalization of `RegisterTopology::kColorSign`). The result block R emerges
/// (`emergesResult() == true`); it is read off the relaxed geometry, never
/// pinned. The carried object is a color rep, so `MergeCobordism` reads a rep.
class TripartiteRegisterTopology : public TopologyBuilder {
  public:
    /// The van Raamsdonk MI normalization \f$ I_{\max} = 2\log 3 \f$ --- the
    /// mutual information of a maximally-entangled qutrit (color-triple) pair.
    static constexpr double kVanRaamsdonkMaxMI = 2.0 * 1.0986122886681098;

    [[nodiscard]] std::shared_ptr<Spacetime> build(
        std::size_t stateDim, std::uint64_t seed,
        std::vector<std::vector<std::uint64_t>> &boundaryCells) override;

    /// The EXACT triangle-hole read-out (the #353 period path). Pins the supplied
    /// input states on windows A, B, C over `residualForPeriods`; returns the
    /// first unpinned window (R) as `resultHoles` to be read after the relax.
    void readoutHoles(
        const std::shared_ptr<Spacetime> &cobordism,
        const std::vector<std::vector<std::complex<double>>> &states,
        std::vector<std::vector<std::uint64_t>> &inputHoles,
        std::vector<std::complex<double>> &inputTargets,
        std::vector<std::vector<std::uint64_t>> &resultHoles) const override;

    /// The result block (window R) EMERGES from the three pinned inputs; the merge
    /// pins inputs alone and reads the emergent result, so no outputStates/U.
    [[nodiscard]] bool emergesResult() const override { return true; }

    [[nodiscard]] std::size_t carriedDim(std::size_t /*stateDim*/) const override {
      return 2;  // the result lives in the S_3 standard rep (Sigma=0, b1 window)
    }

    [[nodiscard]] std::size_t loopsPerState() const override {
      return 3;  // the three color holes per register window
    }

    /// The color register is \f$ d = 3 \f$ (the color triple on \f$ \sum=0 \f$).
    void validateStateDim(std::size_t d) const override;

    [[nodiscard]] std::string name() const override {
      return "tripartite register ((S^2-12holes) x I, 4 windows A,B,C->R)";
    }

    /// Seed the metric from the color-singlet ENTANGLEMENT (van Raamsdonk) rather
    /// than random jitter: each same-window edge gets the intra-party mutual
    /// information `intraMI` (high MI → short, bound), each cross-window edge the
    /// inter-party `crossMI`, and bulk edges 0 (capped long). The per-edge length
    /// is \f$ l^2 = (-\log(I/\text{iMax}))^2 \f$ (`Edge::vanRaamsdonkSquaredLength`).
    /// The caller computes `intraMI`/`crossMI` from the 3-party state's reduced
    /// density matrices (the geometry from entanglement), so the SAME state seeds
    /// the metric here and the complex boundary inputs (passed as `MergeCobordism`
    /// inputStates). If never called, `build()` uses the uniform symmetric seed
    /// (\f$ l^2 = 1 \f$), which (like the party-based VR seed) respects the \f$ A_4 \f$
    /// window symmetry the singlet transport relies on.
    /// @param intraMI the within-party mutual information (party marginal entropy).
    /// @param crossMI the between-party mutual information.
    /// @param iMax    the max MI normalization (qutrit: \f$ 2\log 3 \f$).
    void setEntangledMetric(double intraMI, double crossMI,
                            double iMax = kVanRaamsdonkMaxMI);

    /// Make the junction LORENTZIAN: the cross-layer (forward-time) worldline
    /// edges are set timelike (\f$ l^2 = \text{worldlineLsq} < 0 \f$), so the dual
    /// Regge action goes complex (\f$ \mathrm{Im}\,S \neq 0 \f$) and its harmonics
    /// can carry the singlet's \f$ \omega \f$-phases (a real metric cannot). Null
    /// edges (photons) may EMERGE under the relax as a worldline's \f$ l^2 \to 0 \f$
    /// (lightlike). Unset (default): the junction is all-spacelike (Riemannian).
    /// @param worldlineLsq the timelike squared length for cross-layer edges (<0).
    void setLorentzianWorldlines(double worldlineLsq = -1.0);

    /// Set the geodesic subdivision **frequency** N (the tunable lattice
    /// granularity, \#404). The base surface is a frequency-N geodesic icosahedron:
    /// \f$ 12 + 30(N-1) + 20\binom{N-1}{2} \f$ vertices, \f$ 20N^2 \f$ faces. The
    /// four \f$ A_4 \f$-orbit windows are generated from the symmetry at any N, so
    /// the construction is unchanged in structure (\f$ N=2 \f$, the default, is the
    /// \#398 base of 42 vertices); larger N refines the lattice, shrinking the
    /// intertwining residual and driving the singlet overlap \f$ \to 1 \f$.
    /// @param frequency the subdivision frequency \f$ N \ge 2 \f$.
    void setFrequency(int frequency);

  private:
    // Lorentzian worldlines (set by setLorentzianWorldlines): when on, build()
    // sets cross-layer edges timelike.
    bool lorentzian_{false};
    double lorentzWorldlineLsq_{-1.0};

    // The geodesic subdivision frequency N (set by setFrequency); default 2 (the
    // #398 base). Larger N refines the lattice (tunable granularity, #404).
    int frequency_{2};

    // The van Raamsdonk metric seed (set by setEntangledMetric); when unset the
    // build uses the uniform symmetric seed (l^2 = 1).
    bool vrSeed_{false};
    double vrIntraMI_{0.0};
    double vrCrossMI_{0.0};
    double vrIMax_{kVanRaamsdonkMaxMI};  // 2 log 3 (max MI of a qutrit pair)

    // Cached by build() for readoutHoles(): the four windows (A,B,C,R) of three
    // color holes each (sorted absolute vertex triples), and their per-hole
    // induced-orientation signs (ChainComplex::endSignCovector of the base
    // surface). Windows live on ONE surface, so holes carry absolute ids (no
    // per-block stride offset, unlike RegisterTopology's layered blocks).
    std::vector<std::vector<std::vector<std::uint64_t>>> blockHoles_{};
    std::vector<std::vector<int>> signTable_{};
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_TRIPARTITEREGISTERTOPOLOGY_H
