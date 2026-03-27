// MIT License
// Copyright (c) 2025 Andrew Kelleher
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef CASET_CDT_H
#define CASET_CDT_H

#include "simulations/Simulation.h"
#include "spacetime/Spacetime.h"
#include <memory>
#include <random>
#include <vector>
#include <map>

namespace caset {

/// # Causal Dynamical Triangulations (CDT) Simulation
///
/// This class implements the Metropolis Monte Carlo algorithm for Causal Dynamical
/// Triangulations in \f$ d \f$ dimensions (typically \f$ d = 4 \f$). CDT is a
/// non-perturbative approach to quantum gravity introduced by Ambjorn, Jurkiewicz,
/// and Loll in which the gravitational path integral is approximated by a sum over
/// causal triangulations of spacetime.
///
/// ## The Regge Action
///
/// The Euclidean Regge action for CDT (after Wick rotation) is
///
/// \f[
///   S_{\text{Regge}} = -(k_0 + 6\Delta)\, N_0
///                     + (k_4 + 2\Delta)\, N_{41}
///                     + (k_4 + \Delta)\, N_{32}
/// \f]
///
/// where:
///   - \f$ N_0 \f$ is the number of vertices,
///   - \f$ N_{41} \f$ is the number of \f$(d,1) + (1,d)\f$-type simplices,
///   - \f$ N_{32} \f$ is the number of \f$(d\!-\!1, 2) + (2, d\!-\!1)\f$-type simplices,
///   - \f$ k_0 \f$ is a coupling related to the inverse bare Newton constant \f$ G_N^{-1} \f$,
///   - \f$ k_4 \f$ is a coupling related to the bare cosmological constant \f$ \Lambda \f$,
///   - \f$ \Delta \f$ is the asymmetry parameter encoding the ratio of timelike to
///     spacelike squared edge lengths via \f$ \ell_t^2 = -\alpha\, a^2 \f$.
///
/// An additional volume-fixing term
///
/// \f[
///   S_{\text{fix}} = \varepsilon \left( N_4 - \bar{N}_4 \right)^2
/// \f]
///
/// constrains the total four-volume \f$ N_4 = N_{41} + N_{32} \f$ to fluctuate
/// around the target value \f$ \bar{N}_4 \f$.
///
/// ## Metropolis Algorithm
///
/// Each Monte Carlo sweep proposes \f$ N_4 \f$ random local moves and accepts or
/// rejects each according to the Metropolis criterion:
///
/// \f[
///   P_{\text{accept}} = \min\!\left(1,\; e^{-\Delta S}\right)
/// \f]
///
/// where \f$ \Delta S = S_{\text{new}} - S_{\text{old}} \f$. This satisfies
/// detailed balance with respect to the CDT partition function
/// \f$ Z = \sum_{\mathcal{T}} e^{-S[\mathcal{T}]} \f$.
///
/// ## Pachner Moves
///
/// The local moves are Pachner (bistellar) moves that preserve the piecewise-linear
/// manifold structure and the causal (foliated) time slicing:
///
///   - **Add** (grow): Cone an external facet to a new vertex, adding one \f$ d \f$-simplex.
///   - **Remove** (shrink): Remove a \f$ d \f$-simplex whose apex vertex has no other
///     incident top-simplices, reversing an add move.
///   - **Flip** \f$(2, d)\f$: Replace two \f$ d \f$-simplices sharing a \f$(d\!-\!1)\f$-face
///     with \f$ d \f$ new simplices sharing an edge. This is the standard bistellar flip.
///   - **Shift** \f$(3, 3)\f$: Replace three \f$ d \f$-simplices sharing a \f$(d\!-\!2)\f$-face
///     with three new simplices sharing a different \f$(d\!-\!2)\f$-face.
///
/// These moves are ergodic: any causal triangulation of a given topology can be
/// reached from any other by a finite sequence of moves (Alexander's theorem).
///
/// ## Phase Structure
///
/// In 4D, the coupling-constant space \f$(k_0, \Delta)\f$ exhibits four phases:
///   - **Phase A** (branched polymer): elongated, fractal geometry,
///   - **Phase B** (crumpled): collapsed, high-connectivity geometry,
///   - **Phase \f$ C_{dS} \f$** (de Sitter): extended, four-dimensional geometry
///     whose volume profile \f$ N_3(t) \f$ matches the Euclidean four-sphere,
///   - **Phase \f$ C_b \f$** (bifurcation): exhibits a singular vertex.
///
/// The physically relevant phase is \f$ C_{dS} \f$, where the average spatial
/// volume profile follows
///
/// \f[
///   \langle N_3(t) \rangle \propto \cos^4\!\left(\frac{\pi\, t}{T}\right)
/// \f]
///
/// matching the round \f$ S^4 \f$ metric of Euclidean de Sitter space.
///
/// ## References
///
///   - Ambjorn, Jurkiewicz, Loll, *Reconstructing the Universe*, Phys. Rev. D 72 (2005)
///   - Gorlich, *Introduction to Causal Dynamical Triangulations* (2013)
///   - Loll, *Quantum Gravity from Causal Dynamical Triangulations: A Review*, Class. Quant. Grav. 37 (2020)
///
class CDT : public Simulation {
  public:
    /// Construct a CDT simulation for a given spacetime.
    ///
    /// The spacetime should already be initialized via `build()` before constructing the
    /// simulation. The coupling constants \f$(k_0, k_4, \Delta)\f$ determine which phase
    /// the simulation explores. Typical de Sitter phase values are
    /// \f$ k_0 \approx 2 \f$, \f$ \Delta \approx 0.6 \f$.
    ///
    /// @param spacetime The built spacetime triangulation to simulate
    /// @param k0 Coupling constant \f$ k_0 \f$ (related to \f$ G_N^{-1} \f$)
    /// @param k4 Coupling constant \f$ k_4 \f$ (related to \f$ \Lambda \f$)
    /// @param delta Asymmetry parameter \f$ \Delta \f$
    /// @param epsilon Volume-fixing strength \f$ \varepsilon \f$
    /// @param targetN41 Target \f$(d,1)\f$-type four-volume \f$ \tilde{N}_4 = N_4^{(4,1)} \f$
    /// @param quadraticVolumeFix If true, use \f$ \varepsilon(N_{41} - \tilde{N}_4)^2 \f$;
    ///   if false, use \f$ \varepsilon |N_{41} - \tilde{N}_4| \f$ (Reconstructing the Universe eq. 6)
    CDT(std::shared_ptr<Spacetime> spacetime, double k0, double k4, double delta,
        double epsilon, std::size_t targetN41, bool quadraticVolumeFix = true);

    /// \f$(2, 2d)\f$ vertex insertion move (Brunekreef Sec. 2.3.1, adapted to 4D).
    ///
    /// Picks a random \f$ N_{41} \f$-type simplex, finds its spatial \f$(d\!-\!1)\f$-face
    /// and the adjacent simplex of opposite orientation. Inserts a new vertex at the
    /// shared spatial time slice, replacing the 2 simplices with \f$ 2d \f$ new ones.
    ///
    /// @return true if the move was accepted by the Metropolis-Hastings criterion
    bool add();

    /// \f$(2d, 2)\f$ vertex deletion move (inverse of add).
    ///
    /// Picks a random vertex and checks if it has exactly \f$ 2d \f$ incident
    /// top-simplices with the structure of a previous vertex insertion. If so,
    /// collapses them back to 2 simplices sharing a spatial face.
    ///
    /// @return true if the move was accepted
    bool remove();

    /// Bistellar \f$(2, d)\f$ flip: replace two \f$ d \f$-simplices with \f$ d \f$ new ones.
    ///
    /// Finds a \f$(d\!-\!1)\f$-face shared by exactly two \f$ d \f$-simplices. The
    /// two simplices together span \f$ d + 2 \f$ vertices (\f$ d \f$ shared, 2 unique).
    /// They are replaced by \f$ d \f$ new simplices, each containing both unique
    /// vertices and \f$ d - 1 \f$ of the \f$ d \f$ shared vertices. The move preserves
    /// vertex count and topology.
    ///
    /// @return true if the move was accepted
    bool flip();

    /// \f$(3, 3)\f$ Pachner move: replace three simplices sharing a \f$(d\!-\!2)\f$-face
    /// with three new simplices sharing the complementary \f$(d\!-\!2)\f$-face.
    ///
    /// The three old simplices share a triangle (\f$ d - 2 \f$ face) of 3 vertices and
    /// have 3 unique vertices. The replacement simplices share the dual triangle
    /// formed by the 3 unique vertices.
    ///
    /// @return true if the move was accepted
    bool shift();

    /// Inverse bistellar \f$(d, 2)\f$ flip: replace \f$ d \f$ simplices sharing an edge
    /// with 2 simplices sharing a \f$(d\!-\!1)\f$-face.
    ///
    /// Finds an edge shared by exactly \f$ d \f$ top-simplices. The \f$ d \f$ simplices
    /// span \f$ d + 2 \f$ vertices (2 shared edge endpoints, \f$ d \f$ unique). They are
    /// replaced by 2 new simplices, each containing all \f$ d \f$ unique vertices
    /// and one of the 2 shared vertices.
    ///
    /// @return true if the move was accepted
    bool iflip();

    /// Deprecated alias for shift(). The \f$(3,3)\f$ move is self-inverse.
    bool ishift();

    /// Adjust the cosmological coupling \f$ k_4 \f$ to drive the total four-volume
    /// \f$ N_4 \f$ toward the target \f$ \bar{N}_4 \f$. Uses a proportional controller
    /// that increases \f$ k_4 \f$ when volume exceeds the target (making growth
    /// more expensive) and decreases it when volume is below target.
    void tune() override;

    /// Run Monte Carlo sweeps until the action \f$ S \f$ stabilizes, indicating
    /// the system has reached thermal equilibrium. Equilibrium is detected when
    /// the relative change in \f$ S \f$ between sweeps drops below 1%.
    void thermalize() override;

    /// Execute one Monte Carlo sweep: propose \f$ N_4 \f$ random moves (uniformly
    /// chosen among add, remove, flip, iflip, shift) and accept or reject each
    /// via the Metropolis-Hastings criterion.
    ///
    /// @return Number of accepted moves in this sweep
    int sweep();

    /// Evaluate the full Regge action \f$ S = S_{\text{Regge}} + S_{\text{fix}} \f$
    /// for the current triangulation state.
    ///
    /// @return The action value \f$ S \f$
    [[nodiscard]] double computeAction() const;

    /// Compute the spatial volume profile \f$ N_3(t) \f$: the number of top-dimensional
    /// simplices whose earliest vertex lies at time slice \f$ t \f$.
    ///
    /// In the de Sitter phase, the average of this quantity over many configurations
    /// approximates \f$ \langle N_3(t) \rangle \propto \cos^4(\pi t / T) \f$.
    ///
    /// @return Vector of simplex counts indexed by time slice offset
    [[nodiscard]] std::vector<int> getVolumeProfile() const;

    /// @return Acceptance rate (accepted/attempted) for each move type.
    [[nodiscard]] std::map<std::string, double> getAcceptanceRates() const;

    /// @return The spacetime being simulated.
    [[nodiscard]] std::shared_ptr<Spacetime> getSpacetime() const noexcept;

    /// @return Coupling constant \f$ k_0 \f$.
    [[nodiscard]] double getK0() const noexcept;

    /// @return Coupling constant \f$ k_4 \f$ (may be modified by tuning).
    [[nodiscard]] double getK4() const noexcept;

    /// @return Asymmetry parameter \f$ \Delta \f$.
    [[nodiscard]] double getDelta() const noexcept;

    /// Enable or disable vertex relabeling after add/remove moves.
    /// [BGL] Sec. 2.2.1: relabeling ensures correct detailed balance.
    /// Enabled by default. Disable for deterministic topology tests
    /// where fingerprint-based before/after comparison is needed.
    void setRelabelVertices(bool enabled) noexcept { relabelVertices_ = enabled; }

  private:
    std::shared_ptr<Spacetime> spacetime;
    double k0, k4, delta, epsilon;
    std::size_t targetN41;
    bool quadraticVolumeFix;
    bool relabelVertices_{true};
    std::mt19937 rng{std::random_device{}()};

    /// Metropolis-Hastings acceptance test.
    ///
    /// Accepts if \f$ \Delta S - \log P \le 0 \f$, otherwise accepts with
    /// probability \f$ e^{-\Delta S + \log P} \f$, where \f$ \log P \f$ is the
    /// log of the combinatorial prefactor \f$ g(T' \to T) P_l(T') / [g(T \to T') P_l(T)] \f$
    /// divided by \f$ e^{-\Delta S} \f$.
    bool accept(double deltaS, double logPrefactor = 0.0);

    /// Compute the incremental action change for a proposed move without
    /// recomputing the full sum over simplices.
    double computeDeltaAction(int dN0, int dN41, int dN32) const;

    /// Shared implementation for shift and ishift moves.
    bool shiftImpl();

    int addAttempts = 0, addAccepted = 0;
    int removeAttempts = 0, removeAccepted = 0;
    int flipAttempts = 0, flipAccepted = 0;
    int iflipAttempts = 0, iflipAccepted = 0;
    int shiftAttempts = 0, shiftAccepted = 0;
    int ishiftAttempts = 0, ishiftAccepted = 0;

#ifdef CASET_LAZY_CATALOGS
    std::vector<std::uint64_t> n41Catalog_; ///< Fingerprints of N41-type top simplices
    std::vector<std::uint64_t> removeCatalog_; ///< Vertex IDs with top-degree near 2d
    bool catalogsDirty_{true};
    /// Rebuild catalogs from scratch (only if dirty).
    void rebuildCatalogs();
    void markCatalogsDirty() { catalogsDirty_ = true; }
#endif
};

} // caset

#endif //CASET_CDT_H
