// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_CHAINCOMPLEX_H
#define TESSERA_COBORDISM_CHAINCOMPLEX_H

#include <complex>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;

/// # ChainComplex
///
/// The simplicial chain complex of a triangulation: the boundary maps
/// ∂_k : C_k → C_{k-1} over ℤ, plus the homology invariants derived from them
/// (Betti numbers over ℤ and ℤ/2, torsion coefficients) and the ∂²=0 sanity
/// check. Purely combinatorial — built from the vertex sets of the complex,
/// independent of any geometry.
///
/// Simplices are enumerated as the full face-closure of the complex's simplices
/// (every sub-face of every registered simplex). Each k-simplex is identified
/// by its sorted vertex-id tuple and assigned an index; its reference
/// orientation is the increasing-vertex-id ordering, so
/// ∂[v_0 < … < v_k] = Σ_i (−1)^i [v_0,…,v̂_i,…,v_k].
class ChainComplex {
  public:
    /// Build the chain complex from a triangulation (a Spacetime). Reads vertex
    /// sets only; no coordinates/geometry required.
    [[nodiscard]] static ChainComplex fromSpacetime(const Spacetime &K);

    /// Build the chain complex of a pure simplicial complex from its top cells
    /// alone, as vertex-id tuples, with **no geometry and no mesh**: the face
    /// closure is every subset of every top cell, each simplex is oriented by
    /// **ascending vertex id** (the reference orientation), and
    /// \f$ \partial[v_0<\dots<v_k] = \sum_i (-1)^i [v_0<\dots\widehat{v_i}\dots<v_k] \f$.
    /// The cell order is the same canonical (lexicographic on sorted tuples)
    /// order `fromSpacetime` produces. Vertex order within a supplied cell is
    /// irrelevant; a cell with a repeated vertex or a size differing from the
    /// others is refused.
    /// @throws std::invalid_argument on an impure cell list or a degenerate cell.
    [[nodiscard]] static ChainComplex fromTopCells(
        const std::vector<std::vector<std::uint64_t>> &topCells);

    /// Top dimension n (largest k with a k-simplex), or -1 if empty.
    [[nodiscard]] int dimension() const noexcept { return dimension_; }

    /// |C_k|, the number of k-simplices (0 if k out of range).
    [[nodiscard]] std::size_t numSimplices(int k) const noexcept;

    /// f-vector (|C_0|, …, |C_n|).
    [[nodiscard]] const std::vector<std::size_t> &fVector() const noexcept { return counts_; }

    /// Euler characteristic χ = Σ_k (−1)^k |C_k|.
    [[nodiscard]] int eulerCharacteristic() const noexcept;

    /// The boundary matrix ∂_k (rows = |C_{k-1}|, cols = |C_k|), flat row-major.
    /// Entries in {−1, 0, +1}. ∂_0 is empty. Out-of-range k returns an empty matrix.
    [[nodiscard]] const std::vector<long> &boundaryMatrix(int k) const;

    /// Check ∂_{k-1} ∘ ∂_k = 0 for all k (chain-complex axiom / V3 sanity check).
    [[nodiscard]] bool boundaryComposesToZero() const;

    /// Whether the Poincaré/Lefschetz **dual block decomposition** of a pure
    /// \f$ n \f$-complex is a valid cell complex — equivalently, whether the
    /// primal is a combinatorial manifold-with-boundary. Decidable for
    /// \f$ n \le 3 \f$ by the classification of surfaces; checked as: facet
    /// coface counts in \f$ \{1, 2\} \f$; no dangling facets (when
    /// `facetCells` is non-empty it is the full \f$ (n-1) \f$-cell universe —
    /// e.g. the cells a Hodge Laplacian is built over — and every entry must
    /// be carried by at least one top cell); ridge links single paths/cycles
    /// (no pinches); and, at \f$ n = 3 \f$, vertex links that are 2-spheres
    /// (interior) or disks (boundary), decided by connectivity, Euler
    /// characteristic, and a single boundary circle. Returns (ok, reason)
    /// with the first violation named. Pure combinatorics on sorted
    /// vertex-id tuples — no geometry required. Topology-changing moves
    /// should be accepted only if this survives: validity in the dual space,
    /// not merely scoreability on the primal lattice.
    [[nodiscard]] static std::pair<bool, std::string> dualComplexIsValid(
        const std::vector<std::vector<std::uint64_t>> &topCells, int dim,
        const std::vector<std::vector<std::uint64_t>> &facetCells = {});

    /// Betti numbers b_0..b_n over ℚ (free ranks of H_k):
    /// b_k = |C_k| − rank ∂_k − rank ∂_{k+1}.
    [[nodiscard]] std::vector<int> bettiNumbers() const;

    /// Betti numbers over GF(2): b_k = |C_k| − rank₂ ∂_k − rank₂ ∂_{k+1}.
    [[nodiscard]] std::vector<int> bettiNumbersGF2() const;

    /// Torsion coefficients of \f$ H_k \f$: the invariant factors \f$ > 1 \f$
    /// of \f$ \partial_{k+1} \f$. (E.g. \f$ \mathbb{RP}^2 \f$ has
    /// \f$ \mathrm{torsion}(1) = \{2\} \f$.)
    [[nodiscard]] std::vector<long> torsion(int k) const;

    /// The k-simplices as sorted vertex-id tuples, in the canonical order of
    /// \f$ C_k \f$ — the column order of \f$ \partial_{k+1} \f$ and, equally, the
    /// row order of \f$ \partial_k \f$. For \f$ k = \f$ dimension() this is
    /// orientedTopSimplices(); for \f$ k = 1 \f$ it is the edge ordering the
    /// rows of boundaryMatrix(2) refer to, which is needed to read a
    /// \f$ \mathbb{Z}_2 \f$ connection \f$ g \in C^1 \f$ on a simplex's edges.
    /// Empty when \f$ k \f$ is out of range.
    [[nodiscard]] std::vector<std::vector<std::uint64_t>> kSimplexVertices(int k) const;

    /// The top simplices as sorted vertex-id tuples, in the canonical column
    /// order of the top boundary matrix \f$ \partial_d \f$ (\f$ d = \f$
    /// dimension()). This is the ordering the orientation signs from
    /// fundamentalClass() refer to. Empty for the empty complex.
    [[nodiscard]] std::vector<std::vector<std::uint64_t>> orientedTopSimplices() const;

    /// The fundamental class \f$ [W] \in H_d \f$ of a closed oriented
    /// \f$ d \f$-manifold: the per top-simplex orientation signs
    /// \f$ \varepsilon_t = \pm 1 \f$ (indexed as orientedTopSimplices(), i.e. the
    /// columns of \f$ \partial_d \f$) that make the top chain
    /// \f$ \sum_t \varepsilon_t\, t \f$ a cycle
    /// (\f$ \partial_d \sum_t \varepsilon_t\, t = 0 \f$, so each codimension-one
    /// face's two incident top simplices cancel). It is the \f$ \pm 1 \f$
    /// generator of \f$ \ker \partial_d \f$, which is one-dimensional
    /// (\f$ b_d = 1 \f$) for a closed connected oriented manifold, hence unique
    /// up to an overall sign; the sign is fixed deterministically by making the
    /// first nonzero entry \f$ +1 \f$.
    /// @throws std::runtime_error if no such class exists — the complex is not a
    ///   closed connected oriented \f$ d \f$-manifold (\f$ \dim \ker \partial_d
    ///   \neq 1 \f$) — or \f$ d < 1 \f$.
    [[nodiscard]] std::vector<int> fundamentalClass() const;

    /// The **end sign covector**: the induced-orientation charge pattern
    /// \f$ \sigma \in \{\pm 1\}^{|\text{holes}|} \f$ of an end surface, read off
    /// its fundamental chain. `surfaceCells` are the end's top cells (sorted
    /// vertex-id tuples, all of one dimension) and `holes` the removed cells
    /// whose boundary cycles carry the periods; the union is oriented by sign
    /// propagation across shared facets (facet \f$ j \f$ of a sorted cell is the
    /// drop-\f$ v_j \f$ tuple with boundary sign \f$ (-1)^j \f$), each connected
    /// component normalized so its lexicographically smallest cell carries
    /// \f$ +1 \f$, and \f$ \sigma_k \f$ is the coefficient the orientation
    /// \f$ \varepsilon \f$ assigns to `holes[k]`. Since
    /// \f$ \partial\big(\sum_{\text{kept}} \varepsilon_c\, c\big) =
    /// -\sum_k \varepsilon_{H_k}\, \partial H_k \f$ on the hole cycles, every
    /// closed form's signed periods obey \f$ \sum_k \sigma_k p_k = 0 \f$ end by
    /// end — the symmetrized charge constraint the register layers apply — for
    /// **either** global sign; this normalization is the deterministic one
    /// (a property of the end surface, not of any fill or spectrum: no
    /// per-fill null-vector normalization, which is sign-unstable). Each
    /// entry is independent of the others, so reordering `holes` permutes
    /// \f$ \sigma \f$ without flips, and relabeling the vertices by any
    /// order-preserving map (e.g. a layer shift) leaves it unchanged.
    /// @throws std::runtime_error if the cells are not all of one dimension,
    ///   a facet has more than two cofaces (not a pseudomanifold), or the
    ///   orientation propagation contradicts itself (non-orientable).
    [[nodiscard]] static std::vector<int> endSignCovector(
        const std::vector<std::vector<std::uint64_t>> &surfaceCells,
        const std::vector<std::vector<std::uint64_t>> &holes);

    /// The **induced-orientation covector** of a whole top-cell complex: the
    /// per-cell sign \f$ \varepsilon_t \in \{\pm 1\} \f$ assigned by orienting
    /// the complex through facet-sharing propagation. Each `topCells` entry is
    /// a sorted vertex-id tuple, all of one dimension; the returned vector is
    /// aligned to the **sorted-unique** (lexicographic) order of those cells —
    /// the canonical \f$ C_d \f$ column order, i.e. the order of
    /// orientedTopSimplices() / kSimplexVertices(dimension()). Roots of each
    /// connected component (its lexicographically smallest cell) carry
    /// \f$ +1 \f$ and the sign propagates across every shared facet by the same
    /// rule endSignCovector() uses (facet \f$ j \f$ of a sorted cell carries
    /// boundary sign \f$ (-1)^j \f$, and across an interior facet the two
    /// induced signs cancel: \f$ \varepsilon_b = -\varepsilon_a s_a s_b \f$).
    /// Unlike fundamentalClass() this does **not** require the complex to be
    /// closed — boundary facets (one coface) simply impose no constraint — so it
    /// is the orientation read-out for an open refinement region (e.g. a single
    /// stellar cone star, or a Lorentzian CDT slab with boundary). Determined
    /// purely combinatorially, independent of geometry, vertex labels (any
    /// order-preserving relabel leaves it unchanged), and the order `topCells`
    /// is supplied in. Empty for the empty complex.
    /// @throws std::runtime_error if the cells are not all of one dimension, a
    ///   facet has more than two cofaces (not a pseudomanifold), or the
    ///   orientation propagation contradicts itself (non-orientable) — exactly
    ///   the conditions a surgical cone must never silently introduce.
    [[nodiscard]] static std::vector<int> orientationCovector(
        const std::vector<std::vector<std::uint64_t>> &topCells);

    /// The symmetric intersection form \f$ Q_{ij} = \langle \alpha_i \cup
    /// \alpha_j, [K] \rangle \f$ on a basis \f$ \{\alpha_i\} \f$ of the free
    /// part of \f$ H^2 \f$, as a flat row-major \f$ b_2 \times b_2 \f$ matrix.
    /// Defined for a closed oriented 4-manifold (\f$ n = 4 \f$); the cup product
    /// is the Alexander–Whitney product evaluated on the fundamental class
    /// \f$ [K] \f$ (the generator of \f$ \ker \partial_4 \f$). Empty when
    /// \f$ n \neq 4 \f$ or \f$ b_2 = 0 \f$.
    /// @throws std::runtime_error if \f$ n = 4 \f$, \f$ b_2 > 0 \f$, but the
    ///   complex is not closed-orientable (no fundamental class).
    [[nodiscard]] std::vector<double> intersectionForm() const;

    /// Signature \f$ \sigma = b_+ - b_- \f$ of the intersection form
    /// (Sylvester inertia). 0 when \f$ n \neq 4 \f$ or \f$ b_2 = 0 \f$.
    [[nodiscard]] int signature() const;

    /// # Cup-product pairing against the fundamental class
    ///
    /// The bilinear form \f$ \langle a \cup b, [K] \rangle = \sum_t
    /// \varepsilon_t\, a(\text{front}_k t)\, b(\text{back}_{d-k} t) \f$ on
    /// \f$ C^k \times C^{d-k} \f$ of a closed connected oriented
    /// \f$ d \f$-manifold, kept in its sparse combinatorial form: one term per
    /// top simplex \f$ t = [v_0 < \cdots < v_d] \f$, whose *front* face is
    /// \f$ [v_0 \ldots v_k] \f$ and *back* face \f$ [v_k \ldots v_d] \f$ (the
    /// Alexander–Whitney product; both faces inherit the ascending reference
    /// orientation, so no extra sign enters), weighted by the fundamental-class
    /// sign \f$ \varepsilon_t \f$ of ``fundamentalClass()``.
    ///
    /// WHY it is exposed on cochains rather than on a cohomology basis (as
    /// ``intersectionForm()`` is): a metric picks *representatives*. The
    /// Whitney chain-Hodge pencil hands back the harmonic images
    /// \f$ z_a \f$ of a given geometry, and the surface's complex structure is
    /// the pairing of those images, \f$ R_{ab} = \int W(z_a) \wedge W(z_b) =
    /// \langle z_a \cup z_b, [K] \rangle \f$ — an identity on closed cochains
    /// because both sides are the cup-product pairing of the classes. The
    /// pairing is metric-free, so it is the one piece of the complex structure
    /// that carries no discretization error; the geometry enters only through
    /// the harmonic Gram. For \f$ d = 2, k = 1 \f$ it is the (antisymmetric)
    /// intersection form of the surface. The 4-manifold ``intersectionForm()``
    /// is its \f$ d = 4, k = 2 \f$ case restricted to a basis of \f$ H^2 \f$.
    struct CupProductForm {
        /// The front degree \f$ k \f$; the back degree is \f$ d - k \f$.
        int degree{0};
        /// \f$ |C_k| \f$ and \f$ |C_{d-k}| \f$: the lengths ``evaluate`` expects.
        std::size_t rows{0};
        std::size_t cols{0};
        /// Per top simplex (canonical \f$ C_d \f$ order): the canonical index of
        /// its front \f$ k \f$-face, its back \f$ (d-k) \f$-face, and
        /// \f$ \varepsilon_t \in \{\pm 1\} \f$.
        std::vector<int> front{};
        std::vector<int> back{};
        std::vector<int> orientation{};
        /// \f$ \sum_t \varepsilon_t\, a[\text{front}_t]\, b[\text{back}_t] \f$
        /// for a \f$ k \f$-cochain \f$ a \f$ and a \f$ (d-k) \f$-cochain
        /// \f$ b \f$ in canonical order.
        /// @throws std::invalid_argument on a length mismatch.
        [[nodiscard]] std::complex<double> evaluate(
            const std::vector<std::complex<double>> &a,
            const std::vector<std::complex<double>> &b) const;
    };

    /// The cup-product pairing \f$ C^k \times C^{d-k} \to \mathbb{R} \f$ of
    /// this complex against its fundamental class.
    /// @throws std::invalid_argument when \f$ k \notin [0, d] \f$;
    ///   std::runtime_error when the complex has no fundamental class (not a
    ///   closed connected oriented manifold; see ``fundamentalClass()``).
    [[nodiscard]] CupProductForm cupProductForm(int k) const;

    /// Mod-2 Stiefel–Whitney numbers of a closed PL \f$ n \f$-manifold:
    /// \f$ \langle w_{i_1}\cdots w_{i_r}, [K] \rangle \in \mathbb{Z}/2 \f$ for
    /// every partition \f$ (i_1,\dots,i_r) \f$ of \f$ n \f$ into positive parts,
    /// keyed by the monomial (e.g. ``"w4"``, ``"w2^2"``, ``"w1^2"``,
    /// ``"w1w3"``). Empty when the complex is empty.
    ///
    /// Computed combinatorially: the mod-2 cohomology \f$ H^*(K;\mathbb{Z}/2) \f$
    /// (from the boundary maps reduced mod 2), the Alexander–Whitney cup product
    /// on cochains, and the Wu classes \f$ v_k \f$ — defined by
    /// \f$ \langle v_k \cup x, [K]\rangle = \langle \mathrm{Sq}^k x, [K]\rangle \f$
    /// for all \f$ x \in H^{n-k} \f$ — from which the total Stiefel–Whitney class
    /// is \f$ w = \mathrm{Sq}(v) \f$. The fundamental class \f$ [K] \f$ over
    /// \f$ \mathbb{Z}/2 \f$ is the sum of all top simplices, and evaluation on it
    /// is the mod-2 sum of a top-degree cochain over those simplices.
    ///
    /// Only the Steenrod squares expressible through the ordinary
    /// (\f$ \cup_0 \f$) product are implemented: \f$ \mathrm{Sq}^k \f$ on a
    /// class of degree \f$ k \f$ (the squaring) and the degree-forced zeros.
    /// These suffice for surfaces and for closed orientable 4-manifolds with
    /// \f$ b_1 = 0 \f$ (so \f$ \mathbb{RP}^2 \f$, \f$ \mathbb{CP}^2 \f$,
    /// \f$ S^4 \f$, \f$ S^2\times S^2 \f$, and their disjoint unions).
    /// @throws std::runtime_error if a required Wu or Stiefel–Whitney class
    ///   genuinely needs a higher cup-\f$ i \f$ product (\f$ i>0 \f$) — the
    ///   general Steenrod squares are deferred (see issue #65).
    [[nodiscard]] std::map<std::string, int> stiefelWhitneyNumbers() const;

    /// Rank of \f$ \partial_k \f$ over \f$ \mathbb{Q} \f$, exact (0 if \f$ k \f$ is out of range).
    [[nodiscard]] int rankOfBoundary(int k) const;

    /// The sign \f$ D_k[j] = \pm1 \f$ relating each stored \f$ k \f$-simplex
    /// orientation to the reference orientation (ascending vertex id,
    /// `fromTopCells`), per degree: the stored boundary maps satisfy
    /// \f$ \partial_k^{\rm stored} = D_{k-1}\,\partial_k^{\rm ref}\,D_k \f$ exactly.
    /// Computed recursively from the stored maps (\f$ D_0 = +1 \f$; at degree
    /// \f$ k \f$ the facet dropping the smallest vertex carries the reference
    /// coefficient \f$ +1 \f$, so \f$ D_k[j] \f$ is that facet's stored entry times
    /// \f$ D_{k-1} \f$ of the facet). A complex built by `fromTopCells` has every
    /// sign \f$ +1 \f$.
    /// @throws std::runtime_error if the stored maps are not a signed-permutation
    ///   image of the reference maps (a facet entry missing or of modulus ≠ 1).
    [[nodiscard]] std::vector<std::vector<int>> orientationSigns() const;

  private:
    int dimension_{-1};
    std::vector<std::size_t> counts_{};                 // |C_k|
    std::vector<std::vector<long>> boundary_{};         // boundary_[k] = ∂_k
    // faceVerts_[k][j] = sorted vertex ids of the j-th k-simplex (column j of
    // ∂_{k+1} / row j of ∂_k). Needed by the cup product (front/back faces).
    std::vector<std::vector<std::vector<std::uint64_t>>> faceVerts_{};
    [[nodiscard]] int gf2RankOfBoundary(int k) const;   // rank ∂_k over GF(2)
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_CHAINCOMPLEX_H
