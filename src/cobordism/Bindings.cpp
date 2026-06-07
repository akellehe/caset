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

// Pybind11 bindings for the cobordism subsystem. Lives outside tessera_core
// (which is pybind-free) so the static library can be reused without pulling
// in the Python dependency. This translation unit is always added to
// _tessera's sources (see CMakeLists.txt, TESSERA_PYBIND_SOURCES).

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cobordism/GeometrySynthesizer.h"
#include "cobordism/ChainComplex.h"
#include "cobordism/Characteristic.h"
#include "cobordism/Cobordism.h"
#include "cobordism/CombinatorialDimension.h"
#include "cobordism/DijkgraafWitten.h"
#include "cobordism/EigenstateSynthesis.h"
#include "cobordism/HodgeLaplacian.h"
#include "cobordism/IntegerLinalg.h"
#include "spacetime/Spacetime.h"  // complete type required by pybind (typeid)

namespace py = pybind11;

using namespace tessera;
using namespace tessera::cobordism;

void register_cobordism(py::module_ m) {
  // Smoke hook: lets tests assert the subsystem loaded before any
  // mathematical capability (issues #63–#70) is implemented. Single leading
  // underscore (not double) to avoid Python name-mangling inside test classes.
  m.def("_cobordism_smoke", [] { return true; },
        "Returns True; confirms the cobordism subsystem is built and importable.");

  // Per-complex scalar measurements are Observables (tessera's convention),
  // not methods on Spacetime or a bespoke wrapper. The characteristic-number
  // capabilities (Euler characteristic, signature, …) follow the same pattern;
  // multi-complex / structural operations (cobordism verification,
  // reconstruction, Pachner search) will be static-only classes taking a
  // Spacetime.
  py::class_<CombinatorialDimension, std::shared_ptr<CombinatorialDimension>>(
      m, "CombinatorialDimension",
      R"doc(Observable: combinatorial dimension of a triangulation.

The largest k with a k-simplex present (max simplex size - 1), or -1 if empty.
A purely combinatorial/topological integer (= n for a PL n-manifold), distinct
from the spectral dimension (a real-valued diffusion quantity) and from the
Spacetime's declared metric dimension.)doc")
      .def(py::init<>())
      .def("compute", &CombinatorialDimension::compute, py::arg("spacetime"),
           "Return the combinatorial dimension of the given Spacetime as a double.");

  // ----- Homology backbone (#64): chain complex + exact linear algebra -----

  py::class_<ChainComplex>(m, "ChainComplex",
      R"doc(Simplicial chain complex of a triangulation.

Boundary maps ∂_k over ℤ plus the homology invariants derived from them — Betti
numbers (over ℚ and GF(2)), torsion coefficients, Euler characteristic, and the
∂²=0 sanity check. Purely combinatorial (built from vertex sets; no geometry).)doc")
      .def_static("fromSpacetime", &ChainComplex::fromSpacetime, py::arg("spacetime"),
                  "Build the chain complex from a triangulation (a Spacetime).")
      .def("dimension", &ChainComplex::dimension)
      .def("numSimplices", &ChainComplex::numSimplices, py::arg("k"))
      .def("fVector", &ChainComplex::fVector)
      .def("eulerCharacteristic", &ChainComplex::eulerCharacteristic)
      .def("boundaryMatrix", &ChainComplex::boundaryMatrix, py::arg("k"),
           "Flat row-major ∂_k (rows=|C_{k-1}|, cols=|C_k|), entries in {-1,0,1}.")
      .def("boundaryComposesToZero", &ChainComplex::boundaryComposesToZero,
           "True iff ∂_{k-1}∘∂_k = 0 for all k.")
      .def("bettiNumbers", &ChainComplex::bettiNumbers, "Betti numbers b_0..b_n over Q.")
      .def("bettiNumbersGF2", &ChainComplex::bettiNumbersGF2, "Betti numbers over GF(2).")
      .def("torsion", &ChainComplex::torsion, py::arg("k"),
           "Torsion coefficients of H_k (invariant factors > 1 of d_{k+1}).")
      .def("kSimplexVertices", &ChainComplex::kSimplexVertices, py::arg("k"),
           "k-simplices as sorted vertex-id tuples in C_k order (the column "
           "order of d_{k+1} / row order of d_k). k=1 gives the edge ordering "
           "the rows of boundaryMatrix(2) refer to; k=dimension() equals "
           "orientedTopSimplices(). Empty when k is out of range.")
      .def("orientedTopSimplices", &ChainComplex::orientedTopSimplices,
           "Top simplices as sorted vertex-id tuples, in the canonical column "
           "order of the top boundary matrix d_d (d = dimension()); the order "
           "the fundamentalClass() signs refer to. Empty for the empty complex.")
      .def("fundamentalClass", &ChainComplex::fundamentalClass,
           "Fundamental class [W] in H_d: the per top-simplex orientation signs "
           "eps_t = +/-1 (the +/-1 generator of ker d_d) making the top chain a "
           "cycle (d_d applied to the signed top chain is 0). Sign-normalized so "
           "the first nonzero entry is +1. Raises if the complex is not a closed "
           "connected oriented manifold (dim ker d_d != 1) or dimension < 1.")
      .def("intersectionForm", &ChainComplex::intersectionForm,
           "Symmetric intersection form on free H^2 (flat b2 x b2), for a closed "
           "oriented 4-manifold; empty if n != 4 or b2 == 0.")
      .def("signature", &ChainComplex::signature,
           "Signature b+ - b- of the intersection form (0 if n != 4 or b2 == 0).")
      .def("stiefelWhitneyNumbers", &ChainComplex::stiefelWhitneyNumbers,
           "Mod-2 Stiefel-Whitney numbers <w_{i1}..w_{ir}, [K]> keyed by "
           "monomial (e.g. 'w4', 'w2^2'); empty for the empty complex. Raises "
           "if a class needs a deferred higher Steenrod cup-i product (#65).");

  // ----- Hodge Laplacian: k=0 Hermitian graph (#90), k>=1 metric Hodge (#104) -----
  py::class_<HodgeLaplacian>(m, "HodgeLaplacian",
      R"doc(Hodge Laplacian on a Spacetime, degree-parameterized by int k.

k=0: the U(1)-weighted graph Laplacian L = D - A on the 1-skeleton, assembled
from each edge's complex weight squaredLength * exp(i*phase). Vertices are
indexed by sorted id (0..N-1). Adjacency is Hermitian (the reverse orientation
negates the phase); the degree uses the magnitude convention
D_ii = sum |squaredLength|. (Unchanged.)

k>=1: the metric Hodge Laplacian from the integer boundary maps d_k, d_{k+1}
(ChainComplex) and diagonal weights W_k = the per-k-simplex Euclidean volumes
(Simplex.volume; W_0 = I). With d_k* = W_k^-1 d_k^T W_{k-1}, the operator is
L_k = d_k* d_k + d_{k+1} d_{k+1}*, returned in its symmetric (W_k-orthonormal)
form W_k^{1/2} L_k W_k^{-1/2} = B_k^T B_k + B_{k+1} B_{k+1}^T,
B_k = W_{k-1}^{1/2} d_k W_k^{-1/2} (symmetric PSD; SelfAdjointEigenSolver). By the
discrete Hodge theorem ker L_k ~= H_k, so dim ker L_k = b_k for any positive
weights; metric=False uses unit weights (the combinatorial d_k^T d_k +
d_{k+1} d_{k+1}^T) as a same-kernel cross-check. k-cells follow the canonical
ChainComplex column order, so the matrices align with boundaryMatrix(k) and
weights(k). Negative k raises; k above the top dimension yields empty results.
Spectra are computed lazily and cached. This is the operator only — fluxes,
cycle bases, and Betti numbers belong to WilsonLoop / ChainComplex.

Lorentzian d'Alembertian (§5.6): the lorentzian* methods weight W_k with the
SIGNED Simplex.volume (timelike l^2 < 0 ⇒ negative volumes), so the inner product
goes indefinite and L_k = d_k* d_k + d_{k+1} d_{k+1}* is assembled directly and is
generally non-self-adjoint — eigenvalues may be negative or complex. ker L_k ~= H_k
degrades: 'harmonic' becomes the small-|lambda| near-kernel and a representative h
can be null (<h,h>_W = sum_i W_{k,i}|h_i|^2 ~= 0). All-spacelike ⇒ reproduces the
Euclidean spectrum/kernel.)doc")
      .def(py::init<std::shared_ptr<Spacetime>>(), py::arg("spacetime"),
           "Build the Hodge Laplacian operator over a triangulation.")
      .def("adjacency", &HodgeLaplacian::adjacency,
           "Weighted adjacency A as a flat row-major N*N complex array "
           "(Hermitian; A_ij = sum squaredLength * exp(i*phase)).")
      .def("degree", &HodgeLaplacian::degree,
           "Degree vector (length N, real): D_ii = sum |squaredLength| over "
           "incident edges (magnitude convention).")
      .def("laplacian", &HodgeLaplacian::laplacian, py::arg("k") = 0,
           py::arg("metric") = true, py::arg("lorentzian") = false,
           "Laplacian L_k as a flat row-major complex array: N*N for k=0 "
           "(L = D - A), else |C_k|*|C_k| (the symmetric metric Laplacian; "
           "imag 0). metric=False uses unit weights (combinatorial) for k>=1 and "
           "is ignored at k=0. lorentzian=True (k>=1) assembles the signed-weight "
           "d'Alembertian directly (generally non-symmetric; still real). Raises "
           "for k<0; empty above the top dimension.")
      .def("weights", &HodgeLaplacian::weights, py::arg("k"),
           py::arg("lorentzian") = false,
           "Diagonal inner-product weights W_k (length |C_k|) in ChainComplex "
           "column order: per-k-simplex |volume| (W_0 = I). lorentzian=True returns "
           "the signed volume (timelike cells negative). Empty for k<0 or k above "
           "the top dimension.")
      .def("isHermitian", &HodgeLaplacian::isHermitian, py::arg("tol") = 1e-12,
           "True iff ||L - L^dagger|| <= tol (Frobenius) for the k=0 Laplacian.")
      .def("unitarityResidual", &HodgeLaplacian::unitarityResidual,
           py::arg("t") = 1.0,
           "Residual ||U U^dagger - I|| of U = e^{-iLt} formed from the "
           "eigendecomposition (~0 for the Hermitian L).")
      .def("eigenvalues", &HodgeLaplacian::eigenvalues, py::arg("k") = 0,
           py::arg("metric") = true,
           "Eigenvalues of L_k (real, ascending). metric selects volume vs. unit "
           "weights for k>=1 (ignored at k=0). Raises for k<0; empty above the "
           "top dimension.")
      .def("eigenvectors", &HodgeLaplacian::eigenvectors, py::arg("k") = 0,
           py::arg("metric") = true,
           "Eigenvectors of L_k as a flat row-major M*M complex array (column j "
           "is the eigenvector for the j-th ascending eigenvalue). metric selects "
           "volume vs. unit weights for k>=1. Raises for k<0; empty above the top "
           "dimension.")
      .def("harmonics", &HodgeLaplacian::harmonics, py::arg("k") = 0,
           py::arg("tol") = 1e-9, py::arg("metric") = true,
           "Harmonic representatives (eigenvectors with |lambda| < tol) as a flat "
           "row-major M*H complex array whose H columns span ker L_k ~= H_k "
           "(H = b_k). metric selects volume vs. unit weights for k>=1. Raises "
           "for k<0; empty above the top dimension.")
      // ----- Lorentzian (signed-weight) d'Alembertian (#105, spec §5.6) -----
      .def("lorentzianEigenvalues", &HodgeLaplacian::lorentzianEigenvalues,
           py::arg("k"), py::arg("metric") = true,
           "Eigenvalues of the signed-weight d'Alembertian L_k (k>=1) as complex "
           "numbers sorted ascending by (Re, Im): may be negative or complex-"
           "conjugate pairs (the indefinite metric ⇒ non-self-adjoint operator). "
           "On an all-spacelike complex they reproduce eigenvalues(k). Raises for "
           "k<0; empty above the top dimension.")
      .def("lorentzianEigenvectors", &HodgeLaplacian::lorentzianEigenvectors,
           py::arg("k"), py::arg("metric") = true,
           "Eigenvectors of the signed-weight L_k as a flat row-major M*M complex "
           "array; column j is the eigenvector for lorentzianEigenvalues(k)[j].")
      .def("lorentzianHarmonics", &HodgeLaplacian::lorentzianHarmonics,
           py::arg("k"), py::arg("tol") = 1e-9, py::arg("metric") = true,
           "Near-kernel ('harmonic') representatives of the d'Alembertian "
           "(eigenvectors with |lambda| < tol) as a flat row-major M*H complex "
           "array. H = b_k on an all-spacelike complex; with timelike cells the "
           "count can differ (pseudo-Hodge decomposition).")
      .def("lorentzianNullNorms", &HodgeLaplacian::lorentzianNullNorms,
           py::arg("k"), py::arg("tol") = 1e-9, py::arg("metric") = true,
           "Indefinite W-norms <h,h>_W = sum_i W_{k,i} |h_i|^2 of the near-kernel "
           "representatives, one per column of lorentzianHarmonics (same order). "
           "A value ~0 flags a NULL (lightlike) harmonic; all positive on an "
           "all-spacelike complex.");

  // ----- §4b eigenstate synthesis: residual + parameter access (#133) -----
  py::class_<EigenstateSynthesis>(m, "EigenstateSynthesis",
      R"doc(§4b inverse eigenvector problem on a fixed complex.

Scores how close the complex's current Hermitian edge weights make a target
state psi to being an eigenvector of the k=0 Hodge Laplacian L = D - A (the
magnitude convention, via HodgeLaplacian), and reads/writes those weights so a
search can perturb them. The non-convex, multi-restart search itself (e.g.
scipy.optimize.minimize L-BFGS-B over the flat {w_ij} + {theta_ij} vector) lives
in the driver and calls residual() here; the cone-and-retry growth loop is a
separate stage (this class is fixed-complex only).

Residual: for a unit target, r(psi) = ||(I - psi psi^dagger) L psi||^2 =
||L psi - lambda psi||^2 with lambda = psi^dagger L psi, so r = 0 iff
L psi || psi (psi is an eigenvector) and the realized eigenvalue is the Rayleigh
quotient lambda. A non-unit psi is normalized internally. L is reassembled from
the live edge weights/phases on every call, so residual() tracks setWeights /
setPhases in place. psi is indexed in the same sorted-vertex-id order as
HodgeLaplacian (k=0).

Parameters: the per-edge squared-length magnitudes {w_ij} (Edge.setSquaredLength)
and U(1) phases {theta_ij} (Edge.setPhase), in a stable edge order fixed at
construction (the weight-carrying edges: both endpoints present, no self-loops).)doc")
      .def(py::init<std::shared_ptr<Spacetime>>(), py::arg("spacetime"),
           "Build the synthesizer over a fixed triangulation.")
      .def("order", &EigenstateSynthesis::order,
           "Number of vertices N — the required length of any psi.")
      .def("numEdges", &EigenstateSynthesis::numEdges,
           "Number of tunable edges — the length of weights() / phases().")
      .def("residual", &EigenstateSynthesis::residual, py::arg("psi"),
           "Eigenvalue-agnostic residual r(psi) = ||(I - psi psi^dagger) L psi||^2 "
           "against the current edge weights/phases (psi normalized internally). "
           "r = 0 iff L psi || psi. Raises if len(psi) != order().")
      .def("rayleigh", &EigenstateSynthesis::rayleigh, py::arg("psi"),
           "Rayleigh quotient lambda = psi^dagger L psi / psi^dagger psi (real; L "
           "Hermitian) — the realized eigenvalue when r = 0. Raises if "
           "len(psi) != order().")
      .def("apply", &EigenstateSynthesis::apply, py::arg("psi"),
           "L psi against the current edge weights/phases (no normalization), for "
           "direct L psi || psi cross-checks. Raises if len(psi) != order().")
      .def("weights", &EigenstateSynthesis::weights,
           "Edge magnitudes {w_ij} (squaredLength) in the stable edge order.")
      .def("phases", &EigenstateSynthesis::phases,
           "Edge phases {theta_ij} (radians) in the stable edge order.")
      .def("setWeights", &EigenstateSynthesis::setWeights, py::arg("w"),
           "Write the edge magnitudes in place. Raises if len(w) != numEdges().")
      .def("setPhases", &EigenstateSynthesis::setPhases, py::arg("theta"),
           "Write the edge phases in place. Raises if len(theta) != numEdges().");

  // ----- §4b cone-and-retry synthesis loop → geo(ψ) (#134) -----
  py::class_<GeometrySynthesizer> gs(m, "GeometrySynthesizer",
      R"doc(§4b cone-and-retry geometry synthesis loop → geo(ψ).

Given a target qubit (c0, c1), finds the simplest simplicial complex whose k=0
Hodge Laplacian L = D - A has ψ = (c0, c1, 0, ..., 0) as an eigenvector — the
two smallest-id (logical) vertices carry the amplitudes, the rest are
zero-amplitude auxiliaries — and returns that minimal complex, its Hermitian
edge weights/phases, and the realized eigenvalue λ (the geometric image
geo(ψ)).

Built on EigenstateSynthesis (#133, the fixed-complex residual + Rayleigh +
parameter core; reused unmodified — a fresh one per optimize pass) and the
pre-geometric vertex insertion of the Pachner family (#112). The loop: run the
non-convex multi-restart optimizer (a bounded Levenberg–Marquardt least-squares
solver on the residual vector Lψ - λψ, with random restarts); if no restart
drives r = ||(I - ψψ†)Lψ||² below ε, cone in one vertex (join a fresh apex to
the current top simplex, Kₙ → Kₙ₊₁, supplying §4b.2's auxiliary freedom while
preserving the homotopy type of these contractible complexes) and re-optimize;
accept the first complex with r < ε. Its (|V|, |E|) is the state's
combinatorial complexity.

A general-amplitude qubit (|c0| ≠ |c1|) cannot be a two-vertex eigenvector
(residual floor w_min²(|c0|²-|c1|²)² > 0, #133); seeded on a single edge it is
synthesized only after coning in one auxiliary vertex (the minimal complex).)doc");

  py::class_<GeometrySynthesizer::Geo>(gs, "Geo",
      "geo(ψ): the accepted complex's size (|V|, |E|) = combinatorial "
      "complexity, its realized edge weights/phases, the realized eigenvalue λ, "
      "and whether the loop converged (r < ε).")
      .def_readonly("converged", &GeometrySynthesizer::Geo::converged,
                    "True iff the loop reached r < ε within the cone budget.")
      .def_readonly("residual", &GeometrySynthesizer::Geo::residual,
                    "Best residual r = ||(I - ψψ†)Lψ||² on the accepted complex.")
      .def_readonly("eigenvalue", &GeometrySynthesizer::Geo::eigenvalue,
                    "Realized eigenvalue λ = ψ†Lψ (Rayleigh quotient).")
      .def_readonly("num_vertices", &GeometrySynthesizer::Geo::numVertices,
                    "|V| of the accepted complex (with num_edges, the "
                    "combinatorial complexity).")
      .def_readonly("num_edges", &GeometrySynthesizer::Geo::numEdges,
                    "|E| of the accepted complex.")
      .def_readonly("cones_applied", &GeometrySynthesizer::Geo::conesApplied,
                    "Number of auxiliary vertices coned in to reach acceptance.")
      .def_readonly("weights", &GeometrySynthesizer::Geo::weights,
                    "The accepted complex's edge magnitudes {w_ij} (EdgeList order).")
      .def_readonly("phases", &GeometrySynthesizer::Geo::phases,
                    "The accepted complex's edge phases {θ_ij} (EdgeList order).");

  gs.def(py::init<std::shared_ptr<Spacetime>>(), py::arg("seed"),
          "Build the loop over a seed complex (§4b.4 seeds on a 4-simplex Δ⁴; a "
          "single edge is the minimal seed exhibiting the §4b.2 two-vertex "
          "floor). The two smallest-id vertices become the logical pair.")
      .def("synthesize", &GeometrySynthesizer::synthesize, py::arg("c0"),
           py::arg("c1"), py::arg("epsilon") = 1e-9, py::arg("restarts") = 64,
           py::arg("max_cones") = 5, py::arg("seed") = 0,
           "Run the cone-and-retry loop for the qubit (c0, c1) and return "
           "geo(ψ). Optimizes the current complex; if it cannot reach r < ε, "
           "cones in one vertex and retries (up to max_cones). Leaves the "
           "complex realized at the accepted optimum.")
      .def("optimize", &GeometrySynthesizer::optimize, py::arg("c0"),
           py::arg("c1"), py::arg("restarts") = 64, py::arg("seed") = 0,
           "Optimize the current complex only (no coning): multi-restart "
           "Levenberg–Marquardt minimizing r(ψ). Leaves the complex at the best "
           "parameters and returns that best residual — the §4b.2 floor probe.")
      .def("cone_in_vertex", &GeometrySynthesizer::coneInVertex,
           "Cone in one auxiliary vertex (join a fresh apex to the current top "
           "simplex, Kₙ → Kₙ₊₁). Returns False without growing if the simplex "
           "has reached the Fingerprint vertex capacity.")
      .def("num_vertices", &GeometrySynthesizer::numVertices,
           "|V| of the current complex.")
      .def("num_edges", &GeometrySynthesizer::numEdges,
           "|E| of the current complex.")
      .def("spacetime", &GeometrySynthesizer::spacetime,
           "The current (growing) complex.");

  // Exact integer / GF(2) / inertia primitives (also exposed for direct
  // testing). Matrices are passed flat row-major with explicit dims.
  py::class_<SmithNormalForm>(m, "SmithNormalForm")
      .def_readonly("rank", &SmithNormalForm::rank)
      .def_readonly("invariant_factors", &SmithNormalForm::invariantFactors);
  m.def("smith_normal_form", &smithNormalForm, py::arg("matrix"), py::arg("rows"),
        py::arg("cols"), "Smith Normal Form (rank + invariant factors) over Z.");
  m.def("integer_rank", &integerRank, py::arg("matrix"), py::arg("rows"), py::arg("cols"),
        "Rank over Q of an integer matrix.");
  m.def("gf2_rank", &gf2Rank, py::arg("matrix"), py::arg("rows"), py::arg("cols"),
        "Rank over GF(2) of a 0/1 matrix.");
  m.def("gf2_nullspace", &gf2Nullspace, py::arg("matrix"), py::arg("rows"),
        py::arg("cols"),
        "Basis of the GF(2) kernel of a 0/1 matrix, as a list of nullity "
        "length-cols vectors (each x with matrix·x == 0 mod 2; independent over "
        "GF(2); nullity == cols - gf2_rank). The cocycles Z1 = ker(d2^T mod 2).");
  m.def("gf2_span", &gf2Span, py::arg("basis"), py::arg("cols"),
        "All 2^k GF(2) combinations of a basis of k length-cols vectors (first "
        "is the zero vector). For a gf2_nullspace basis, the flat Z2 connections.");

  py::class_<Inertia>(m, "Inertia")
      .def_readonly("n_pos", &Inertia::nPos)
      .def_readonly("n_neg", &Inertia::nNeg)
      .def_readonly("n_zero", &Inertia::nZero)
      .def("signature", &Inertia::signature);
  m.def("symmetric_inertia", &symmetricInertia, py::arg("matrix"), py::arg("n"),
        py::arg("tol") = 1e-9,
        "Inertia (#pos,#neg,#zero eigenvalues) of a symmetric integer matrix.");

  // ----- Capability A (#65): characteristic numbers -----
  // Scalar invariants are Observables; families come from CharacteristicNumbers.
  py::class_<EulerCharacteristic, std::shared_ptr<EulerCharacteristic>>(
      m, "EulerCharacteristic",
      "Observable: Euler characteristic chi = sum_k (-1)^k |C_k|.")
      .def(py::init<>())
      .def("compute", &EulerCharacteristic::compute, py::arg("spacetime"));
  // Qualified: tessera::spacetime::Signature (metric signature) is also in
  // scope via the using-directives.
  py::class_<cobordism::Signature, std::shared_ptr<cobordism::Signature>>(
      m, "Signature",
      "Observable: signature b+ - b- of the H_2 intersection form (closed "
      "oriented 4-manifold; 0 if n != 4 or b2 == 0).")
      .def(py::init<>())
      .def("compute", &cobordism::Signature::compute, py::arg("spacetime"));

  py::class_<CharacteristicNumbers>(m, "CharacteristicNumbers",
      "Topological invariants of a closed PL n-manifold: Euler characteristic, "
      "signature (4-manifolds), Stiefel-Whitney numbers (pending), and "
      "Pontryagin numbers (4-manifolds: p1 = 3*signature).")
      .def_readonly("euler", &CharacteristicNumbers::euler,
                    "Euler characteristic (alternating count of cells).")
      .def_readonly("signature", &CharacteristicNumbers::signature,
                    "Signature of the intersection form; None unless an "
                    "orientable 4-manifold.")
      .def_readonly("stiefel_whitney_numbers",
                    &CharacteristicNumbers::stiefelWhitneyNumbers,
                    "Mod-2 Stiefel-Whitney numbers, keyed by monomial (pending; "
                    "currently empty).")
      .def_readonly("pontryagin_numbers",
                    &CharacteristicNumbers::pontryaginNumbers,
                    "Pontryagin numbers (4-manifolds: {'p1': 3*signature}).")
      .def_static("of", &CharacteristicNumbers::of, py::arg("spacetime"),
                  py::arg("oriented") = true,
                  "Compute the characteristic numbers of the given manifold.");

  // ----- Capability C (#66): cobordism verification (boundary structure) -----
  py::enum_<CobordismCheck>(m, "CobordismCheck")
      .value("Ok", CobordismCheck::Ok)
      .value("BoundaryChainNotClosed", CobordismCheck::BoundaryChainNotClosed)
      .value("WrongNumberOfBoundaryComponents",
             CobordismCheck::WrongNumberOfBoundaryComponents)
      .value("BoundaryNotIsomorphic", CobordismCheck::BoundaryNotIsomorphic);

  py::class_<CobordismResult>(m, "CobordismResult",
      "Result of verifying a cobordism: ok flag, machine-readable code, and a "
      "human-readable detail string.")
      .def_readonly("ok", &CobordismResult::ok)
      .def_readonly("code", &CobordismResult::code)
      .def_readonly("detail", &CobordismResult::detail);

  py::class_<Cobordism>(m, "Cobordism",
      "Cobordism verification (static-only): does a triangulation W have "
      "boundary equal to M1 disjoint-union M2? Checks the boundary structure "
      "and that the boundary is itself closed.")
      .def_static("boundaryFaces", &Cobordism::boundaryFaces, py::arg("W"),
                  "Codimension-one faces of W belonging to exactly one top "
                  "simplex (the boundary), as sorted vertex-id tuples.")
      .def_static("connectedComponents", &Cobordism::connectedComponents,
                  py::arg("simplices"),
                  "Split same-dimensional simplices into facet-connected pieces.")
      .def_static("areIsomorphic", &Cobordism::areIsomorphic, py::arg("a"),
                  py::arg("b"),
                  "Whether two triangulations (lists of top-simplex vertex "
                  "tuples) are isomorphic under a vertex relabeling.")
      .def_static("verify", &Cobordism::verify, py::arg("W"), py::arg("M1"),
                  py::arg("M2"),
                  "Verify W is a cobordism from M1 to M2 (boundary structure).")
      .def_static("glue", &Cobordism::glue, py::arg("W1"), py::arg("W2"),
                  "Glue two cobordisms along a shared boundary surface Sigma_C "
                  "(the first isomorphic boundary-component pair): identify its "
                  "two copies by the order-preserving simplicial isomorphism, "
                  "merge the complexes into one, and return the composite "
                  "W2 cup_{Sigma_C} W1 as a new Spacetime. Its boundary is the "
                  "remaining components (Sigma_A from W1, Sigma_B from W2). "
                  "Raises if the inputs are empty, differ in top dimension, or "
                  "share no isomorphic boundary surface.")
      .def_static("selfGlue", &Cobordism::selfGlue, py::arg("W"),
                  "Close a cobordism by gluing its two boundary components to "
                  "each other (the mapping torus / categorical trace): they must "
                  "be isomorphic, and the collar must be thick enough that no "
                  "top simplex touches both (>= 3 layers in the glued "
                  "direction). Returns the closed manifold as a new Spacetime. "
                  "Raises unless dW has exactly two isomorphic components, or if "
                  "the identification collapses a top simplex.");

  // ----- Capability T3 (#108): Dijkgraaf-Witten Z_2 state sum -----
  py::enum_<Cocycle>(m, "Cocycle",
      "The two normalized classes of Z^3(Z_2; U(1)) used as the "
      "Dijkgraaf-Witten weight: Trivial (omega == 1) and Sign "
      "(omega(a,b,c) = (-1)^{abc}, the generator of H^3(Z_2; U(1)) = Z_2).")
      .value("Trivial", Cocycle::Trivial)
      .value("Sign", Cocycle::Sign);

  py::class_<DijkgraafWitten>(m, "DijkgraafWitten",
      R"doc(Dijkgraaf-Witten Z_2 state sum of a closed oriented 3-manifold.

Z(W) = (1/2^|V|) sum_{flat g} prod_t omega(g_01, g_12, g_23)^{eps_t}, summed over
the flat Z_2 connections g in C^1 (the GF(2) nullspace of the coboundary
d_1 = boundaryMatrix(2)^T), with each tetrahedron's orientation sign eps_t from
the fundamental class. For connected W the untwisted value is Z_Trivial(W) =
2^{b_1(W;Z_2) - 1}; the Sign cocycle twists it by (-1)^{<g cup g cup g, [W]>},
which differs from the trivial value exactly when the mod-2 cup cube is nonzero
on W (e.g. RP^3), and agrees with it when the cube vanishes (e.g. T^3, S^2xS^1).
Enumerates the whole flat space (gauge-redundant), so it is intended for small
triangulations; a flat space too large to materialize is refused.)doc")
      .def(py::init<std::shared_ptr<Spacetime>, Cocycle>(), py::arg("W"),
           py::arg("cocycle"),
           "Build the state sum over a closed oriented 3-manifold W with the "
           "chosen cocycle.")
      .def("partitionFunction", &DijkgraafWitten::partitionFunction,
           "The partition function Z(W) as a complex number. Raises if W is not "
           "a closed oriented 3-manifold or the flat space is too large.")
      .def("boundaryVector", &DijkgraafWitten::boundaryVector,
           "The element of Z(dW) for a 3-manifold W with boundary: the amplitude "
           "for every joint boundary flat-connection class, flattened row-major "
           "over the boundary components (each component a closed surface Sigma_i "
           "with Hilbert dimension 2^{b1(Sigma_i)}). Holds g|dW fixed and sums "
           "the same prod omega(g01,g12,g23) over the interior gauge classes "
           "[g] in H^1(W;Z_2). Raises if W is null, not 3D, or closed.")
      .def("boundaryDimensions", &DijkgraafWitten::boundaryDimensions,
           "Per-boundary-component Hilbert-space dimensions 2^{b1(Sigma_i)}, in "
           "the same deterministic component order as boundaryVector()/map().")
      .def("map", &DijkgraafWitten::map,
           "The boundary state sum as a linear map Z(Sigma_B) -> Z(Sigma_A) when "
           "dW has exactly two components: a dense matrix (list of rows), "
           "rows = 2^{b1(Sigma_A)} (component 0), cols = 2^{b1(Sigma_B)} "
           "(component 1), in the flat-connection-class basis. For the trivial "
           "cobordism Sigma x [0,T] this is the identity id_{Z(Sigma)}. Raises "
           "unless dW has exactly two connected components.")
      .def("amplitude", &DijkgraafWitten::amplitude, py::arg("psiA"),
           py::arg("psiB"),
           "The transition amplitude <psiA| Z(W) |psiB> = sum_ab conj(psiA[a]) "
           "Z(W)_ab psiB[b] for boundary states in the flat-connection-class "
           "basis (|psiA| = 2^{b1(Sigma_A)}, |psiB| = 2^{b1(Sigma_B)}); prepared "
           "from the harmonic 1-forms ker L_1(Sigma). For the cylinder Z(W)=id, "
           "so this is the inner product <psiA|psiB>.")
      .def_static("isCocycle", &DijkgraafWitten::isCocycle, py::arg("cocycle"),
                  "Whether omega satisfies the normalized 3-cocycle (pentagon) "
                  "identity over Z_2 (brute-forced over all 16 tuples of "
                  "Z_2^4). True for both Trivial and Sign.");
}
