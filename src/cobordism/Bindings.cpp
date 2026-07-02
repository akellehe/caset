// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Pybind11 bindings for the cobordism subsystem. Lives outside tessera_core
// (which is pybind-free) so the static library can be reused without pulling
// in the Python dependency. This translation unit is always added to
// _tessera's sources (see CMakeLists.txt, TESSERA_PYBIND_SOURCES).

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cobordism/ChainComplex.h"
#include "cobordism/Characteristic.h"
#include "cobordism/Cochain.h"
#include "cobordism/CombinatorialDimension.h"
#include "cobordism/CobordismDAG.h"
#include "cobordism/EigenstateSynthesis.h"
#include "cobordism/MultiCobordism.h"
#include "cobordism/Proton.h"
#include "cobordism/ProtonIngredients.h"
#include "cobordism/HodgeLaplacian.h"
#include "cobordism/IntegerLinalg.h"
#include "cobordism/SurgicalCone.h"
#include "cobordism/Spectrum.h"
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
      .def_static(
          "dualComplexIsValid", &ChainComplex::dualComplexIsValid,
          py::arg("top_cells"), py::arg("dim"),
          py::arg("facet_cells") = std::vector<std::vector<std::uint64_t>>{},
          "(ok, reason): is the dual block decomposition of this pure "
          "n-complex a valid cell complex -- equivalently, is the primal a "
          "combinatorial manifold with boundary? Facet coface counts in "
          "{1,2}; no dangling facets against the optional (n-1)-cell "
          "universe; ridge links single paths/cycles; at n=3, vertex links "
          "2-spheres or disks. Pure combinatorics on sorted vertex-id "
          "tuples; rigorous for n <= 3. Accept topology moves only while "
          "this holds: validity in the DUAL space, not merely scoreability "
          "on the primal lattice.")
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
      .def_static(
          "endSignCovector", &ChainComplex::endSignCovector,
          py::arg("surface_cells"), py::arg("holes"),
          "The end sign covector sigma in {+/-1}^len(holes): the induced-"
          "orientation charge pattern of an end surface, from its fundamental "
          "chain. surface_cells are the end's top cells, holes the removed "
          "cells whose boundary cycles carry the periods; the union is "
          "oriented by sign propagation (component roots = lex-smallest "
          "cells, +1) and sigma_k is the orientation coefficient of "
          "holes[k], so every closed form's signed periods obey "
          "sum_k sigma_k p_k = 0 end by end. Deterministic -- a property of "
          "the end surface, not of any fill or spectrum -- and equivariant "
          "under order-preserving relabelings (e.g. a layer shift). Raises "
          "on mixed-dimension cells, a facet with > 2 cofaces, or a "
          "non-orientable surface.")
      .def_static(
          "orientationCovector", &ChainComplex::orientationCovector,
          py::arg("top_cells"),
          "The induced-orientation covector eps in {+/-1}^len(top_cells): the "
          "per-cell sign from orienting a whole top-cell complex by facet-"
          "sharing propagation (component roots = lex-smallest cells, +1; "
          "across an interior facet the two induced signs cancel). The result "
          "aligns to the sorted-unique (canonical C_d) order of the cells. "
          "Unlike fundamentalClass() it does NOT require closedness (boundary "
          "facets impose nothing), so it reads the orientation of an open "
          "refinement region (a stellar cone star, a CDT slab). Determined "
          "combinatorially, independent of geometry, vertex labels, and input "
          "order. Raises on mixed-dimension cells, a facet with > 2 cofaces, "
          "or a non-orientable propagation contradiction.")
      .def("intersectionForm", &ChainComplex::intersectionForm,
           "Symmetric intersection form on free H^2 (flat b2 x b2), for a closed "
           "oriented 4-manifold; empty if n != 4 or b2 == 0.")
      .def("signature", &ChainComplex::signature,
           "Signature b+ - b- of the intersection form (0 if n != 4 or b2 == 0).")
      .def("stiefelWhitneyNumbers", &ChainComplex::stiefelWhitneyNumbers,
           "Mod-2 Stiefel-Whitney numbers <w_{i1}..w_{ir}, [K]> keyed by "
           "monomial (e.g. 'w4', 'w2^2'); empty for the empty complex. Raises "
           "if a class needs a deferred higher Steenrod cup-i product (#65).");

  // ----- Eigen-backed value objects for the Hodge spectrum (#183) -----
  py::class_<Cochain>(m, "Cochain",
      R"doc(A k-cochain: complex amplitudes over a k-simplex ordering.

An Eigen-backed vector of complex amplitudes together with the degree k and the
k-simplex ordering it indexes (the same HodgeLaplacian / ChainComplex column
order), so its indices are meaningful. simplices()[i] is the sorted vertex-id
tuple of the cell carrying coeffs()[i]; at k=0 each tuple is a single vertex id
(the sorted-id vertex order). Eigen-backed and iTensor-free. The inner product is
Hermitian, np.vdot convention: <a, b> = sum conj(a_i) b_i.)doc")
      .def(py::init<int, std::vector<std::vector<std::uint64_t>>,
                    Eigen::VectorXcd>(),
           py::arg("degree"), py::arg("simplices"), py::arg("coeffs"),
           "A degree-k cochain over simplices (sorted vertex-id tuples, in the "
           "indexing order) carrying coeffs (a 1-D complex array). Raises if "
           "len(simplices) != len(coeffs).")
      .def("degree", &Cochain::degree, "The cochain degree k.")
      .def("size", &Cochain::size,
           "Number of k-cells (= len(coeffs()) = len(simplices())).")
      .def("__len__", &Cochain::size)
      .def("coeffs", &Cochain::coeffs,
           "The complex amplitudes as a 1-D numpy.ndarray (Eigen-backed).")
      .def("simplices", &Cochain::simplices,
           "The k-simplex ordering: simplices()[i] is the sorted vertex-id tuple "
           "of the cell carrying coeffs()[i].")
      .def("amplitude", &Cochain::amplitude, py::arg("index"),
           "Amplitude on the index-th k-cell. Raises IndexError if out of range.")
      .def("__getitem__", &Cochain::amplitude)
      .def("amplitudeFor", &Cochain::amplitudeFor, py::arg("simplex"),
           "Amplitude on the k-cell identified by its sorted vertex-id tuple "
           "(e.g. (vertexId,) at k=0). Raises IndexError if absent.")
      .def("innerProduct", &Cochain::innerProduct, py::arg("other"),
           "The Hermitian inner product <self, other> = sum conj(self_i) other_i "
           "(= np.vdot). Raises if the degrees or orderings differ.")
      .def("norm", &Cochain::norm, "The Euclidean norm sqrt(sum |c_i|^2).")
      .def("normalized", &Cochain::normalized,
           "A copy scaled to unit norm (the cochain itself if its norm is ~0).");

  py::class_<Spectrum>(m, "Spectrum",
      R"doc(The eigendecomposition of a Hodge Laplacian L_k as a value object.

Eigenvalues paired with their eigenvectors-as-Cochains, in matching order
(eigenvalues()[i] is the eigenvalue of eigenvectors()[i]). Eigenvalues are stored
complex to cover both regimes uniformly: in the Hermitian/metric case
(isHermitian() == True) they are real (imag 0) and ascending; in the Lorentzian
(signed-weight d'Alembertian) case they may be negative or complex-conjugate
pairs, sorted by (Re, Im). harmonics(tol) is the kernel subset |lambda| < tol =
ker L_k as Cochains. Supports len() and indexing (spectrum[i] is the i-th
eigenvector Cochain).)doc")
      .def("eigenvalues", &Spectrum::eigenvalues,
           "The eigenvalues as a 1-D complex numpy.ndarray (ascending real, "
           "imag 0, in the Hermitian regime; sorted by (Re, Im) in the Lorentzian "
           "one).")
      .def("eigenvectors", &Spectrum::eigenvectors,
           "The eigenvectors as a list of Cochains, one per eigenvalue.")
      .def("harmonics", &Spectrum::harmonics, py::arg("tol") = 1e-9,
           "The harmonic subset: eigenvectors with |lambda| < tol (a basis for "
           "ker L_k), as a list of Cochains.")
      .def("size", &Spectrum::size, "The number of modes.")
      .def("__len__", &Spectrum::size)
      .def("isHermitian", &Spectrum::isHermitian,
           "Whether the eigenvalues are guaranteed real and ascending (the "
           "metric/self-adjoint regime) vs. the indefinite Lorentzian one.")
      .def("eigenvalue", &Spectrum::eigenvalue, py::arg("i"),
           "The i-th eigenvalue. Raises IndexError if out of range.")
      .def("__getitem__",
           [](const Spectrum &s, std::size_t i) -> const Cochain & { return s[i]; },
           py::return_value_policy::reference_internal,
           "The i-th eigenvector Cochain. Raises IndexError if out of range.");

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
      .def("laplacianGradient", &HodgeLaplacian::laplacianGradient, py::arg("k"),
           py::arg("edgeA"), py::arg("edgeB"),
           "Exact analytic dL_k^sym/dl^2_e of the symmetric metric Hodge Laplacian "
           "(k>=1) w.r.t. one edge's squared length, flat |C_k|x|C_k| row-major. Only "
           "the weights W_j=|vol| depend on l^2; built via dB_k = diag(a_{k-1})B_k + "
           "B_k diag(b_k), a_j=dW_j/(2W_j), dW_j = Simplex.volumeGradient. Empty for "
           "k<1 or an absent edge.")
      .def("isHermitian", &HodgeLaplacian::isHermitian, py::arg("tol") = 1e-12,
           "True iff ||L - L^dagger|| <= tol (Frobenius) for the k=0 Laplacian.")
      .def("unitarityResidual", &HodgeLaplacian::unitarityResidual,
           py::arg("t") = 1.0,
           "Residual ||U U^dagger - I|| of U = e^{-iLt} formed from the "
           "eigendecomposition (~0 for the Hermitian L).")
      .def("spectrum", &HodgeLaplacian::spectrum, py::arg("k") = 0,
           py::arg("metric") = true,
           "The eigendecomposition of L_k as a Spectrum (real ascending "
           "eigenvalues + eigenvectors as Cochains; isHermitian()==True). metric "
           "selects volume vs. unit weights for k>=1 (ignored at k=0). Raises for "
           "k<0; empty above the top dimension.")
      .def("eigenvalues", &HodgeLaplacian::eigenvalues, py::arg("k") = 0,
           py::arg("metric") = true,
           "Eigenvalues of L_k (real, ascending), a flat view consistent with "
           "spectrum(k, metric). metric selects volume vs. unit weights for k>=1 "
           "(ignored at k=0). Raises for k<0; empty above the top dimension.")
      .def("eigenvectors", &HodgeLaplacian::eigenvectors, py::arg("k") = 0,
           py::arg("metric") = true,
           "Eigenvectors of L_k as a flat row-major M*M complex array (column j "
           "is the eigenvector for the j-th ascending eigenvalue), a flat view "
           "consistent with spectrum(k, metric).eigenvectors(). metric selects "
           "volume vs. unit weights for k>=1. Raises for k<0; empty above the top "
           "dimension.")
      .def("harmonics", &HodgeLaplacian::harmonics, py::arg("k") = 0,
           py::arg("tol") = 1e-9, py::arg("metric") = true,
           "Harmonic representatives (eigenvectors with |lambda| < tol) as a list "
           "of Cochains spanning ker L_k ~= H_k (the count is b_k). metric selects "
           "volume vs. unit weights for k>=1. Raises for k<0; empty above the top "
           "dimension.")
      .def("harmonicMatrix", &HodgeLaplacian::harmonicMatrix, py::arg("k") = 0,
           py::arg("tol") = 1e-9, py::arg("metric") = true,
           "The harmonic amplitude matrix: the harmonics(k, tol, metric) "
           "representatives stacked as the ROWS of a flat row-major "
           "(dim ker L_k) x M complex array (M = |V| at k=0, else |C_k|), "
           "columns in the canonical cell order (cellSimplices / "
           "kSimplexVertices). Entry [r*M + c] equals "
           "harmonics(k)[r].amplitude(c) exactly -- one call instead of one "
           "amplitudeFor round-trip per cell per harmonic. Raises for k<0; "
           "empty when the kernel is empty or k is above the top dimension.")
      // ----- Lorentzian (signed-weight) d'Alembertian (#105, spec §5.6) -----
      .def("lorentzianSpectrum", &HodgeLaplacian::lorentzianSpectrum,
           py::arg("k"), py::arg("metric") = true,
           "The eigendecomposition of the signed-weight d'Alembertian L_k as a "
           "Spectrum (isHermitian()==False): complex eigenvalues sorted by "
           "(Re, Im) + eigenvectors as Cochains. metric=False falls back to unit "
           "weights (the real combinatorial spectrum). Raises for k<0; empty "
           "above the top dimension.")
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
           "(eigenvectors with |lambda| < tol) as a list of Cochains. The count "
           "is b_k on an all-spacelike complex; with timelike cells it can differ "
           "(pseudo-Hodge decomposition). Matching W-norms: lorentzianNullNorms.")
      .def("lorentzianNullNorms", &HodgeLaplacian::lorentzianNullNorms,
           py::arg("k"), py::arg("tol") = 1e-9, py::arg("metric") = true,
           "Indefinite W-norms <h,h>_W = sum_i W_{k,i} |h_i|^2 of the near-kernel "
           "representatives, one per column of lorentzianHarmonics (same order). "
           "A value ~0 flags a NULL (lightlike) harmonic; all positive on an "
           "all-spacelike complex.");

  // ----- §4b eigenstate synthesis: residual + parameter access (#133) -----
  auto eigenstateSynthesis = py::class_<EigenstateSynthesis>(m, "EigenstateSynthesis",
      R"doc(§4b inverse eigenvector problem on a fixed complex, degree-k.

Scores how close the complex's current Hermitian edge weights make a target
state psi to being an eigenvector of the degree-k Hodge Laplacian L_k (via
HodgeLaplacian), and reads/writes those weights so a search can perturb them.
At k=0 L_0 = D - A is the graph Laplacian (magnitude convention) and psi is a
vertex vector (|V|, sorted-id order). At k>=1 L_k is the metric Hodge Laplacian
on k-forms (|C_k|, ChainComplex k-cell order); the tunable parameters stay the
edge squared-lengths, which feed the volume weights W_k of L_k via Simplex.volume
(phases enter only k=0). cellSimplices() gives each psi component's vertex tuple,
so a caller can pin the boundary k-cells to a target form (the #176 k=1
3-manifold boundary-harmonic synthesis). The non-convex, multi-restart search
itself (e.g.
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

Parameters: the per-edge SIGNED real squared lengths {w_ij} = Re l^2
(Edge.setSquaredLength; weights() reads Re, not a magnitude — #581) and U(1)
phases {theta_ij} (Edge.setPhase), in a stable edge order fixed at
construction (the weight-carrying edges: both endpoints present, no self-loops).

Fixed-boundary interior fill (§5.0): the tunable edges split into a boundary set
dW (edges on a codim-1 face in exactly one top cell — held fixed) and an interior
set (free). interiorWeights / interiorPhases + setInteriorWeights /
setInteriorPhases read/write only the interior edges, so a search drives r -> 0
for a target output eigenvector while dW stays byte-identical (boundaryEdges()
exposes that fixed set). growInterior() cones a fresh interior vertex via the
boundary-fixed pre-geometric Pachner add (#112), enriching the interior with dW
untouched; interiorVertexCount / numInteriorEdges report the interior complexity
reached. On a 1-complex there is no boundary — every edge is interior.)doc")
      .def(py::init<std::shared_ptr<Spacetime>, int>(), py::arg("spacetime"),
           py::arg("k") = 0,
           "Build the synthesizer over a fixed triangulation at Hodge degree k "
           "(default 0, the vertex graph Laplacian; k=1 is the metric Hodge "
           "Laplacian on edge 1-forms for the 3-manifold boundary-harmonic "
           "synthesis). Raises if k < 0.")
      .def("degree", &EigenstateSynthesis::degree,
           "The cochain degree k of L_k this synthesizer scores against.")
      .def("order", &EigenstateSynthesis::order,
           "Operator dimension N — the required length of any psi (|V| at k=0, "
           "else |C_k|, the number of k-cells).")
      .def("cellSimplices", &EigenstateSynthesis::cellSimplices,
           "The sorted vertex-id tuple of each psi component, in operator order "
           "(a single-vertex tuple per component at k=0, else the k-cell tuples "
           "in canonical ChainComplex column order) — used to pin the boundary "
           "k-cells to a target form and leave the interior free.")
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
           "The SIGNED real parts {Re l^2_ij} of the edge squared lengths, in "
           "the stable edge order — not magnitudes (a timelike edge reads "
           "negative), and any resident Im l^2 is not reported (#581).")
      .def("phases", &EigenstateSynthesis::phases,
           "Edge phases {theta_ij} (radians) in the stable edge order.")
      .def("setWeights", &EigenstateSynthesis::setWeights, py::arg("w"),
           "Write the edge squared lengths in place as REAL signed values "
           "(l^2 = w + 0i, zeroing any resident Im — the ordinary-Lorentzian "
           "convention, #581). Raises if len(w) != numEdges().")
      .def("setPhases", &EigenstateSynthesis::setPhases, py::arg("theta"),
           "Write the edge phases in place. Raises if len(theta) != numEdges().")
      // ----- Fixed-boundary interior fill (§5.0, #147) -----
      .def("numInteriorEdges", &EigenstateSynthesis::numInteriorEdges,
           "Number of interior tunable edges (not on dW) — the length of "
           "interiorWeights() / interiorPhases() and the free parameters a "
           "fixed-boundary search varies.")
      .def("numBoundaryEdges", &EigenstateSynthesis::numBoundaryEdges,
           "Number of boundary tunable edges (on dW, held fixed).")
      .def("interiorVertexCount", &EigenstateSynthesis::interiorVertexCount,
           "Number of interior vertices (on no boundary face) — the coned-in "
           "apexes; the interior complexity the synthesis grows / reports.")
      .def("interiorWeights", &EigenstateSynthesis::interiorWeights,
           "Interior edge SIGNED real squared lengths {Re l^2_ij} in "
           "interior-edge order (Re, not magnitudes — #581).")
      .def("interiorPhases", &EigenstateSynthesis::interiorPhases,
           "Interior edge phases {theta_ij} (radians) in interior-edge order.")
      .def("setInteriorWeights", &EigenstateSynthesis::setInteriorWeights,
           py::arg("w"),
           "Write the interior edge squared lengths in place as REAL signed "
           "values (l^2 = w + 0i, zeroing any resident Im — #581); the boundary "
           "edges are left untouched. Raises if len(w) != numInteriorEdges().")
      .def("setInteriorPhases", &EigenstateSynthesis::setInteriorPhases,
           py::arg("theta"),
           "Write the interior edge phases in place; the boundary edges are left "
           "untouched. Raises if len(theta) != numInteriorEdges().")
      .def("boundaryEdges", &EigenstateSynthesis::boundaryEdges,
           "The boundary tunable edges as sorted (min_id, max_id) endpoint "
           "tuples — the fixed dW edge set, for asserting it is untouched through "
           "an interior fill / growth sweep.")
      .def("interiorEdges", &EigenstateSynthesis::interiorEdges,
           "The interior tunable edges as sorted (min_id, max_id) endpoint tuples "
           "(the complement of boundaryEdges()).")
      .def("growInterior", &EigenstateSynthesis::growInterior, py::arg("seed"),
           "Cone a fresh interior vertex into a top cell via the boundary-fixed "
           "pre-geometric Pachner add (#112): a 1->(d+1) stellar subdivision that "
           "leaves dW exactly fixed while enriching the interior. Re-captures the "
           "vertex order and interior/boundary partition, so order() grows by one "
           "(extend psi on the new apex, appended last in sorted-id order) and "
           "numInteriorEdges() grows. Returns False if no top cell can be "
           "subdivided (e.g. a 1-complex), leaving the complex unchanged.")
      // ----- Free interior connectivity (general growth primitive, #200) -----
      .def("attachInteriorVertex", &EigenstateSynthesis::attachInteriorVertex,
           py::arg("incident_simplices"),
           "Add a fresh interior vertex with an arbitrary specified set of "
           "incident simplices — the cone-free generalization of growInterior. "
           "incident_simplices is a list of vertex-id lists; the new vertex + each "
           "such set forms one new simplex (its full 1-skeleton is materialized), "
           "so a singleton [u] wires the new vertex to u by an edge and the d "
           "facets of a top cell reproduce coning. The new vertex takes the "
           "largest id. Validates ONLY (a) a valid downward-closed complex and "
           "(b) the pinned boundary dW bit-exact; no manifold/topology constraint. "
           "Returns False, leaving the complex unchanged, on an invalid spec "
           "(missing/repeated vertex, empty) or any perturbation of dW.")
      .def("detachLastInteriorVertex",
           &EigenstateSynthesis::detachLastInteriorVertex,
           "Undo the most recent attachInteriorVertex (LIFO): remove its created "
           "simplices/edges and the interior vertex, restoring the complex bit-"
           "exactly, and re-capture. Returns False if there is no attach to undo. "
           "Lets a search try a candidate connectivity, score it, and roll back.")
      .def("vertexIds", &EigenstateSynthesis::vertexIds,
           "All vertex ids, sorted — the candidate pool a connectivity search "
           "wires a fresh interior vertex into.")
      .def("boundaryVertexIds", &EigenstateSynthesis::boundaryVertexIds,
           "The boundary (dW) vertex ids, sorted — the vertices on a codim-one "
           "face of exactly one top cell (a 'boundary-star' candidate).")
      .def("topCells", &EigenstateSynthesis::topCells,
           "The top cells as sorted vertex-id tuples (the d+1-vertex simplices); "
           "wiring the new vertex to one reproduces growInterior's 1-skeleton.")
      .def("dualComplexValid", &EigenstateSynthesis::dualComplexValid,
           "(ok, reason): ChainComplex.dualComplexIsValid for the CURRENT "
           "complex -- top cells from the surgery state, with the k-cell "
           "universe checked for dangling facets when k = n-1 (the register "
           "layers). Accept topology moves only while this stays true.")
      // ----- The carried register read-outs (#286) -----
      .def("cyclePeriods", &EigenstateSynthesis::cyclePeriods, py::arg("holes"),
           "The period matrix of the current harmonics over the boundary "
           "cycles of the given (removed) cells: flat row-major "
           "(dim ker L_k) x len(holes), complex. Entry [r*m + q] sums "
           "harmonic r over hole q's facets with the boundary operator's "
           "induced-orientation signs (facet j of the sorted hole drops v_j, "
           "sign (-1)^j) -- degree-general: circles at k=1, spheres at k=2. "
           "Harmonics are read fresh from the live complex, rows ascending "
           "by eigenvalue. Raises if a hole is not a (k+2)-vertex tuple "
           "whose facets are all current k-cells.")
      .def("carriedRepresentative", &EigenstateSynthesis::carriedRepresentative,
           py::arg("holes"), py::arg("target_periods"),
           "The carried representative psi that residualForPeriods scores, as a "
           "cochain in its own right (it builds this internally but does not "
           "return it). Least-squares-projects target_periods onto the carried "
           "period rows (minimum-norm, as numpy.linalg.lstsq), forms the harmonic "
           "combination psi = sum_r c_r h_r, and attaches each hole's uncarried "
           "remainder (the minimal leak) to the hole's first walk-order facet so "
           "psi's periods are exactly target_periods. A full order()-length cell "
           "vector; residual(psi) is residualForPeriods. Raises on a hole/target "
           "length mismatch or a malformed hole.")
      .def("residualForPeriods", &EigenstateSynthesis::residualForPeriods,
           py::arg("holes"), py::arg("target_periods"),
           "The verdict primitive in one call: the genuine residual of the "
           "carried representative of target_periods over the holes' cycles. "
           "Least-squares-projects the targets onto the carried period rows "
           "(minimum-norm, as numpy.linalg.lstsq), forms the harmonic "
           "combination, attaches each hole's uncarried remainder (the "
           "minimal leak) to the hole's first walk-order facet (the (a,b) "
           "edge of a circle at k=1, the drop-v0 facet otherwise; boundary "
           "sign +1), and returns residual(psi): -> 0 iff the targets lie "
           "in the carried register, floored otherwise. Raises on a "
           "hole/target length mismatch or a malformed hole.")
      .def("residualForPeriodsGradient",
           &EigenstateSynthesis::residualForPeriodsGradient,
           py::arg("holes"), py::arg("target_periods"),
           "Arbitrary-degree exact analytic gradient d r_U / d l^2 of "
           "residualForPeriods w.r.t. each edge's squared length, in ChainComplex "
           "1-cell (edge) order. M = L_k, the per-edge dL_k/dl^2 = HodgeLaplacian."
           "laplacianGradient (built on Simplex.volumeGradient), through eigenvector-"
           "perturbation theory; period covector + leak from each removed-(k+1)-cell "
           "hole's facets. Reproduces the k=1 edge-loop core on triangle holes; "
           "certified by the exact Euler identity Σ l² ∂r_U = −r_U (FD does not "
           "converge). Raises on a hole/target length mismatch.")
      .def("residualForPeriodsGradientGpu",
           &EigenstateSynthesis::residualForPeriodsGradientGpu,
           py::arg("holes"), py::arg("target_periods"),
           "FP32 cuBLAS (SGEMM) GPU port of residualForPeriodsGradient (#348): "
           "the identical analytic gradient, with the dominant per-edge GEMMs run "
           "in single precision on the GPU and the eigensolve + cheap small-dim "
           "algebra + final reductions kept on the CPU in FP64. FP32 is the only "
           "approximation (~1e-5 vs FP64); residualForPeriodsGradient stays the "
           "default and the correctness oracle. Requires a TESSERA_CUDA build "
           "(raises otherwise), and raises on a hole/target length mismatch.")
      // ----- The hard period-pin r_psi (the realizability alternative, #377) -----
      .def("periodGapForPeriods", &EigenstateSynthesis::periodGapForPeriods,
           py::arg("holes"), py::arg("target_periods"),
           "The hard period-pin r_psi over the holes' cycles: r_psi = "
           "||P^T c - target||^2, where the columns of P^T are the live "
           "harmonics' periods over the holes and c is their least-squares fit "
           "-- the squared norm of the part of target_periods no pure harmonic "
           "can carry. Unlike residualForPeriods (r_U), the carried object stays "
           "a pure harmonic (NO leak). -> 0 iff the targets lie in the carried "
           "period span (the same realizable set as r_U), floored otherwise. "
           "Raises on a hole/target length mismatch or a malformed hole.")
      .def("periodGapForPeriodsGradient",
           &EigenstateSynthesis::periodGapForPeriodsGradient,
           py::arg("holes"), py::arg("target_periods"),
           "The exact analytic gradient d r_psi / d l^2 of periodGapForPeriods, "
           "in cellSimplices() (k=1 cell) order. By least-squares optimality "
           "(A^T r = 0, the envelope theorem) only the harmonic-subspace "
           "perturbation enters: d r_psi = 2 Re( r^H (Q dUn) c ) -- no leak, no "
           "dpsi chain. Raises on a hole/target length mismatch or a malformed "
           "hole.")
      // ----- The discovered operator: ker L1(W - dW) (#363) -----
      .def("bulkMinusBoundaryCells",
           &EigenstateSynthesis::bulkMinusBoundaryCells,
           "The interior 1-cells of W - dW (edges both of whose endpoints are "
           "interior vertices, on no dW face), as sorted (u,v) tuples in "
           "canonical ChainComplex C_1 order -- the column ordering of "
           "bulkMinusBoundaryHarmonicMatrix. Empty for a bare (un-grown) "
           "cobordism (all boundary, no interior bulk).")
      .def("bulkMinusBoundaryHarmonicMatrix",
           &EigenstateSynthesis::bulkMinusBoundaryHarmonicMatrix,
           py::arg("tol") = 1e-9,
           "ker L1(W - dW): the harmonic 1-forms of the combinatorial "
           "(unit-weight, signature-blind) Hodge Laplacian L1 of the bulk with "
           "the full dW subcomplex deleted (the subcomplex induced on the "
           "interior vertices). Restricts the integer boundary maps d1, d2 to "
           "the interior cells and eigendecomposes L1 = d1^T d1 + d2 d2^T, "
           "returning the |lambda| < tol eigenvectors stacked as the ROWS of a "
           "flat row-major (dim ker L1) x len(bulkMinusBoundaryCells()) complex "
           "array (ascending eigenvalue). The geometry the discovered operator "
           "is read from -- surgery must first grow the interior so this is "
           "nonzero. Read fresh from the live complex.")
      // ----- Surgery: the topology-changing interior remove move (#196) -----
      .def("interiorTopCells", &EigenstateSynthesis::interiorTopCells,
           "The interior top cells (all-interior vertices, on no dW face) as "
           "sorted vertex-id tuples — the surgery removal candidates. Removing one "
           "(removeInteriorCell) cannot touch dW, so it is the boundary-fixed "
           "TOPOLOGY-CHANGING move that can open a hole/handle and MOVE b_k, unlike "
           "growInterior's subdivision and the additive attach.")
      .def("removeInteriorCell", &EigenstateSynthesis::removeInteriorCell,
           py::arg("cell"),
           "Surgery (#196): remove the interior top cell `cell` (a tuple from "
           "interiorTopCells()) and any edges it leaves orphaned, keeping a valid "
           "downward-closed complex. Topology-CHANGING: b_k moves (a filled disk "
           "b_1=0 becomes an annulus b_1=1). dW is held bit-exact — the cell has no "
           "boundary vertex, and the move is rejected if a dW edge would vanish; "
           "the EXPOSED interior boundary (the opened hole) is allowed. Records the "
           "removal for restoreLastRemoval. Returns False, complex unchanged, if "
           "`cell` is not an interior top cell or the removal would touch dW.")
      .def("restoreLastRemoval", &EigenstateSynthesis::restoreLastRemoval,
           "Undo the most recent removeInteriorCell (LIFO): re-create the removed "
           "top cell and the edges it orphaned, restoring their weights/phases bit-"
           "exactly, and re-capture. Returns False if there is no removal to undo. "
           "Lets a surgery search try a removal, score it, and roll back.")
      // ----- Gated moves: the checked cut and the composed stellar move -----
      .def("removeInteriorCellChecked",
           &EigenstateSynthesis::removeInteriorCellChecked, py::arg("cell"),
           "(ok, reason): the gated surgery cut — removeInteriorCell(cell), then "
           "the dual-validity gate (dualComplexValid), rolled back via "
           "restoreLastRemoval when the cut violates the dual. (True, 'ok') means "
           "the cut is applied and the dual complex stayed valid; (False, reason) "
           "means the complex is unchanged — the cell was not a removable interior "
           "top cell, or the reason names the dual violation. The gate is rigorous "
           "for n <= 3; dimension-4 callers use explicit constructions, not gated "
           "moves.")
      .def("stellarSubdivideInterior",
           &EigenstateSynthesis::stellarSubdivideInterior, py::arg("cell"),
           "(ok, reason): the composed gated stellar move — attach a fresh "
           "interior vertex onto `cell`'s facet fan (attachInteriorVertex with "
           "the d+1 codim-one facets; dW untouched), remove the subdivided parent "
           "(removeInteriorCell; its facets keep two cofaces, so dW stays "
           "bit-exact), gate on dualComplexValid, and roll back BOTH in LIFO "
           "order (restoreLastRemoval, then detachLastInteriorVertex) on "
           "violation. Each accepted move adds exactly ONE interior vertex and "
           "preserves ker L_k (the fan is homotopic to the cell it replaces). On "
           "acceptance the bulk's edges are re-pinned uniform (squaredLength 1, "
           "phase 0) — the unit cochain metric the register/fill seeds are built "
           "with, held by construction rather than by the createSimplexTracked "
           "time-rule coincidence on all-same-time seeds.")
      // ----- Charge sector: the E/B split of F in Omega^2 (#417) -----
      .def("curvatureFromConnection",
           &EigenstateSynthesis::curvatureFromConnection, py::arg("A"),
           "The curvature 2-cochain F = dA from a U(1) connection 1-cochain A by "
           "discrete coboundary: on each sorted degree-2 cell (a,b,c), F = "
           "A(a,b) + A(b,c) - A(a,c), the induced-orientation signed edge sum the "
           "period read-out uses (cyclePeriods). A is a degree-1 cochain in the "
           "canonical ChainComplex 1-cell order (length = the number of edges, "
           "i.e. EigenstateSynthesis(st, 1).order()); this instance is degree 2 "
           "and returns an order()-length 2-cochain. Gauge-invariant (d.d = 0): a "
           "pure gauge A -> A + d chi leaves F unchanged. Raises if degree() != 2, "
           "if len(A) is not the number of 1-cells, or if a 2-cell edge is "
           "missing.")
      .def("fieldStrengthSplit", &EigenstateSynthesis::fieldStrengthSplit,
           py::arg("F"),
           "The electric/magnetic split of a field-strength 2-cochain F by the "
           "causal type of each plaquette: electric = F on plaquettes carrying a "
           "timelike edge (one temporal leg, the discrete F_{0i}); magnetic = F "
           "on purely-spacelike plaquettes (F_{ij}). Returns a FieldStrengthSplit "
           "whose electric/magnetic are order()-length cochains (agreeing with F "
           "on their own support, zero elsewhere, so electric + magnetic == F) "
           "and whose electricCells/magneticCells are the disjoint, complete "
           "index lists into cellSimplices(). A plaquette is electric iff any of "
           "its three edges is Edge.isTimelike() on the live complex. Raises if "
           "degree() != 2, if len(F) != order(), or if a plaquette edge is "
           "missing.")
      .def("gaussLawCharge", &EigenstateSynthesis::gaussLawCharge, py::arg("F"),
           py::arg("enclosedVertices"), py::arg("electricOnly") = true,
           "The discrete Gauss-law charge Q = oint_S E (#411): the temporal-sector "
           "flux of a field-strength 2-cochain F through the closed surface S = dV "
           "bounding the worldtube V (the closed star of enclosedVertices, the quark "
           "windows). Sums F over S's plaquettes with their induced (-1)^j "
           "orientation (interior faces of V cancel), restricted to the ELECTRIC "
           "(timelike-leg, F_{0i}) plaquettes when electricOnly, else the full flux. "
           "For an exact F = d psi the full flux is <psi, d^2 V> = 0 to round-off -- "
           "the topological protection that makes Q a metric-robust gauged-U(1) "
           "holonomy (unlike a hand-weighted flavor covector). On an all-spacelike "
           "(Riemannian) complex no plaquette is electric, so the electric Q is "
           "exactly 0 (the neutral total of the reduced color-only sector). Raises "
           "if degree() != 2 or len(F) != order().");

  // The result of fieldStrengthSplit (#417): the E/B partition of F in Omega^2.
  py::class_<EigenstateSynthesis::FieldStrengthSplit>(
      eigenstateSynthesis, "FieldStrengthSplit",
      "The E/B split of a field-strength 2-cochain F by plaquette causal type "
      "(EigenstateSynthesis.fieldStrengthSplit): electric (timelike-leg "
      "plaquettes), magnetic (purely-spacelike plaquettes), and their disjoint "
      "index lists into cellSimplices(). electric + magnetic == F.")
      .def_readonly("electric",
                    &EigenstateSynthesis::FieldStrengthSplit::electric,
                    "F on plaquettes with a timelike leg (zero elsewhere); an "
                    "order()-length 2-cochain.")
      .def_readonly("magnetic",
                    &EigenstateSynthesis::FieldStrengthSplit::magnetic,
                    "F on purely-spacelike plaquettes (zero elsewhere); an "
                    "order()-length 2-cochain.")
      .def_readonly("electricCells",
                    &EigenstateSynthesis::FieldStrengthSplit::electricCells,
                    "Indices into cellSimplices() of the electric (timelike-leg) "
                    "plaquettes.")
      .def_readonly("magneticCells",
                    &EigenstateSynthesis::FieldStrengthSplit::magneticCells,
                    "Indices into cellSimplices() of the magnetic "
                    "(purely-spacelike) plaquettes.");

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


  // === MultiCobordism (#491): the C++ source-of-truth fully-emergent merge
  // optimizer — emergent topology at a user-defined degree k. ===
  py::class_<MultiCobordism::BoundaryBlock>(m, "MultiCobordismBlock",
      "An emergent boundary block of a MultiCobordism (an input or output): the "
      "vertex set whose own sub-complex carries the block, and its target period "
      "vector. Read the block's sub-complex with Spacetime.fromCells over the "
      "cells inside `vertices`, then its holes with MultiCobordism.emergent_holes.")
      .def_property_readonly(
          "vertices",
          [](const MultiCobordism::BoundaryBlock &block) {
            return std::vector<std::uint64_t>(block.vertices.begin(),
                                              block.vertices.end());
          })
      .def_property_readonly("target",
                             [](const MultiCobordism::BoundaryBlock &block) {
                               return block.target;
                             });
  auto multiCobordismClass =
      py::class_<MultiCobordism, std::shared_ptr<MultiCobordism>>(m, "MultiCobordism",
      "The fully-emergent MultiCobordism merge optimizer (#491): merge as a "
      "fully emergent optimization. From a bare host it grows the register by "
      "gated surgical moves under F = ||grad S||^2 + gamma*(r_U(output) + "
      "sum_i r_U(input_i)) at a USER-DEFINED degree k (degrees), reading holes "
      "dynamically off getBoundary. Two stages: run_stage1 (combinatorial), "
      "run_stage2 (geometric). An EMPTY output_targets list is supported (#555): "
      "nothing is pinned downstream, r_u sums only the input blocks, and the "
      "whole's final state emerges (read after the fact).")
      .def(py::init<std::shared_ptr<Spacetime>,
                    std::vector<std::vector<std::complex<double>>>,
                    std::vector<std::vector<std::complex<double>>>,
                    std::vector<int>, double, std::uint64_t, int>(),
           py::arg("host"), py::arg("input_targets"), py::arg("output_targets"),
           py::arg("degrees") = std::vector<int>{3}, py::arg("gamma") = 1.0,
           py::arg("seed") = 0, py::arg("precone") = 0)
      .def_static("betti", &MultiCobordism::betti, py::arg("st"))
      .def_static("emergent_holes", &MultiCobordism::emergentHoles,
                  py::arg("st"), py::arg("k"))
      .def_static("regge_action_gradient", &MultiCobordism::reggeActionGradient, py::arg("st"))
      .def_static("r_state", &MultiCobordism::residualOfTargetStateAgainstHarmonic,
                  py::arg("st"), py::arg("k"), py::arg("target"))
      .def("r_u", &MultiCobordism::rU, py::arg("st"))
      .def("objective", &MultiCobordism::objective)
      .def("set_input_residual_weight", &MultiCobordism::setInputResidualWeight,
           py::arg("weight"))
      .def("seed_inputs", &MultiCobordism::seedInputs, py::arg("seeds"))
      .def("seed_outputs", &MultiCobordism::seedOutputs, py::arg("seeds"))
      // Long pure-C++ compute: release the GIL for the duration so a background thread can
      // drive a pass (a single call, per the register-growth constraint) without blocking the
      // main thread -- e.g. multicobordism_animation.py --live keeps its GUI responsive.
      .def("run_stage1", &MultiCobordism::runStage1, py::arg("max_steps") = 200,
           py::arg("n_candidate_moves") = 12, py::arg("patience") = 8,
           py::arg("grow_boundaries") = false,
           py::call_guard<py::gil_scoped_release>())
      .def("run_stage2", &MultiCobordism::runStage2, py::arg("beta") = 1.0,
           py::arg("max_iters") = 200, py::arg("alpha0") = 0.05,
           py::arg("rel_tol") = 1e-9,
           py::call_guard<py::gil_scoped_release>(),
           "Stage 2 (geometric): relax every edge l^2 toward a stationary point of "
           "beta*||grad S||^2 + gamma*r_U (Wirtinger steepest descent, backtracking "
           "line search). Stops on the RELATIVE stationarity test -- no line-search "
           "step lowers F by more than rel_tol*max(|F|,1) -- or the max_iters budget "
           "cap. Read last_stage2_stationary for which one ended the run. Returns the "
           "F trace.")
      .def_property_readonly("st", &MultiCobordism::spacetime)
      .def_property_readonly("inputs", &MultiCobordism::inputs,
                             py::return_value_policy::reference_internal,
                             "The emergent input blocks (each a MultiCobordismBlock).")
      .def_property_readonly("outputs", &MultiCobordism::outputs,
                             py::return_value_policy::reference_internal,
                             "The emergent output blocks (each a MultiCobordismBlock).")
      .def_property_readonly("last_stage2_stationary",
                             &MultiCobordism::lastStage2Stationary,
                             "True iff the last run_stage2 stopped on the relative-"
                             "tolerance stationarity test (delta_rel < rel_tol); False "
                             "if it hit the max_iters budget cap.");
  py::enum_<MultiCobordism::BuildAction>(multiCobordismClass, "BuildAction",
      "One canonical solve action a search policy (Proton's build restart loop, a greedy "
      "driver, or the RL agent) composes, so the solve runs through the engine rather than "
      "being re-implemented by each consumer.")
      .value("GROW", MultiCobordism::BuildAction::Grow)
      .value("EVOLVE", MultiCobordism::BuildAction::Evolve)
      .value("RELAX", MultiCobordism::BuildAction::Relax)
      .value("CONE_OUT", MultiCobordism::BuildAction::ConeOut)
      .value("CONE_IN", MultiCobordism::BuildAction::ConeIn);
  py::enum_<MultiCobordism::HolePlacementStrategy>(multiCobordismClass,
      "HolePlacementStrategy",
      "Secondary ordering for the directed cone-out probe (both interior-first): "
      "ADJACENT_HOLES_LAST sends cells sharing vertices with existing holes to the back "
      "(separated register), ADJACENT_HOLES_FIRST to the front (clustered).")
      .value("ADJACENT_HOLES_FIRST", MultiCobordism::HolePlacementStrategy::AdjacentHolesFirst)
      .value("ADJACENT_HOLES_LAST", MultiCobordism::HolePlacementStrategy::AdjacentHolesLast);
  multiCobordismClass
      .def("build_step", &MultiCobordism::buildStep, py::arg("action"),
           py::arg("max_steps") = 30, py::arg("n_candidate_moves") = 8,
           py::arg("patience") = 15, py::arg("stage2_beta") = 1.0,
           py::arg("stage2_max_iters") = 10, py::arg("stage2_alpha0") = 0.05,
           py::arg("hole_placement_strategy") =
               MultiCobordism::HolePlacementStrategy::AdjacentHolesLast,
           // Composes run_stage1/run_stage2 internally (C++ -> C++, so no nested guard); release
           // the GIL here too so a background thread driving the build stays off the main thread.
           py::call_guard<py::gil_scoped_release>(),
           "Apply one BuildAction to this node in place (GROW/EVOLVE = run_stage1 with "
           "grow_boundaries true/false; RELAX = run_stage2; CONE_OUT/CONE_IN = the directed "
           "probes) -- the canonical solve step a policy (build, greedy, or RL) composes.")
      .def("directed_cone_out", &MultiCobordism::directedConeOut,
           py::arg("strategy") = MultiCobordism::HolePlacementStrategy::AdjacentHolesLast,
           py::arg("max_open") = 6,
           "Directed gated cone-out: deliberately open register holes, keeping the opener "
           "that most lowers this node's rU (which absorbs r_state). Returns #holes opened.")
      .def("directed_cone_in", &MultiCobordism::directedConeIn, py::arg("max_close") = 6,
           "Directed gated cone-in: select the register by capping the hole whose removal "
           "most lowers rU. Returns #holes capped.");

  // === CobordismDAG (#491): chain emergent merges, output -> input ===
  py::class_<CobordismDAG>(m, "CobordismDAG",
      "Chain emergent merges (MultiCobordism) into a DAG: the output of one "
      "cobordism is an input to the next (the proton_merge_sequence compose, "
      "generalized). add_node returns a node id; edges pipe upstream outputs into "
      "downstream input slots; run() executes in topological order, recording each "
      "node's output (its verified output_target) and realizability residual r_U.")
      .def(py::init<>())
      .def("add_node", &CobordismDAG::addNode, py::arg("host"),
           py::arg("literal_inputs"), py::arg("upstream"),
           py::arg("output_targets"), py::arg("degrees") = std::vector<int>{3},
           py::arg("gamma") = 1.0, py::arg("seed") = 0,
           "Add a node (one co-optimized MultiCobordism system): a bare host, "
           "literal input targets, `upstream` as (node_id, output_index) tuples "
           "whose outputs feed further inputs, and `output_targets` (one for a "
           "merge, two for a 2->2 recombination). Returns the node id.")
      .def("run", &CobordismDAG::run, py::arg("stage1_max_steps") = 30,
           py::arg("stage1_candidate_moves") = 8, py::arg("stage1_patience") = 8,
           py::arg("stage2_beta") = 1.0, py::arg("stage2_max_iters") = 40,
           "Run all nodes in topological order (raises on a cycle).")
      .def("output", &CobordismDAG::output, py::arg("node"),
           py::arg("output_index") = 0)
      .def("num_outputs", &CobordismDAG::numOutputs, py::arg("node"))
      .def("residual", &CobordismDAG::residual, py::arg("node"))
      .def("__len__", &CobordismDAG::size);

  // === Proton (#503): the canonical two-step MultiCobordism proton build ===
  auto protonClass = py::class_<Proton>(m, "Proton",
      R"doc(The canonical, footgun-free proton builder, composing MultiCobordism.

A proton is THREE quarks in a colorless bound state, so it is built in TWO steps
(a single merge would be physically invalid). omega = exp(2*pi*i/3).
  * Step A (recombination, one 2->2 node): two neutral q-qbar pairs {1,-1,0},
    {1,0,-1} -> a colored diquark {1,w} + antidiquark {1,w*w} (2-vectors).
  * Step B (formation, a separate 2->1 node): the diquark {1,w} + the third
    quark {w*w} -> the proton {1,w,w*w} (the 3-vector color singlet).
build() builds the closed-S^4 hosts internally and restarts across distinct
seeds until step B's proton block carries the singlet with >=3 color holes. The
accessors lazily trigger build() on first use, so `Proton().block()` just works.
Observable readers (charge/mass/radius/spin) read OFF block() in their own
tickets.)doc");
  protonClass
      .def(py::init<std::uint64_t, int, double, double, int, bool>(), py::arg("seed") = 0,
           py::arg("register_degree") = 3, py::arg("gamma") = 50.0,
           py::arg("input_weight") = 20.0, py::arg("precone") = 0,
           py::arg("should_use_directed_surgery") = false)
      .def_static("omega", &Proton::omega, "omega = exp(2*pi*i/3).")
      .def_static("singlet", &Proton::singlet,
                  "The proton color singlet {1, w, w*w}.")
      .def("build", &Proton::build, py::arg("max_restarts") = 16,
           py::arg("init_steps") = 180,
           py::arg("evolve_steps") = 60, py::arg("stage1_candidate_moves") = 8,
           py::arg("stage1_patience") = 15, py::arg("stage2_beta") = 1.0,
           py::arg("stage2_max_iters") = 10, py::arg("color_tolerance") = 0.5,
           py::arg("min_quark_holes") = 3,
           "Restart across seeds until the whole step-B cobordism carries the singlet "
           "with >= min_quark_holes holes. Each step runs an init pass (grow the "
           "boundary until it carries) then an evolution pass (boundary frozen).")
      .def("recombination_node", &Proton::recombinationNode, py::arg("seed"),
           "A fresh, seeded (not-yet-run) Step A node: two neutral q-qbar pairs -> a "
           "diquark {1,w} + antidiquark {1,w*w}, on a single Delta^4 seed. Drive it with "
           "run_stage1/run_stage2 -- the exact node build() uses for recombination.")
      .def("formation_node", &Proton::formationNode, py::arg("seed"),
           "A fresh, seeded (not-yet-run) Step B node: the diquark {1,w} + the third "
           "quark {w*w} -> the proton singlet, on a single Delta^4 seed (output read off "
           "the whole). Drive it with run_stage1/run_stage2.")
      .def("converged", &Proton::converged,
           "True iff step B's proton block carries the singlet with enough holes.")
      .def("seed", &Proton::seed, "Base seed of the converged (or best) attempt.")
      .def("spacetime", &Proton::spacetime,
           "Step B's full relaxed closed-S^4 complex.")
      .def("block", &Proton::block,
           "Step B's proton sub-complex, with the relaxed metric copied in.")
      .def("quark_holes", &Proton::quarkHoles,
           "The emergent color holes on the proton block (>=3 when converged).")
      .def("color_residual", &Proton::colorResidual,
           "Step B's proton singlet r_state (~0 => carried).")
      .def("diquark_residual", &Proton::diquarkResidual,
           "Step A's r_U (small => the diquark recombination converged).");

  // === ProtonIngredients (#555): the emergent arm — nothing pinned downstream ===
  py::class_<ProtonIngredients>(m, "ProtonIngredients",
      R"doc(The emergent arm of the proton build (#555). Proton is the canonical line
in the sand and is composed here unchanged; ProtonIngredients prepares the same
ingredients through the same two-step drive EXCEPT that the final state is never
pinned: step B's output-target list is EMPTY, so the objective is
F = ||grad S||^2 + gamma * sum_i r_U(input_i) and whatever the whole cobordism
comes to carry is READ afterwards, never driven. Exactly one variable differs
from Proton.build() (the singlet output target), so the two classes form a clean
A/B experiment. The seed stays uniform and all-spacelike by design: at
initialization no time has passed — causal structure marks sequences of events
and may only emerge. Convergence carries no answer-shaped gate: an attempt
converges iff it is STATIONARY (stage 2 stopped on its stationarity test) and
PERSISTENT (a continued evolve+relax pass leaves holes, b_k, and F stable).
Everything physical is a post-hoc observable, including the singlet residual —
a diagnostic for comparing against the canonical build's carried level.)doc")
      .def(py::init<std::uint64_t, int, double, double, int, bool>(),
           py::arg("seed") = 0, py::arg("register_degree") = 3,
           py::arg("gamma") = 50.0, py::arg("input_weight") = 20.0,
           py::arg("precone") = 0, py::arg("should_use_directed_surgery") = false)
      .def("build", &ProtonIngredients::build, py::arg("max_restarts") = 16,
           py::arg("init_steps") = 180, py::arg("evolve_steps") = 60,
           py::arg("stage1_candidate_moves") = 8, py::arg("stage1_patience") = 15,
           py::arg("stage2_beta") = 1.0, py::arg("stage2_max_iters") = 10,
           py::arg("persist_rel_tol") = 0.05,
           "Restart across seeds until an attempt is stationary AND persistent (no "
           "color tolerance, no minimum hole count); otherwise keep the lowest-F "
           "attempt. Same drive per node as Proton.build().")
      .def("recombination_node", &ProtonIngredients::recombinationNode,
           py::arg("seed"),
           "Step A verbatim: the composed canonical Proton's recombination_node.")
      .def("formation_node", &ProtonIngredients::formationNode, py::arg("seed"),
           "Step B with nothing pinned: the same ideal diquark {1,w} + third quark "
           "{w*w} inputs on the same single Delta^4 seed as Proton.formation_node, "
           "but with an EMPTY output-target list — the final state emerges.")
      .def("joint_node", &ProtonIngredients::jointNode, py::arg("seed"),
           "The joint inputs-only node: ONE MultiCobordism whose inputs are the three "
           "Z3-symmetric neutral q-qbar pairs {1,-1,0} | {0,1,-1} | {-1,0,1} (each "
           "Sigma = 0 — the only prepared content, fixed for the whole build) and "
           "whose output-target list is EMPTY. No diquark, no bare quark, no "
           "intermediate imposed; the pre-registered expectation (a baryon with a "
           "conjugate partner) is READ off the relaxed whole afterwards — singlet and "
           "conjugate-singlet residuals as diagnostics, never drives. The two-step "
           "nodes remain the reference oracle. NOT run (the caller drives it).")
      .def("converged", &ProtonIngredients::converged,
           "True iff the kept attempt was stationary AND persistent — never a "
           "statement about the singlet or the hole count.")
      .def("stationary", &ProtonIngredients::stationary,
           "Whether the kept attempt's final run_stage2 stopped on stationarity.")
      .def("persistent", &ProtonIngredients::persistent,
           "Whether continued evolve+relax left holes, b_k, and F stable.")
      .def("seed", &ProtonIngredients::seed, "Base seed of the kept attempt.")
      .def("spacetime", &ProtonIngredients::spacetime,
           "The full relaxed emergent step-B complex.")
      .def("block", &ProtonIngredients::block,
           "The emergent object IS the whole step-B cobordism (parity with "
           "Proton.block).")
      .def("emergent_holes", &ProtonIngredients::emergentHoles,
           "The emergent register holes on the whole — an observable, not a gate; "
           "may be any count, including zero.")
      .def("singlet_residual", &ProtonIngredients::singletResidual,
           "DIAGNOSTIC only: the singlet r_state of Proton.singlet() against the "
           "whole, read after the fact for comparison with the canonical build. It "
           "never steers or gates this build.")
      .def("input_residual", &ProtonIngredients::inputResidual,
           "Step B's inputs-only r_U — the whole matter term of the emergent arm.")
      .def("final_objective", &ProtonIngredients::finalObjective,
           "The kept attempt's final objective F.")
      .def("diquark_residual", &ProtonIngredients::diquarkResidual,
           "Step A's r_U — reported exactly as Proton reports it.");

  // ----- Gated surgical cone-out/cone-in (topology change, #460) -----
  py::class_<SurgicalCone>(m, "SurgicalCone",
      R"doc(Gated surgical cone-out/cone-in: the topology-CHANGING move (#460, T3).

The genuine b_k-hole creator of the Emergent Color Topology epic (#457). Pachner
moves and the orientation-safe stellar refinement cone (T1/T2) are topology-
PRESERVING; this is not. coneOut removes one top cell (its orphaned edges, then
any isolated vertex) -- on a closed manifold this opens a manifold-with-boundary
and, for a cell disjoint from an existing hole, raises b_{d-1} by 1 (on S^3, the
color register's b_2). coneIn adds one top cell on a fresh vertex joined to d
existing vertices, lowering b_{d-1} by 1 when it caps a hole. EVERY move is gated
on ChainComplex.dualComplexIsValid (a valid manifold-with-boundary; the #429
n>=4 recursive check) -- surgery is allowed BECAUSE it is gated; bypassing the
gate is what broke the #353 weld. Rejected moves roll back bit-identically.
Accepted moves stack; rollback() undoes the last LIFO, restoring every edge
length and phase so a round trip leaves the dual Regge action (Re AND Im)
invariant.)doc")
      .def(py::init<Spacetime *>(), py::arg("spacetime"), py::keep_alive<1, 2>(),
           "Bind the cone to a spacetime (does not mutate it).")
      .def("coneOut", &SurgicalCone::coneOut, py::arg("cell"),
           "(ok, reason): gated surgical cone-out -- remove the top cell whose "
           "sorted vertex ids equal `cell` (plus orphaned edges and any vertex "
           "thereby isolated). Accepts only a valid manifold-with-boundary; "
           "otherwise restores the cell and names the reason. Rejects removing "
           "the last top cell.")
      .def("coneIn", &SurgicalCone::coneIn, py::arg("target_verts"),
           "(ok, reason): gated surgical cone-in -- create a fresh vertex, join "
           "it to the d `target_verts` to form a new top cell. Accepts only a "
           "valid manifold-with-boundary; otherwise undoes the additions.")
      .def("rollback", &SurgicalCone::rollback,
           "Undo the last accepted move (LIFO), restoring the complex bit-for-"
           "bit (edge lengths and phases). False if nothing is applied.")
      .def("rollbackAll", &SurgicalCone::rollbackAll,
           "Roll every accepted move back; returns the number undone.")
      .def_property_readonly("depth", &SurgicalCone::depth,
           "Number of accepted, not-yet-rolled-back moves on the stack.")
      .def_property_readonly("isApplied", &SurgicalCone::isApplied,
           "True iff at least one move is accepted and not yet rolled back.")
      .def("bettiNumbers", &SurgicalCone::bettiNumbers,
           "Betti numbers b_0..b_n (over Q) of the CURRENT complex -- the read-"
           "out the b_k-delta tests assert a surgical move shifts by one.")
      .def("validate", &SurgicalCone::validate,
           "(ok, reason): the manifold-with-boundary verdict on the CURRENT "
           "complex -- the same gate coneOut / coneIn apply.");
}
