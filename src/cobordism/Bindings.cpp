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
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cobordism/GeometrySynthesizer.h"
#include "cobordism/BoundaryStateSpace.h"
#include "cobordism/ChainComplex.h"
#include "cobordism/Characteristic.h"
#include "cobordism/Cochain.h"
#include "cobordism/Cobordism.h"
#include "cobordism/CombinatorialDimension.h"
#include "cobordism/DijkgraafWitten.h"
#include "cobordism/EigenstateSynthesis.h"
#include "cobordism/HodgeLaplacian.h"
#include "cobordism/IntegerLinalg.h"
#include "cobordism/PreparedBoundaryState.h"
#include "cobordism/RealizabilityOracle.h"
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
  py::class_<EigenstateSynthesis>(m, "EigenstateSynthesis",
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

Parameters: the per-edge squared-length magnitudes {w_ij} (Edge.setSquaredLength)
and U(1) phases {theta_ij} (Edge.setPhase), in a stable edge order fixed at
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
           "Edge magnitudes {w_ij} (squaredLength) in the stable edge order.")
      .def("phases", &EigenstateSynthesis::phases,
           "Edge phases {theta_ij} (radians) in the stable edge order.")
      .def("setWeights", &EigenstateSynthesis::setWeights, py::arg("w"),
           "Write the edge magnitudes in place. Raises if len(w) != numEdges().")
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
           "Interior edge magnitudes {w_ij} in interior-edge order.")
      .def("interiorPhases", &EigenstateSynthesis::interiorPhases,
           "Interior edge phases {theta_ij} (radians) in interior-edge order.")
      .def("setInteriorWeights", &EigenstateSynthesis::setInteriorWeights,
           py::arg("w"),
           "Write the interior edge magnitudes in place; the boundary edges are "
           "left untouched. Raises if len(w) != numInteriorEdges().")
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
           "Lets a surgery search try a removal, score it, and roll back.");

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

  // ----- §5.0 realizability oracle: spectral bulk synthesis for U (#138) -----
  py::class_<RealizabilityOracle> ro(m, "RealizabilityOracle",
      R"doc(§5.0 realizability oracle — spectral bulk synthesis for an operation U.

Decides whether U : H_B -> H_A is realizable as a bulk cobordism W_AB by
*synthesizing the bulk spectrally*, not by TQFT membership. U is bent to a
boundary state via Choi-Jamiolkowski (vec(U), the operator-as-state); the bulk's
boundary dW (the synthesized geo's / output surface) is *pinned*, and the
interior — its Hermitian edge weights and, via boundary-fixed Pachner growth, its
topology — is filled so the output-boundary k=0 graph-Laplacian eigenvector
matches the bent target, i.e. the §4b residual r = ||(I - psi psi^dagger) L psi||^2
is driven to 0.

Realizability is itself the test: U is realizable iff r can be driven below
epsilon, and non-realizable iff r floors away from 0 under the pinned boundary —
a spectral obstruction, the analogue of §4b's two-vertex floor
w_min^2(|c0|^2-|c1|^2)^2. The floor IS the certificate (non-existence is certified
by the residual floor at the explored interior complexity, not by exhausting
triangulations).

Pure orchestration of the merged building blocks (no new math):
ChoiJamiolkowski (the bend), EigenstateSynthesis (the fixed-boundary interior-fill
engine: setInteriorWeights/Phases vary only dW's complement, growInterior is the
boundary-fixed Pachner add), and LevenbergMarquardt (the same bounded multi-restart
least-squares solver the §4b GeometrySynthesizer drives, here over the interior
parameters + the free auxiliary amplitudes).

vec(U) is carried by the first dA*dB sorted-id vertices (the output-surface
support); remaining vertices (interior + coned-in apices, larger ids) carry free
auxiliary amplitudes the fill solves for. The caller assembles the bulk and pins
its boundary edges (Spacetime is built/pinned outside the synthesis classes);
decide() realizes the held bulk in place and returns it as the witness.)doc");

  py::enum_<RealizabilityOracle::GrowthMode>(ro, "GrowthMode",
      "How the interior fill grows when a pass cannot reach r < epsilon.")
      .value("CONE", RealizabilityOracle::GrowthMode::Cone,
             "Cone a fresh interior vertex into one top cell (the historical "
             "biased move, wiring it to exactly the d+1 cell vertices).")
      .value("FREE_CONNECTIVITY",
             RealizabilityOracle::GrowthMode::FreeConnectivity,
             "Free interior connectivity (#200): search a bounded set of candidate "
             "interior connectivities for the new vertex at each growth step and "
             "keep the one reaching the lowest residual — topology is emergent.")
      .value("SURGERY", RealizabilityOracle::GrowthMode::Surgery,
             "Surgery (#196): the topology-CHANGING move-set. At each step score "
             "every interior-top-cell removal (removeInteriorCell, which opens a "
             "hole/handle with dW held bit-exact) and commit the best improving "
             "one. Unlike FREE_CONNECTIVITY (additive only — spectrally inert at "
             "k>=1, so b_k is frozen at the seed), removal lets the search reach an "
             "arbitrary valid complex with the fixed boundary: b_k MOVES on its "
             "own. The companion of harmonic=True (realizable iff the boundary "
             "class is carried by H_k(W)).")
      .value("SURGERY_AND_CONE",
             RealizabilityOracle::GrowthMode::SurgeryAndCone,
             "The composed move-set: additions as well as surgical cuts. Each "
             "growth step commits the best IMPROVING interior-top-cell removal "
             "(the SURGERY step); when no cut improves it falls back to the "
             "additive cone (growInterior). max_cones budgets the ADDITIVE "
             "commits only — the added vertices, the resource a caller's "
             "--max-additional-vertices flag caps; cuts are bounded by the "
             "improving-only rule and the finite interior-cell set.");

  py::class_<RealizabilityOracle::Verdict>(ro, "Verdict",
      "The oracle's verdict on U: the realizability decision, the residual (the "
      "obstruction floor when non-realizable), the realized witness state and "
      "bulk W_AB, the realized eigenvalue, and the interior complexity reached.")
      .def_readonly("realizable", &RealizabilityOracle::Verdict::realizable,
                    "True iff the interior fill drove r < epsilon within the cone "
                    "budget (U realizable, witnessed by state on witness).")
      .def_readonly("residual", &RealizabilityOracle::Verdict::residual,
                    "Best residual r = ||(I - psi psi^dagger) L psi||^2 reached: "
                    "< epsilon when realizable, the certified floor otherwise.")
      .def_readonly("floor", &RealizabilityOracle::Verdict::floor,
                    "The obstruction floor (== residual when non-realizable, 0 "
                    "when realizable): no bulk of the explored interior complexity "
                    "realizes U under the pinned boundary.")
      .def_readonly("eigenvalue", &RealizabilityOracle::Verdict::eigenvalue,
                    "Realized eigenvalue lambda = psi^dagger L psi (Rayleigh "
                    "quotient) of the witness state — meaningful when realizable.")
      .def_readonly("interior_vertex_count",
                    &RealizabilityOracle::Verdict::interiorVertexCount,
                    "Interior vertices coned in to reach the verdict (the interior "
                    "complexity, the §4b (|V|,|E|) analogue under fixed ends).")
      .def_readonly("cones_applied", &RealizabilityOracle::Verdict::conesApplied,
                    "Boundary-fixed cones applied during the interior fill.")
      .def_readonly("interior_edge_count",
                    &RealizabilityOracle::Verdict::interiorEdgeCount,
                    "Interior tunable edges in the realized witness — the emergent "
                    "connectivity size. Cone growth adds exactly d+1 per vertex; "
                    "free-connectivity growth can differ (the headline observable).")
      .def_readonly("connectivity_candidates",
                    &RealizabilityOracle::Verdict::connectivityCandidates,
                    "Candidate interior EDGE (1-simplex) connectivities scored per "
                    "growth step (FREE_CONNECTIVITY only; 0 under CONE). The only "
                    "spectrally relevant atom at k=0.")
      .def_readonly("triangle_candidates",
                    &RealizabilityOracle::Verdict::triangleCandidates,
                    "Candidate interior TRIANGLE (2-simplex) connectivities scored "
                    "per growth step (FREE_CONNECTIVITY at k>=1 only; 0 at k=0 and "
                    "under CONE). At k>=1 the metric L_k reads the 2-cells through "
                    "d_2, so the search must also propose triangle attachments — "
                    "reported alongside connectivity_candidates so the full scored "
                    "breadth (edges + triangles) is surfaced, never capped silently.")
      .def_readonly("connectivity_space_size",
                    &RealizabilityOracle::Verdict::connectivitySpaceSize,
                    "Full per-step incidence space (2^N - 1 nonempty vertex "
                    "subsets) the candidates are pruned from at the last growth "
                    "step — connectivity_candidates << this documents the bound.")
      .def_readonly("surgery_removals",
                    &RealizabilityOracle::Verdict::surgeryRemovals,
                    "Interior top-cell removals committed by the surgery search "
                    "(SURGERY mode only; 0 otherwise). Each is a topology-changing "
                    "move that can shift b_k of the witness — the emergent-topology "
                    "trace (read the grown b_k off the witness with ChainComplex).")
      .def_readonly("state", &RealizabilityOracle::Verdict::state,
                    "The witness state: the realized unit Laplacian eigenvector on "
                    "W_AB (length = the bulk's vertex count); its first dA*dB "
                    "components are the output-boundary block matching target. A "
                    "genuine eigenstate iff realizable.")
      .def_readonly("target", &RealizabilityOracle::Verdict::target,
                    "The bent target psi_U = vec(U)/||vec(U)|| (length dA*dB) the "
                    "output boundary is matched against.")
      .def_readonly("witness", &RealizabilityOracle::Verdict::witness,
                    "The realized bulk W_AB: the witness cobordism (realizable), or "
                    "the complex on which the residual floors (non-realizable).");

  ro.def(py::init<std::shared_ptr<Spacetime>>(), py::arg("bulk"),
          "Build the oracle over the assembled bulk W_AB: a pre-geometric "
          "Spacetime whose codim-1 boundary is the (pinned) output surface plus "
          "input boundaries, output-surface vertices on the smallest ids. Raises "
          "if bulk is null.")
      .def("decide", &RealizabilityOracle::decide, py::arg("U"), py::arg("dA"),
           py::arg("dB"), py::arg("epsilon") = 1e-10, py::arg("restarts") = 64,
           py::arg("max_cones") = 4, py::arg("seed") = 0,
           py::arg("growth_mode") = RealizabilityOracle::GrowthMode::Cone,
           py::arg("connectivity_candidates") = 8, py::arg("harmonic") = false,
           "Decide whether the dA x dB operator U (flat row-major) is realizable "
           "as a bulk cobordism: bend it to vec(U), fill the pinned-boundary "
           "interior to drive the §4b residual to zero (multi-restart "
           "Levenberg-Marquardt over the interior weights/phases + auxiliary "
           "amplitudes, growing the interior up to max_cones times), and return "
           "the Verdict. growth_mode=CONE (default) keeps the historical cone-only "
           "growth; growth_mode=FREE_CONNECTIVITY searches connectivity_candidates "
           "interior connectivities per growth step and keeps the best by residual "
           "(the new vertex's incidence is a free variable; topology is emergent). "
           "Realizes the held bulk in place. Raises if U.size() != dA*dB, a "
           "dimension is non-positive, or the bulk has fewer vertices than dA*dB.")
      .def("decideHarmonic", &RealizabilityOracle::decideHarmonic,
           py::arg("target"), py::arg("epsilon") = 1e-10, py::arg("restarts") = 64,
           py::arg("max_cones") = 4, py::arg("seed") = 0,
           py::arg("growth_mode") = RealizabilityOracle::GrowthMode::Cone,
           py::arg("connectivity_candidates") = 8, py::arg("harmonic") = false,
           "Decide whether a target boundary harmonic k-form (a degree-k Cochain, "
           "k = target.degree(); the k=1 DW setting) is realizable on the held "
           "3-manifold-with-boundary bulk W: pin the boundary surface dW byte-"
           "fixed, fill the interior (interior edge squared-lengths + boundary-"
           "fixed growth) to drive r = ||(I - psi psi^dagger) L_k psi||^2 "
           "to 0, and return the Verdict. target is matched to the bulk's boundary "
           "k-cells by sorted vertex-id tuple (the readout of a "
           "PreparedBoundaryState); interior k-cells carry free auxiliary "
           "amplitudes. growth_mode=CONE (default) keeps the boundary-fixed "
           "1->(d+1) Pachner add; growth_mode=FREE_CONNECTIVITY searches interior "
           "connectivity per growth step — at k>=1 proposing both edge (1-simplex) "
           "and TRIANGLE (2-simplex) attachments, the latter being what L_k reads "
           "through d_2 (connectivity_candidates per atom kind, surfaced as "
           "connectivity_candidates / triangle_candidates). Realizable iff "
           "r < epsilon, else the floor certifies non-realizability at the explored "
           "complexity. Realizes the held bulk in place. Raises if target is "
           "empty/negative-degree or none of its k-cells are boundary cells.");

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
      .def_static("disjointUnion", &Cobordism::disjointUnion, py::arg("W1"),
                  py::arg("W2"),
                  "Disjoint union W1 ⊔ W2: shift W2's vertices into a fresh id "
                  "range above W1's (nothing identified) and concatenate the top "
                  "simplices into one Spacetime. ∂(W1 ⊔ W2) = ∂W1 ⊔ ∂W2 and the "
                  "bulk is disconnected, so two solid tori S¹×D² give the cap-and-"
                  "create cobordism T² → T² (∂W = T² ⊔ T²) — not a mapping "
                  "cylinder — whose DijkgraafWitten.map() is the rank-1, non-"
                  "invertible outer product |st⟩⟨st|. Returns the new Spacetime, "
                  "ready for DijkgraafWitten.map() (whose two-component boundary "
                  "requirement it meets while the bulk need not be connected). "
                  "Raises if either input is empty or their top dimensions differ.")
      .def_static("selfGlue", &Cobordism::selfGlue, py::arg("W"),
                  "Close a cobordism by gluing its two boundary components to "
                  "each other (the mapping torus / categorical trace): they must "
                  "be isomorphic, and the collar must be thick enough that no "
                  "top simplex touches both (>= 3 layers in the glued "
                  "direction). Returns the closed manifold as a new Spacetime. "
                  "Raises unless dW has exactly two isomorphic components, or if "
                  "the identification collapses a top simplex.")
      .def_static("twistedCylinder", &Cobordism::twistedCylinder,
                  py::arg("sigma"), py::arg("phi"),
                  "Build the phi-twisted product cobordism Sigma x [0,T] : "
                  "Sigma -> Sigma whose two boundary copies of the surface Sigma "
                  "are threaded through a finite-order simplicial automorphism "
                  "phi (phi[x] = image of vertex x; Sigma's vertices must be "
                  "0..|V|-1). The ordinary product cylinder (phi = identity) has "
                  "DW map the identity; a phi acting non-trivially on "
                  "H^1(Sigma; Z_2) makes DijkgraafWitten.map() the corresponding "
                  "non-identity permutation of the holonomy classes Z(Sigma) — "
                  "e.g. the coordinate swap (x,y)->(y,x) of a square-product "
                  "torus transposes [a]<->[b]. Three stacked Sigma levels (ids "
                  "x, |V|+x, 2|V|+x) with the twist carried only by the seam "
                  "(level 1) identification; returns the new Spacetime, ready for "
                  "DijkgraafWitten.map(). Raises if Sigma is not a 2D "
                  "triangulation with vertices 0..|V|-1, or phi is not a length-"
                  "|V| permutation / not a simplicial automorphism of Sigma.");

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
           "Z(W)_ab psiB[b] for already-prepared boundary states (two "
           "PreparedBoundaryStates, type-safe; reads each one's coeffs() in the "
           "flat-connection-class basis, |psiA| = 2^{b1(Sigma_A)}, |psiB| = "
           "2^{b1(Sigma_B)}). For the cylinder Z(W)=id this is the inner product "
           "<psiA|psiB>; with states from BoundaryStateSpace.prepare it is the "
           "harmonic overlap.")
      .def_static("isCocycle", &DijkgraafWitten::isCocycle, py::arg("cocycle"),
                  "Whether omega satisfies the normalized 3-cocycle (pentagon) "
                  "identity over Z_2 (brute-forced over all 16 tuples of "
                  "Z_2^4). True for both Trivial and Sign.");

  // ----- §5 the DW boundary Hilbert space Z(Sigma) + its states (#175, #187) ----
  py::class_<BoundaryStateSpace, std::shared_ptr<BoundaryStateSpace>>(
      m, "BoundaryStateSpace",
      R"doc(The DW boundary Hilbert space Z(Sigma) of a closed surface Sigma.

A per-Sigma context / factory (conceptually Z(Sigma) = C[H^1(Sigma; Z_2)]). It
owns Sigma and the cached Hodge harmonic basis of ker L_1(Sigma) (HodgeLaplacian
at k=1, computed once), and manufactures the value objects (PreparedBoundaryState)
that live in it. Two distinct objects sit on Sigma: the harmonic 1-forms
ker L_1(Sigma) (the spectral qubit, dim b_1, an orthonormal basis of degree-1
Cochains) and the flat-connection-class space Z(Sigma) (dim 2^{b_1}).

b_1 vs 2^{b_1}: H^1(Sigma; Z_2) = (Z_2)^{b_1} has b_1 single-generator classes,
the weight-one holonomy patterns, at gf2Span indices 2^0, 2^1, ..., 2^{b_1-1}
(mask 2^i = basis vector i alone). This 'harmonic i -> index 2^i' convention is
owned here (generatorIndices()) — callers never spell out a power of two.

prepare(form) embeds a harmonic 1-form Cochain onto those generator slots
(scatter c_i = <h_i, form> to index 2^i); state(amplitudes) wraps a raw Z(Sigma)
vector directly. Both return a PreparedBoundaryState bound to this space. The
harmonic basis is orthonormal, so prepare is an isometry and
PreparedBoundaryState.readout() (its adjoint) round-trips it; hence on the
trivial cobordism Sigma x [0,T] (Z(W)=id), DijkgraafWitten.amplitude reproduces
<psi|phi>. Reuses HodgeLaplacian (k=1); the basis is cached at construction.)doc")
      .def(py::init<std::shared_ptr<Spacetime>, double, bool>(),
           py::arg("sigma"), py::arg("tol") = 1e-9, py::arg("metric") = true,
           "Build Z(Sigma) over a closed surface Sigma. Computes and caches "
           "ker L_1(Sigma) (HodgeLaplacian k=1); tol is the |lambda|<tol harmonic "
           "threshold and metric selects volume vs unit weights (both forwarded to "
           "harmonics(1, tol, metric); the embedding is isometric either way). "
           "Raises if Sigma is null or b_1(Sigma) > 24.")
      .def("harmonicDimension", &BoundaryStateSpace::harmonicDimension,
           "b_1(Sigma) = dim ker L_1(Sigma): the spectral-qubit dimension (the "
           "number of harmonic 1-forms).")
      .def("boundaryDimension", &BoundaryStateSpace::boundaryDimension,
           "dim Z(Sigma) = 2^{b_1(Sigma)}: the DW boundary Hilbert-space "
           "dimension (the length of a prepared state).")
      .def("numEdges", &BoundaryStateSpace::numEdges,
           "|C_1(Sigma)|: the number of edges — the length of a harmonic 1-form "
           "(the input to prepare / output of readout).")
      .def("harmonics", &BoundaryStateSpace::harmonics,
           "The cached orthonormal harmonic 1-form basis as a list of degree-1 "
           "Cochains (the k=1 simplex ordering), ascending-eigenvalue order — the "
           "deterministic basis prepare/readout use.")
      .def("generatorIndices", &BoundaryStateSpace::generatorIndices,
           "The b_1 flat-connection-class indices in Z(Sigma) carrying harmonic "
           "data: (2^0, 2^1, ..., 2^{b_1-1}), the single-generator classes "
           "(harmonic i lands on index 2^i).")
      .def("prepare", &BoundaryStateSpace::prepare, py::arg("form"),
           "Prepare a boundary state from a harmonic 1-form: ker L_1(Sigma) -> "
           "Z(Sigma). form is a degree-1 Cochain of length |C_1|; its harmonic "
           "coordinates c_i = <h_i, form> are scattered onto the amplitudes at "
           "indices 2^i, the rest zero. Returns a PreparedBoundaryState. A "
           "non-harmonic component is projected out. Raises if form.degree() != 1 "
           "or len(form) != numEdges().")
      .def("state", &BoundaryStateSpace::state, py::arg("amplitudes"),
           "Wrap a raw Z(Sigma) amplitude vector (the flat-connection-class "
           "basis, length 2^{b_1}) as a PreparedBoundaryState over this space — "
           "the direct counterpart to prepare for states already in the "
           "holonomy-class basis. Raises if len(amplitudes) != "
           "boundaryDimension().");

  py::class_<PreparedBoundaryState>(m, "PreparedBoundaryState",
      R"doc(A state in the DW boundary Hilbert space Z(Sigma) of a surface Sigma.

A value object wrapping the 2^{b_1(Sigma)}-long complex amplitude vector (the
flat-connection-class basis) plus a handle to the BoundaryStateSpace that defines
Sigma. Produced by BoundaryStateSpace.prepare (from a harmonic 1-form) or
BoundaryStateSpace.state (from a raw amplitude vector); the held handle keeps its
space alive. The 'harmonic i -> index 2^i' convention lives in the space, so
generatorAmplitude(i) and readout() delegate to it.

readout() is the adjoint of prepare: it rebuilds the harmonic 1-form
sum_i c_i h_i in ker L_1(Sigma) from c_i = coeffs[2^i], so
space.prepare(form).readout() == form for form in ker L_1. overlap(other) is the
Hermitian inner product <self, other> = sum conj(self) other (= np.vdot) and, for
prepared states, reproduces the harmonic overlap.)doc")
      .def("coeffs", &PreparedBoundaryState::coeffs,
           "The Z(Sigma) amplitude vector (the flat-connection-class basis) as a "
           "1-D complex numpy.ndarray (Eigen-backed).")
      .def("size", &PreparedBoundaryState::size,
           "dim Z(Sigma) = 2^{b_1(Sigma)}, the number of holonomy classes.")
      .def("__len__", &PreparedBoundaryState::size)
      .def("space",
           [](const PreparedBoundaryState &p) {
             return std::const_pointer_cast<BoundaryStateSpace>(p.space());
           },
           "The BoundaryStateSpace Z(Sigma) this state belongs to.")
      .def("amplitude", &PreparedBoundaryState::amplitude,
           py::arg("holonomyClass"),
           "The amplitude for a single holonomy class (flat-connection class "
           "index 0..2^{b_1}-1). Raises IndexError if out of range.")
      .def("__getitem__", &PreparedBoundaryState::amplitude)
      .def("generatorAmplitude", &PreparedBoundaryState::generatorAmplitude,
           py::arg("harmonic"),
           "The amplitude carried by the harmonic-th harmonic 1-form — the "
           "single-generator class at Z(Sigma) index 2^harmonic (the convention "
           "owned by the BoundaryStateSpace). Raises IndexError if not in "
           "[0, b_1).")
      .def("readout", &PreparedBoundaryState::readout,
           "Read the harmonic 1-form back out (the adjoint of prepare): the "
           "degree-1 Cochain sum_i c_i h_i in ker L_1(Sigma) with c_i = "
           "coeffs[2^i]. space.prepare(form).readout() == form for form in "
           "ker L_1(Sigma).")
      .def("overlap", &PreparedBoundaryState::overlap, py::arg("other"),
           "The Hermitian inner product <self, other> = sum conj(self) other "
           "(= np.vdot) of the two Z(Sigma) amplitude vectors. For prepared "
           "states this is the harmonic overlap. Raises if the boundary "
           "dimensions differ.")
      .def("norm", &PreparedBoundaryState::norm,
           "The Euclidean norm sqrt(sum |c_a|^2) of the amplitude vector.");
}
