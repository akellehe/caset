// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include <limits>

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "chainhodge/ChainHodge.h"
#include "chainhodge/LorentzianFamily.h"
#include "chainhodge/WhitneyMass.h"
#include "cobordism/ChainComplex.h"
#include "spacetime/Spacetime.h"

namespace py = pybind11;
using namespace tessera::chainhodge;
using tessera::cobordism::ChainComplex;

void register_chainhodge(py::module_ m) {
  py::enum_<Preset>(m, "Preset",
      "Chain metric family: L2 (Whitney, default) or GRASSMANN_ALL (the retained "
      "Grassmann projection metric with its documented deviation).")
      .value("L2", Preset::L2)
      .value("GRASSMANN_ALL", Preset::GRASSMANN_ALL);

  py::enum_<Branch>(m, "Branch",
      "How sqrt(det g_T) is fixed per top simplex: continuation from the unit "
      "Euclidean reference along the straight segment, or the Kontsevich-Segal "
      "rule (principal eigenvalue roots, cut resolved to +i).")
      .value("Continuation", Branch::Continuation)
      .value("KontsevichSegal", Branch::KontsevichSegal);

  py::class_<InstanceCertificate>(m, "InstanceCertificate",
      R"doc(The instance certificate of specification §4.2: Kontsevich-Segal allowability
of every top simplex, the minimal margin pi - sum_i |arg lambda_i(g_T)|, the volumes on
the declared branch, the Gram determinants, continuation ambiguity, and the Lorentzian
protocol rotation epsilon (NaN until set).)doc")
      .def_readonly("branch", &InstanceCertificate::branch)
      .def_readonly("allowable", &InstanceCertificate::allowable)
      .def_readonly("margin", &InstanceCertificate::margin)
      .def_readonly("margins", &InstanceCertificate::margins)
      .def_readonly("volumes", &InstanceCertificate::volumes)
      .def_readonly("gramDeterminants", &InstanceCertificate::gramDeterminants)
      .def_readonly("continuationAmbiguous", &InstanceCertificate::continuationAmbiguous)
      .def_readonly("ambiguousTopSimplices", &InstanceCertificate::ambiguousTopSimplices)
      .def_readwrite("epsilon", &InstanceCertificate::epsilon);

  py::class_<TopSimplexBlock>(m, "TopSimplexBlock",
      "One top simplex's local block of M_k over its k-faces, with the canonical "
      "cell and edge indices and (when requested) the derivative per local edge.")
      .def_readonly("topIndex", &TopSimplexBlock::topIndex)
      .def_readonly("cellIndices", &TopSimplexBlock::cellIndices)
      .def_readonly("edgeIndices", &TopSimplexBlock::edgeIndices)
      .def_readonly("block", &TopSimplexBlock::block)
      .def_readonly("derivative", &TopSimplexBlock::derivative);

  py::class_<WhitneyMass>(m, "WhitneyMass",
      R"doc(Sparse complex symmetric inverse chain metrics M_k of the chain-level Whitney
Hodge pencil, assembled per top simplex from the complex squared edge lengths alone
(specification §4.1, §4.2). Reference orientation is ascending vertex id
(ChainComplex.fromTopCells). Sparse results are scipy CSC matrices.)doc")
      .def_static("complexOf", &WhitneyMass::complexOf, py::arg("spacetime"),
           "ChainComplex.fromTopCells over the spacetime's top simplices (sorted vertex ids).")
      .def_static("squaredLengthsOf", &WhitneyMass::squaredLengthsOf,
           py::arg("spacetime"), py::arg("complex"),
           "Complex squared edge lengths l^2 in the complex's canonical edge order.")
      .def_static("assemble",
           py::overload_cast<const ChainComplex &, const SquaredLengths &, int, Branch>(
               &WhitneyMass::assemble),
           py::arg("complex"), py::arg("squared_lengths"), py::arg("k"),
           py::arg("branch") = Branch::Continuation,
           "The Whitney inverse chain metric M_k (Preset.L2) on the declared branch.")
      .def_static("assemblePreset",
           py::overload_cast<const ChainComplex &, const SquaredLengths &, int, Preset, Branch>(
               &WhitneyMass::assemble),
           py::arg("complex"), py::arg("squared_lengths"), py::arg("k"), py::arg("preset"),
           py::arg("branch") = Branch::Continuation,
           "Preset dispatch: L2 -> M_k (inverse chain metric); GRASSMANN_ALL -> G_k (chain metric).")
      .def_static("assembleGrassmann", &WhitneyMass::assembleGrassmann,
           py::arg("complex"), py::arg("squared_lengths"), py::arg("k"),
           "The Grassmann projection chain metric G_k = multiplicity o blade pairing (CH §6).")
      .def_static("allowabilityMargin", &WhitneyMass::allowabilityMargin,
           py::arg("complex"), py::arg("squared_lengths"),
           "min over top simplices of pi - sum_i |arg lambda_i(g_T)|.")
      .def_static("certificate", &WhitneyMass::certificate,
           py::arg("complex"), py::arg("squared_lengths"),
           py::arg("branch") = Branch::Continuation,
           "The full instance certificate (§4.2).")
      .def_static("volumeOnBranch",
           [](const Eigen::MatrixXcd &gram, Branch branch) {
             bool ambiguous = false;
             const Complex v = WhitneyMass::volumeOnBranch(gram, branch, &ambiguous);
             return py::make_tuple(v, ambiguous);
           },
           py::arg("gram"), py::arg("branch") = Branch::Continuation,
           "(sqrt(det g)/d!, ambiguous) for one Gram matrix on the declared branch.")
      .def_static("marginOf", &WhitneyMass::marginOf, py::arg("gram"),
           "pi - sum_i |arg lambda_i(g)| for one Gram matrix.")
      .def_static("topSimplexBlocks", &WhitneyMass::topSimplexBlocks,
           py::arg("complex"), py::arg("squared_lengths"), py::arg("k"),
           py::arg("branch") = Branch::Continuation, py::arg("with_derivative") = false,
           "Per-top-simplex local blocks of M_k (and derivatives when requested).")
      .def_static("assembleDerivative", &WhitneyMass::assembleDerivative,
           py::arg("complex"), py::arg("squared_lengths"), py::arg("k"), py::arg("edge_index"),
           py::arg("branch") = Branch::Continuation,
           "dM_k/ds_e for the edge at the given canonical index, sparse.")
      .def_static("derivativeContraction", &WhitneyMass::derivativeContraction,
           py::arg("complex"), py::arg("squared_lengths"), py::arg("k"),
           py::arg("X"), py::arg("Y"), py::arg("branch") = Branch::Continuation,
           "Per-edge tr(X^T (dM_k/ds_e) Y) from the local blocks (transpose pairing).");
  py::enum_<PencilVariable>(m, "PencilVariable",
      "Which vector a pencil eigenproblem A x = lambda B x is written in: geometric "
      "images (Whitney) or chains (Grassmann).")
      .value("GeometricImage", PencilVariable::GeometricImage)
      .value("Chain", PencilVariable::Chain);

  py::class_<Pencil>(m, "Pencil", "A complex symmetric pencil A - lambda B at one degree, dense.")
      .def_readonly("degree", &Pencil::degree)
      .def_readonly("variable", &Pencil::variable)
      .def_readonly("A", &Pencil::A)
      .def_readonly("B", &Pencil::B);

  py::class_<HarmonicRead>(m, "HarmonicRead",
      "Harmonic chains H_k, their geometric images, and the kernel's rank certificate.")
      .def_readonly("degree", &HarmonicRead::degree)
      .def_readonly("chains", &HarmonicRead::chains)
      .def_readonly("images", &HarmonicRead::images)
      .def_readonly("nullity", &HarmonicRead::nullity)
      .def_readonly("rank", &HarmonicRead::rank)
      .def_readonly("tolerance", &HarmonicRead::tolerance)
      .def_readonly("gap", &HarmonicRead::gap)
      .def_readonly("dense", &HarmonicRead::dense);

  py::class_<RankReport>(m, "RankReport",
      "The rank conditions (R1)-(R4) of specification Prop. 4.2 at one degree.")
      .def_readonly("degree", &RankReport::degree)
      .def_property_readonly("measured", [](const RankReport &r) {
        return std::vector<int>(r.measured.begin(), r.measured.end()); })
      .def_property_readonly("expected", [](const RankReport &r) {
        return std::vector<int>(r.expected.begin(), r.expected.end()); })
      .def_property_readonly("holds", [](const RankReport &r) {
        return std::vector<bool>(r.holds.begin(), r.holds.end()); })
      .def_readonly("decompositionHolds", &RankReport::decompositionHolds)
      .def_readonly("kernelIsHarmonic", &RankReport::kernelIsHarmonic)
      .def_readonly("kappa", &RankReport::kappa);

  py::class_<SpectrumRead>(m, "SpectrumRead", "Dense spectrum of one degree's pencil.")
      .def_readonly("degree", &SpectrumRead::degree)
      .def_readonly("eigenvalues", &SpectrumRead::eigenvalues)
      .def_readonly("residual", &SpectrumRead::residual)
      .def_readonly("vectors", &SpectrumRead::vectors);

  py::class_<ChainHodge>(m, "ChainHodge",
      R"doc(The chain-level Hodge pencil of a complexified simplicial complex (specification
§4.3, §9, §13): sparse inverse chain metrics, geometric images by solves, the symmetric
pencil and its auxiliary form, harmonic chains H_k = M_k ker S, rank conditions R1-R4,
exact Betti numbers, and the dense spectrum below the crossover. The adjoint is the
transpose; no conjugation enters any operator.)doc")
      .def(py::init<ChainComplex, SquaredLengths, Preset, Branch, int, double>(),
           py::arg("complex"), py::arg("squared_lengths"), py::arg("preset") = Preset::L2,
           py::arg("branch") = Branch::Continuation,
           py::arg("crossover_dimension") = ChainHodge::kDefaultCrossoverDimension,
           py::arg("epsilon") = std::numeric_limits<double>::quiet_NaN())
      .def("complex", &ChainHodge::complex, py::return_value_policy::reference_internal)
      .def("squaredLengths", &ChainHodge::squaredLengths)
      .def("dimension", &ChainHodge::dimension)
      .def("preset", &ChainHodge::preset)
      .def("branch", &ChainHodge::branch)
      .def("crossoverDimension", &ChainHodge::crossoverDimension)
      .def("certificate", &ChainHodge::certificate)
      .def("size", &ChainHodge::size, py::arg("k"))
      .def("Minv", [](const ChainHodge &c, int k) { return SparseMatrix(c.Minv(k)); }, py::arg("k"),
           "The sparse inverse chain metric M_k (Whitney preset).")
      .def("chainMetricSparse", [](const ChainHodge &c, int k) { return SparseMatrix(c.chainMetricSparse(k)); },
           py::arg("k"), "The sparse chain metric G_k (Grassmann preset).")
      .def("boundary", [](const ChainHodge &c, int k) { return SparseMatrix(c.boundary(k)); }, py::arg("k"),
           "The sparse boundary map d_k.")
      .def("applyG", &ChainHodge::applyG, py::arg("k"), py::arg("c"), "G_k c: the geometric image, by solve.")
      .def("applyMinv", &ChainHodge::applyMinv, py::arg("k"), py::arg("c"), "M_k c.")
      .def("pencil", &ChainHodge::pencil, py::arg("k"), "The dense pencil at degree k.")
      .def("pencilAux", &ChainHodge::pencilAux, py::arg("k"), "A~_k = M_k A_k M_k (Whitney), dense.")
      .def("hodgeOperator", &ChainHodge::hodgeOperator, py::arg("k"), "The dense L_k on chains.")
      .def("harmonicChains", &ChainHodge::harmonicChains, py::arg("k"), py::arg("kappa") = 10.0,
           py::arg("force_sparse") = false, "H_k = M_k ker S with the kernel's rank certificate.")
      .def("geometricImage", &ChainHodge::geometricImage, py::arg("k"), py::arg("H"), "G_k H.")
      .def("harmonicGram", &ChainHodge::harmonicGram, py::arg("read"), "Phi^T G_k Phi = Z^T M_k Z.")
      .def("rankConditions", &ChainHodge::rankConditions, py::arg("k"), py::arg("kappa") = 10.0,
           "The rank conditions (R1)-(R4) at degree k.")
      .def("betti", &ChainHodge::betti, "Betti numbers over Q, exact.")
      .def("spectrum", &ChainHodge::spectrum, py::arg("k"), "Dense spectrum of the degree-k pencil.");
  py::enum_<CausalType>(m, "CausalType",
      "Declared causal type of an edge (an input, never inferred from a squared length).")
      .value("Spacelike", CausalType::Spacelike)
      .value("Timelike", CausalType::Timelike)
      .value("Null", CausalType::Null);

  py::class_<LorentzianRead>(m, "LorentzianRead",
      "One member of the epsilon family: allowability, margin, the harmonic read with its "
      "gap, and the dense spectrum when requested.")
      .def_readonly("epsilon", &LorentzianRead::epsilon)
      .def_readonly("allowable", &LorentzianRead::allowable)
      .def_readonly("margin", &LorentzianRead::margin)
      .def_readonly("degree", &LorentzianRead::degree)
      .def_readonly("harmonic", &LorentzianRead::harmonic)
      .def_readonly("eigenvalues", &LorentzianRead::eigenvalues);

  py::class_<LorentzianExtrapolation>(m, "LorentzianExtrapolation",
      "A labeled least-squares extrapolation of reads at epsilon > 0 to epsilon -> 0.")
      .def_readonly("epsilons", &LorentzianExtrapolation::epsilons)
      .def_readonly("values", &LorentzianExtrapolation::values)
      .def_readonly("order", &LorentzianExtrapolation::order)
      .def_readonly("extrapolated", &LorentzianExtrapolation::extrapolated)
      .def_readonly("residual", &LorentzianExtrapolation::residual)
      .def_readonly("label", &LorentzianExtrapolation::label);

  py::class_<LorentzianFamily>(m, "LorentzianFamily",
      R"doc(The Lorentzian protocol (specification §10): the family s_e(epsilon) with the
timelike squared lengths rotated by e^{-2 i epsilon} at reported epsilon > 0; reads at
epsilon = 0 exist only inside a family and carry their gap; extrapolation to
epsilon -> 0 is a separate, labeled step.)doc")
      .def_static("rotate", &LorentzianFamily::rotate, py::arg("squared_lengths"),
           py::arg("causal_types"), py::arg("epsilon"),
           "Timelike entries times e^{-2 i epsilon}; others unchanged.")
      .def_static("instance", &LorentzianFamily::instance, py::arg("complex"),
           py::arg("squared_lengths"), py::arg("causal_types"), py::arg("epsilon"),
           py::arg("preset") = Preset::L2, py::arg("branch") = Branch::Continuation,
           py::arg("crossover_dimension") = ChainHodge::kDefaultCrossoverDimension,
           "The ChainHodge at epsilon, with epsilon on its certificate.")
      .def_static("sweep", &LorentzianFamily::sweep, py::arg("complex"), py::arg("squared_lengths"),
           py::arg("causal_types"), py::arg("epsilons"), py::arg("degree"),
           py::arg("preset") = Preset::L2, py::arg("branch") = Branch::Continuation,
           py::arg("kappa") = 10.0, py::arg("with_spectrum") = false,
           py::arg("crossover_dimension") = ChainHodge::kDefaultCrossoverDimension,
           "Reads at every epsilon of the family at one degree.")
      .def_static("extrapolateToZero", &LorentzianFamily::extrapolateToZero, py::arg("epsilons"),
           py::arg("values"), py::arg("order") = 2,
           "Labeled polynomial extrapolation of reads at epsilon > 0 to epsilon -> 0.");
}
