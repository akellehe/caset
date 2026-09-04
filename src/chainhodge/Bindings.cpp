// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
}
