// Pybind11 bindings for tessera.quantum.cuda. Registered as a sub-
// module of tessera.quantum from src/quantum/bindings.cpp under the
// TESSERA_QUANTUM_CUDA build flag.
//
// Surface: the torch::Tensor type is exposed natively (via the torch
// pybind type caster) so callers can chain torch ops on the GPU
// without round-tripping through numpy. This breaks the "scalars in,
// scalars out" rule that the ITensor bindings keep, but it's the
// whole point of the libtorch mirror — the user gets a real
// torch.Tensor back, on the device the MPS lives on.

#include "quantum/cuda/mps.hpp"
#include "quantum/cuda/mutual_information.hpp"

#include <torch/torch.h>
// torch's pybind11 type caster for at::Tensor and the small types
// (Device, Dtype, ScalarType). Required so we can pass torch::Tensor
// values across the binding boundary directly.
#include <torch/csrc/utils/pybind.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <vector>

namespace py = pybind11;

namespace {

torch::Device parseDevice(py::object obj) {
    if (py::isinstance<py::str>(obj)) {
        return torch::Device(obj.cast<std::string>());
    }
    return obj.cast<torch::Device>();
}

torch::Dtype parseDtype(py::object obj) {
    if (py::isinstance<py::str>(obj)) {
        auto name = obj.cast<std::string>();
        if (name == "complex128") return torch::kComplexDouble;
        if (name == "complex64")  return torch::kComplexFloat;
        if (name == "float64")    return torch::kFloat64;
        if (name == "float32")    return torch::kFloat32;
        throw std::invalid_argument(
            "Unknown dtype string: " + name);
    }
    return obj.cast<torch::Dtype>();
}

} // namespace

PYBIND11_MODULE(_tessera_cuda, cuda) {
    using namespace tessera::quantum::cuda;
    cuda.def("debug_return_zeros", []() {
        return torch::zeros({2, 2}, torch::TensorOptions().dtype(torch::kComplexDouble));
    });

    cuda.doc() = R"doc(
torch / libtorch backend mirroring the C++ ITensor quantum pipeline.

Tensors live on the configured torch device (default ``cuda:0`` if
available, ``cpu`` otherwise) at ``torch.complex128``. Class names
mirror those in ``tessera.quantum`` — :class:`MPS`,
:class:`MutualInformation`, … — so cross-validation tests can be
written without translation.

Site indices are 0-based throughout this module. The ITensor binding's
``MutualInformation`` and ``MPS`` analogues use 1-based site indices;
translate at the call boundary when cross-validating.
)doc";

    py::class_<MPS>(cuda, "MPS",
        "Matrix product state on torch tensors.")

        .def_static("product_state",
            [](std::vector<torch::Tensor> site_states,
                py::object device, py::object dtype) {
                return MPS::productState(std::move(site_states),
                                            parseDevice(device),
                                            parseDtype(dtype));
            },
            py::arg("site_states"),
            py::arg("device") = py::str("cpu"),
            py::arg("dtype")  = py::str("complex128"),
            "Build a product MPS from one length-d ket per site.")

        .def_static("computational_basis",
            [](std::vector<int64_t> bits, int64_t d,
                py::object device, py::object dtype) {
                return MPS::computationalBasis(std::move(bits), d,
                                                  parseDevice(device),
                                                  parseDtype(dtype));
            },
            py::arg("bits"),
            py::arg("d") = 2,
            py::arg("device") = py::str("cpu"),
            py::arg("dtype")  = py::str("complex128"),
            "Product state where bits[k] selects |bits[k]> on site k.")

        .def_static("bell_chain",
            [](int64_t N_pairs, py::object device, py::object dtype) {
                return MPS::bellChain(N_pairs,
                                        parseDevice(device),
                                        parseDtype(dtype));
            },
            py::arg("N_pairs"),
            py::arg("device") = py::str("cpu"),
            py::arg("dtype")  = py::str("complex128"),
            "|Phi+>^(N_pairs) on 2*N_pairs sites.")

        .def_static("ghz",
            [](int64_t N, int64_t d,
                py::object device, py::object dtype) {
                return MPS::ghz(N, d,
                                  parseDevice(device),
                                  parseDtype(dtype));
            },
            py::arg("N"),
            py::arg("d") = 2,
            py::arg("device") = py::str("cpu"),
            py::arg("dtype")  = py::str("complex128"),
            "GHZ state on N sites with physical dim d.")

        .def_static("random",
            [](int64_t N, int64_t d, int64_t max_chi,
                py::object device, py::object dtype,
                std::optional<int64_t> seed) {
                return MPS::random(N, d, max_chi,
                                     parseDevice(device),
                                     parseDtype(dtype),
                                     seed);
            },
            py::arg("N"), py::arg("d"), py::arg("max_chi"),
            py::arg("device") = py::str("cpu"),
            py::arg("dtype")  = py::str("complex128"),
            py::arg("seed")   = py::none(),
            "Random complex MPS, left-canonical and unit-normalised.")

        .def("__len__", &MPS::length)
        .def_property_readonly("length", &MPS::length)
        .def_property_readonly("device",
            [](MPS const& m) { return m.device(); })
        .def_property_readonly("dtype",
            [](MPS const& m) { return m.dtype(); })
        .def_property_readonly("physical_dim", &MPS::physicalDim)
        .def_property_readonly("orth_centre", &MPS::orthCentre)
        .def("bond_dim", &MPS::bondDim, py::arg("k"))
        .def("canonicalize", &MPS::canonicalize, py::arg("oc"),
            "Move the orthogonality centre to site oc.")
        .def("norm_squared", &MPS::normSquared)
        .def("normalize", &MPS::normalize)
        .def("one_site_reduced_density",
            &MPS::oneSiteReducedDensity, py::arg("i"),
            "rho_i as a (d, d) torch.Tensor.")
        .def("two_site_reduced_density",
            &MPS::twoSiteReducedDensity, py::arg("i"), py::arg("j"),
            "rho_ij as a (d^2, d^2) torch.Tensor; basis order matches "
            "MutualInformation::twoSiteReducedDensity.")
        .def("is_left_canonical", &MPS::isLeftCanonical,
            py::arg("k"), py::arg("atol") = 1e-9)
        .def("is_right_canonical", &MPS::isRightCanonical,
            py::arg("k"), py::arg("atol") = 1e-9)
        .def("dense_state_vector", &MPS::denseStateVector,
            "Full d^N state vector — small N only.");

    py::class_<MutualInformation>(cuda, "MutualInformation",
        "Static MI / entropy helpers acting on MPS and torch density "
        "matrices.")
        .def_static("von_neumann_entropy",
            &MutualInformation::vonNeumannEntropy,
            py::arg("rho"), py::arg("tol") = 1e-12)
        .def_static("edge_length",
            &MutualInformation::edgeLength,
            py::arg("I"), py::arg("epsilon") = 1e-6)
        .def_static("site_site",
            &MutualInformation::siteSite,
            py::arg("psi"), py::arg("i"), py::arg("j"))
        .def_static("all_pairs",
            &MutualInformation::allPairs,
            py::arg("psi"),
            "Symmetric (N, N) MI matrix as a real torch.Tensor.");
}
