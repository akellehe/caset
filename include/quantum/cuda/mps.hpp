// Side-by-side libtorch mirror of the C++ ITensor MPS machinery.
//
// Storage: a std::vector<torch::Tensor>. Site k's tensor has shape
// (chi_left, d, chi_right). All tensors live on a single torch::Device
// at a single complex torch::Dtype (default kComplexDouble — matches
// the double-precision ITensor pipeline). The right bond of tensor k
// shares its dimension with the left bond of tensor k+1; the boundary
// bonds (left of site 0, right of site N-1) have dim 1.
//
// Site indices are 0-based throughout this namespace, in contrast with
// the 1-based indexing of the ITensor-backed C++ classes in the
// sibling include/quantum/*.hpp headers. Translate at the boundary
// when cross-validating.

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace tessera::quantum::cuda {

class MPS {
public:
    explicit MPS(std::vector<torch::Tensor> tensors,
                  std::optional<int64_t> oc = std::nullopt);

    // ── Factories ─────────────────────────────────────────────────────

    // Product MPS from one site-state vector per site (each ket is a
    // 1-D tensor of length d). Result is in trivial canonical form
    // (bond dim 1 everywhere); orth centre set to 0 by convention.
    static MPS productState(std::vector<torch::Tensor> site_states,
                              torch::Device device,
                              torch::Dtype dtype);

    // Computational-basis product state: bits[k] selects |bits[k]> at
    // site k.
    static MPS computationalBasis(std::vector<int64_t> bits,
                                    int64_t d,
                                    torch::Device device,
                                    torch::Dtype dtype);

    // |Phi+>^{⊗N_pairs} on 2·N_pairs sites: sites (2k, 2k+1) form the
    // k-th Bell pair |Phi+> = (|00> + |11>)/sqrt(2). Output is left-
    // canonical, orth centre at the rightmost site.
    static MPS bellChain(int64_t N_pairs,
                          torch::Device device,
                          torch::Dtype dtype);

    // GHZ state (|0…0> + |1…1>)/sqrt(2) on N sites of physical dim d.
    // Bond dim 2 everywhere except the boundaries.
    static MPS ghz(int64_t N,
                    int64_t d,
                    torch::Device device,
                    torch::Dtype dtype);

    // Random complex MPS with bond dims capped at max_chi. Left-
    // canonical, unit-normalised on return.
    static MPS random(int64_t N,
                       int64_t d,
                       int64_t max_chi,
                       torch::Device device,
                       torch::Dtype dtype,
                       std::optional<int64_t> seed);

    // ── Properties ────────────────────────────────────────────────────

    int64_t length() const noexcept {
        return static_cast<int64_t>(tensors_.size());
    }
    torch::Device device() const {
        return tensors_.front().device();
    }
    torch::Dtype dtype() const {
        return tensors_.front().scalar_type();
    }
    int64_t physicalDim() const {
        return tensors_.front().size(1);
    }
    std::optional<int64_t> orthCentre() const noexcept { return oc_; }
    int64_t bondDim(int64_t k) const;
    std::vector<torch::Tensor> const& tensors() const noexcept {
        return tensors_;
    }

    // ── Canonical form ────────────────────────────────────────────────

    // Move the orthogonality centre to site oc. After this call, sites
    // [0, oc) are left-canonical and sites (oc, N) are right-canonical.
    void canonicalize(int64_t oc);

    // ── Norm + reduced density matrices ───────────────────────────────

    // <psi|psi> as a real double. Triggers canonicalize() if needed.
    double normSquared();

    // Rescale so <psi|psi> = 1. Mutates the tensor at the orth centre.
    void normalize();

    // rho_i = Tr_{!=i} |psi><psi| as a (d, d) complex tensor.
    torch::Tensor oneSiteReducedDensity(int64_t i) const;

    // rho_{ij} as a (d^2, d^2) complex tensor. Basis convention
    // matches the C++ MutualInformation::twoSiteReducedDensity pack:
    // row index = d * phys_i + phys_j (ket), col index = d * phys_i' +
    // phys_j' (bra). For d = 2 the basis is (|↑↑>, |↑↓>, |↓↑>, |↓↓>).
    //
    // Algorithm: transfer-matrix sweep through sites i..j. Intermediate
    // memory is O(d^2 * chi^2) regardless of |j - i| — the same fix as
    // src/quantum/mutual_information.cpp.
    torch::Tensor twoSiteReducedDensity(int64_t i, int64_t j);

    // ── Diagnostics ───────────────────────────────────────────────────

    bool isLeftCanonical(int64_t k, double atol = 1e-9) const;
    bool isRightCanonical(int64_t k, double atol = 1e-9) const;

    // Full d^N state vector via successive bond contractions. Memory
    // grows as d^N — use only for small N during cross-validation.
    torch::Tensor denseStateVector() const;

private:
    void leftCanonicalStep(int64_t k);
    void rightCanonicalStep(int64_t k);

    std::vector<torch::Tensor> tensors_;
    std::optional<int64_t> oc_;
};

} // namespace tessera::quantum::cuda
