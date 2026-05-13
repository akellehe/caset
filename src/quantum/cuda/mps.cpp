// MPS implementation. See include/quantum/cuda/mps.hpp for the
// architectural overview and conventions.

#include "quantum/cuda/mps.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tessera::quantum::cuda {

namespace {

torch::Dtype realDtypeFor(torch::Dtype complex_dtype) {
    return complex_dtype == torch::kComplexDouble
            ? torch::kFloat64
            : torch::kFloat32;
}

} // namespace

MPS::MPS(std::vector<torch::Tensor> tensors,
          std::optional<int64_t> oc)
    : tensors_(std::move(tensors)), oc_(oc) {
    if (tensors_.empty()) {
        throw std::invalid_argument(
            "MPS must have at least one site");
    }
    for (std::size_t k = 0; k < tensors_.size(); ++k) {
        if (tensors_[k].dim() != 3) {
            throw std::invalid_argument(
                "MPS site tensor must be rank-3 (left, phys, right)");
        }
    }
    for (std::size_t k = 0; k + 1 < tensors_.size(); ++k) {
        if (tensors_[k].size(2) != tensors_[k + 1].size(0)) {
            throw std::invalid_argument(
                "MPS bond mismatch between adjacent sites");
        }
    }
    if (tensors_.front().size(0) != 1 || tensors_.back().size(2) != 1) {
        throw std::invalid_argument(
            "MPS boundary bonds must have dimension 1");
    }
    auto d = tensors_.front().size(1);
    for (auto const& t : tensors_) {
        if (t.size(1) != d) {
            throw std::invalid_argument(
                "MPS physical dimension is inconsistent across sites");
        }
    }
}

// ─── Factories ───────────────────────────────────────────────────────

MPS MPS::productState(std::vector<torch::Tensor> site_states,
                        torch::Device device,
                        torch::Dtype dtype) {
    if (site_states.empty()) {
        throw std::invalid_argument(
            "productState requires at least one site");
    }
    std::vector<torch::Tensor> tensors;
    tensors.reserve(site_states.size());
    for (auto& vec : site_states) {
        auto v = vec.to(device).to(dtype);
        if (v.dim() != 1) {
            throw std::invalid_argument(
                "product-state ket must be 1-d");
        }
        tensors.push_back(v.reshape({1, v.size(0), 1}));
    }
    return MPS(std::move(tensors), 0);
}

MPS MPS::computationalBasis(std::vector<int64_t> bits,
                              int64_t d,
                              torch::Device device,
                              torch::Dtype dtype) {
    if (bits.empty()) {
        throw std::invalid_argument(
            "computationalBasis requires at least one bit");
    }
    auto opts = torch::TensorOptions().device(device).dtype(dtype);
    std::vector<torch::Tensor> states;
    states.reserve(bits.size());
    for (auto b : bits) {
        if (b < 0 || b >= d) {
            throw std::invalid_argument(
                "computationalBasis: bit out of [0, d)");
        }
        auto v = torch::zeros({d}, opts);
        v.index_put_({b},
                       c10::Scalar(c10::complex<double>(1.0, 0.0)));
        states.push_back(v);
    }
    return MPS::productState(std::move(states), device, dtype);
}

MPS MPS::bellChain(int64_t N_pairs,
                    torch::Device device,
                    torch::Dtype dtype) {
    if (N_pairs < 1) {
        throw std::invalid_argument("N_pairs must be >= 1");
    }
    auto real_dtype = realDtypeFor(dtype);
    auto opts_real = torch::TensorOptions()
                       .device(device).dtype(real_dtype);
    const double s2inv = 1.0 / std::sqrt(2.0);
    std::vector<torch::Tensor> tensors;
    tensors.reserve(static_cast<std::size_t>(2 * N_pairs));

    for (int64_t k = 0; k < N_pairs; ++k) {
        // First site of the k-th pair: identity in (phys, right_bond).
        auto re0 = torch::zeros({1, 2, 2}, opts_real);
        re0.index_put_({0, 0, 0}, 1.0);
        re0.index_put_({0, 1, 1}, 1.0);
        auto im0 = torch::zeros({1, 2, 2}, opts_real);
        tensors.push_back(torch::complex(re0, im0).to(dtype));

        // Second site of the k-th pair: 1/sqrt(2) on the diagonal,
        // absorbing the Bell norm into this site so the first site
        // stays left-canonical.
        auto re1 = torch::zeros({2, 2, 1}, opts_real);
        re1.index_put_({0, 0, 0}, s2inv);
        re1.index_put_({1, 1, 0}, s2inv);
        auto im1 = torch::zeros({2, 2, 1}, opts_real);
        tensors.push_back(torch::complex(re1, im1).to(dtype));
    }
    return MPS(std::move(tensors), 2 * N_pairs - 1);
}

MPS MPS::ghz(int64_t N,
              int64_t d,
              torch::Device device,
              torch::Dtype dtype) {
    if (N < 1) {
        throw std::invalid_argument("N must be >= 1");
    }
    if (d < 2) {
        throw std::invalid_argument("d must be >= 2 for GHZ");
    }
    auto real_dtype = realDtypeFor(dtype);
    auto opts_real = torch::TensorOptions()
                       .device(device).dtype(real_dtype);
    const double s2inv = 1.0 / std::sqrt(2.0);

    std::vector<torch::Tensor> tensors;
    tensors.reserve(static_cast<std::size_t>(N));

    // Site 0: branch selector with the GHZ 1/sqrt(2) absorbed here.
    auto re0 = torch::zeros({1, d, 2}, opts_real);
    re0.index_put_({0, 0, 0}, s2inv);
    re0.index_put_({0, 1, 1}, s2inv);
    tensors.push_back(torch::complex(re0,
                       torch::zeros_like(re0)).to(dtype));

    // Middle sites: branch-preserving identity on (phys, bond).
    for (int64_t k = 1; k + 1 < N; ++k) {
        auto rem = torch::zeros({2, d, 2}, opts_real);
        rem.index_put_({0, 0, 0}, 1.0);
        rem.index_put_({1, 1, 1}, 1.0);
        tensors.push_back(torch::complex(rem,
                           torch::zeros_like(rem)).to(dtype));
    }

    if (N > 1) {
        auto ren = torch::zeros({2, d, 1}, opts_real);
        ren.index_put_({0, 0, 0}, 1.0);
        ren.index_put_({1, 1, 0}, 1.0);
        tensors.push_back(torch::complex(ren,
                           torch::zeros_like(ren)).to(dtype));
    }
    return MPS(std::move(tensors), N - 1);
}

MPS MPS::random(int64_t N,
                 int64_t d,
                 int64_t max_chi,
                 torch::Device device,
                 torch::Dtype dtype,
                 std::optional<int64_t> seed) {
    if (N < 1 || d < 1 || max_chi < 1) {
        throw std::invalid_argument(
            "random: N, d, max_chi must all be >= 1");
    }
    std::vector<int64_t> bonds(static_cast<std::size_t>(N + 1));
    bonds[0] = 1;
    bonds[static_cast<std::size_t>(N)] = 1;
    for (int64_t k = 1; k < N; ++k) {
        int64_t left_max  = 1;
        int64_t right_max = 1;
        for (int64_t e = 0; e < k; ++e) {
            if (left_max > max_chi / d + 1) { left_max = max_chi; break; }
            left_max *= d;
        }
        for (int64_t e = 0; e < N - k; ++e) {
            if (right_max > max_chi / d + 1) { right_max = max_chi; break; }
            right_max *= d;
        }
        bonds[static_cast<std::size_t>(k)] =
            std::min({left_max, right_max, max_chi});
    }

    if (seed.has_value()) {
        torch::manual_seed(*seed);
    }

    auto real_dtype = realDtypeFor(dtype);
    auto opts_real = torch::TensorOptions()
                       .device(device).dtype(real_dtype);

    std::vector<torch::Tensor> tensors;
    tensors.reserve(static_cast<std::size_t>(N));
    for (int64_t k = 0; k < N; ++k) {
        auto re = torch::randn(
            {bonds[static_cast<std::size_t>(k)], d,
              bonds[static_cast<std::size_t>(k + 1)]}, opts_real);
        auto im = torch::randn(
            {bonds[static_cast<std::size_t>(k)], d,
              bonds[static_cast<std::size_t>(k + 1)]}, opts_real);
        tensors.push_back(torch::complex(re, im).to(dtype));
    }

    MPS mps(std::move(tensors), std::nullopt);
    mps.canonicalize(N - 1);
    mps.normalize();
    return mps;
}

// ─── Properties ──────────────────────────────────────────────────────

int64_t MPS::bondDim(int64_t k) const {
    if (k == -1) return 1;
    if (k < 0 || k >= length()) {
        throw std::out_of_range("MPS::bondDim: site out of range");
    }
    return tensors_[static_cast<std::size_t>(k)].size(2);
}

// ─── Canonical form ──────────────────────────────────────────────────

void MPS::canonicalize(int64_t oc) {
    auto N = length();
    if (oc < 0 || oc >= N) {
        throw std::out_of_range("canonicalize: oc out of [0, N)");
    }

    if (!oc_.has_value()) {
        for (int64_t k = 0; k < oc; ++k) leftCanonicalStep(k);
        for (int64_t k = N - 1; k > oc; --k) rightCanonicalStep(k);
    } else if (*oc_ < oc) {
        for (int64_t k = *oc_; k < oc; ++k) leftCanonicalStep(k);
    } else if (*oc_ > oc) {
        for (int64_t k = *oc_; k > oc; --k) rightCanonicalStep(k);
    }
    oc_ = oc;
}

void MPS::leftCanonicalStep(int64_t k) {
    auto const& A = tensors_[static_cast<std::size_t>(k)];
    auto chi_l = A.size(0);
    auto d     = A.size(1);
    auto chi_r = A.size(2);
    auto mat = A.reshape({chi_l * d, chi_r});

    auto qr = torch::linalg_qr(mat, "reduced");
    auto Q = std::get<0>(qr);
    auto R = std::get<1>(qr);
    auto new_chi = Q.size(1);

    tensors_[static_cast<std::size_t>(k)] = Q.reshape({chi_l, d, new_chi});
    if (k + 1 < length()) {
        auto const& B = tensors_[static_cast<std::size_t>(k + 1)];
        tensors_[static_cast<std::size_t>(k + 1)] =
            torch::einsum("ij,jpr->ipr", std::vector<at::Tensor>{R, B});
    }
}

void MPS::rightCanonicalStep(int64_t k) {
    auto const& A = tensors_[static_cast<std::size_t>(k)];
    auto chi_l = A.size(0);
    auto d     = A.size(1);
    auto chi_r = A.size(2);
    auto mat = A.reshape({chi_l, d * chi_r});

    // LQ via QR on the conjugate transpose.
    auto qr = torch::linalg_qr(mat.conj().transpose(0, 1), "reduced");
    auto Qh = std::get<0>(qr);
    auto Rh = std::get<1>(qr);
    auto L = Rh.conj().transpose(0, 1);
    auto Q = Qh.conj().transpose(0, 1);
    auto new_chi = Q.size(0);

    tensors_[static_cast<std::size_t>(k)] = Q.reshape({new_chi, d, chi_r});
    if (k - 1 >= 0) {
        auto const& B = tensors_[static_cast<std::size_t>(k - 1)];
        tensors_[static_cast<std::size_t>(k - 1)] =
            torch::einsum("lpi,ij->lpj", std::vector<at::Tensor>{B, L});
    }
}

// ─── Norm + reduced density matrices ─────────────────────────────────

double MPS::normSquared() {
    if (!oc_.has_value()) canonicalize(0);
    auto const& A = tensors_[static_cast<std::size_t>(*oc_)];
    return at::real(torch::einsum("lpr,lpr->",
                                std::vector<at::Tensor>{A, A.conj()}))
                .item<double>();
}

void MPS::normalize() {
    if (!oc_.has_value()) canonicalize(0);
    auto& A = tensors_[static_cast<std::size_t>(*oc_)];
    auto n_sq = at::real(
        torch::einsum("lpr,lpr->",
                        std::vector<at::Tensor>{A, A.conj()}));
    if (n_sq.item<double>() == 0.0) {
        throw std::runtime_error("MPS has zero norm");
    }
    auto scale = torch::sqrt(n_sq).to(A.scalar_type());
    A = A / scale;
}

torch::Tensor MPS::oneSiteReducedDensity(int64_t i) const {
    // DEBUG: const, fresh tensor.
    return torch::zeros({2, 2}, torch::TensorOptions().dtype(torch::kComplexDouble));
}

torch::Tensor MPS::twoSiteReducedDensity(int64_t i, int64_t j) {
    if (i == j) {
        throw std::invalid_argument(
            "twoSiteReducedDensity: require i != j; use "
            "oneSiteReducedDensity for the i == j case");
    }
    if (i > j) std::swap(i, j);

    auto N = length();
    if (i < 0 || j >= N) {
        throw std::out_of_range(
            "twoSiteReducedDensity: site out of [0, N)");
    }
    const auto d = physicalDim();

    canonicalize(i);

    // Site i: open phys on both ket and bra; left bond auto-traces via
    // the orth-canonical condition.
    auto const& Ai = tensors_[static_cast<std::size_t>(i)];
    auto T = torch::einsum("lpr,lqs->pqrs", std::vector<at::Tensor>{Ai, Ai.conj()});

    // Interior k in (i, j): site_k unprimed on both → auto-traces.
    for (int64_t k = i + 1; k < j; ++k) {
        auto const& Ak = tensors_[static_cast<std::size_t>(k)];
        T = torch::einsum("pqrs,rxt,sxu->pqtu",
                       std::vector<at::Tensor>{T, Ak, Ak.conj()});
    }

    // Site j: open phys on both ket and bra; right bond auto-traces via
    // the right-canonical condition (sites > i are right-canonical when
    // oc = i, so the right environment past j is identity).
    auto const& Aj = tensors_[static_cast<std::size_t>(j)];
    auto chi = Aj.size(2);
    auto eye_r = torch::eye(chi, Aj.options());
    auto rho4 = torch::einsum(
        "pqrs,rat,sbu,tu->pqab",
        std::vector<at::Tensor>{T, Aj, Aj.conj(), eye_r});
    auto rho = rho4.permute({0, 2, 1, 3})
                    .reshape({d * d, d * d});
    return 0.5 * (rho + rho.conj().transpose(0, 1));
}

// ─── Diagnostics ─────────────────────────────────────────────────────

bool MPS::isLeftCanonical(int64_t k, double atol) const {
    auto const& A = tensors_[static_cast<std::size_t>(k)];
    auto chi_l = A.size(0);
    auto d     = A.size(1);
    auto chi_r = A.size(2);
    auto mat = A.reshape({chi_l * d, chi_r});
    auto gram = mat.conj().transpose(0, 1).matmul(mat);
    auto eye = torch::eye(chi_r, A.options());
    return torch::allclose(gram, eye, /*rtol=*/1e-7, /*atol=*/atol);
}

bool MPS::isRightCanonical(int64_t k, double atol) const {
    auto const& A = tensors_[static_cast<std::size_t>(k)];
    auto chi_l = A.size(0);
    auto d     = A.size(1);
    auto chi_r = A.size(2);
    auto mat = A.reshape({chi_l, d * chi_r});
    auto gram = mat.matmul(mat.conj().transpose(0, 1));
    auto eye = torch::eye(chi_l, A.options());
    return torch::allclose(gram, eye, /*rtol=*/1e-7, /*atol=*/atol);
}

torch::Tensor MPS::denseStateVector() const {
    if (tensors_.empty()) {
        throw std::runtime_error("MPS is empty");
    }
    auto psi = tensors_[0];
    for (std::size_t k = 1; k < tensors_.size(); ++k) {
        auto last_axis = psi.dim() - 1;
        psi = torch::tensordot(psi, tensors_[k],
                                 /*dims_self=*/{last_axis},
                                 /*dims_other=*/{0});
    }
    int64_t total = 1;
    for (int64_t k = 0; k < length(); ++k) total *= physicalDim();
    return psi.reshape({total});
}

} // namespace tessera::quantum::cuda
