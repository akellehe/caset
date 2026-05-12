// Cross-check the TDVP integrator against full unitary evolution
// e^{-iHt} on the dense Hilbert space, for small N. This catches any
// integrator / sweep / quench bug that the heavy-quark flux-tube test
// might miss (because it only verifies static plateaus, not actual
// time evolution).
//
// Pipeline:
//   1. Build the dense Schwinger H for N ≤ 8.
//   2. Diagonalize H (Eigen::SelfAdjointEigenSolver). Pick the GS in
//      the Sz=0 sector to match the DMRG path.
//   3. Apply σ⁻_{i0} σ⁺_{i0+d} on the dense GS state vector to mimic
//      the quench from quench.cpp.
//   4. For each evolution time t ∈ {dt, 2dt, …, T}, compute
//        |ψ(t)⟩ = U exp(-i E t) U† |ψ(0)⟩
//      using the eigendecomposition. This is exact unitary evolution.
//   5. From the dense state, compute ⟨σ^z_n⟩(t) for every site, and
//      then ⟨L_n⟩(t) via the same closed form as tdvp_runner.cpp.
//   6. Run SchwingerQuench::evolve() through the C++ pipeline at the same
//      parameters and compare snapshot-by-snapshot to the dense
//      reference.
//
// The agreement bound is set by the TDVP Trotter / SVD truncation
// error, not by the dense reference (which is exact). For small N
// with bondDim sufficient to be exact, agreement should be ~1e-6 over
// the full evolution.

#include "quantum/quench.hpp"
#include "quantum/schwinger_model.hpp"
#include "quantum/tdvp_runner.hpp"

#include <Eigen/Dense>

#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tessera::quantum;

namespace {

using Cplx   = std::complex<double>;
using VecCx  = Eigen::VectorXcd;
using VecRe  = Eigen::VectorXd;
using MatRe  = Eigen::MatrixXd;

// σ^z eigenvalue on basis state s at 1-based site n. Same MSB-first bit
// layout as SchwingerHamiltonian::denseMatrix.
inline double sigma_z(std::uint64_t s, int n, int N) {
    return ((s >> (N - n)) & 1ull) == 0 ? +1.0 : -1.0;
}

// Apply σ⁻_{i0} σ⁺_{i0+d} to a dense state vector. σ⁻ kills Dn (bit=1),
// σ⁺ kills Up (bit=0) — applying both to the heavy-quark vacuum for
// (i0 odd, d odd, i0+d even) yields a flux-tube state.
VecCx apply_quench_dense(VecCx const& psi, int N, int i0, int d) {
    const std::uint64_t bit_i0 = 1ull << (N - i0);
    const std::uint64_t bit_id = 1ull << (N - (i0 + d));
    VecCx out = VecCx::Zero(psi.size());
    for (Eigen::Index s = 0; s < psi.size(); ++s) {
        // σ⁻_{i0}: needs bit_i0 = 0 (Up), maps to bit_i0 = 1 (Dn).
        if (((s >> (N - i0))      & 1ull) != 0) continue;
        // σ⁺_{i0+d}: needs bit_id = 1 (Dn), maps to bit_id = 0 (Up).
        if (((s >> (N - (i0 + d))) & 1ull) != 1) continue;
        const std::uint64_t s_after =
            (static_cast<std::uint64_t>(s) | bit_i0) & ~bit_id;
        out(static_cast<Eigen::Index>(s_after)) =
            psi(static_cast<Eigen::Index>(s));
    }
    out.normalize();
    return out;
}

// Build dense H, diagonalize, return (eigenvalues, eigenvectors,
// Sz=0 sector ground-state vector embedded in full 2^N space).
struct DenseDecomp {
    VecRe eigs;
    MatRe vecs;
    VecCx gs_sz0;
};
DenseDecomp dense_decomp_with_gs(SchwingerParams const& p) {
    auto sd = SchwingerHamiltonian{p}.denseMatrix();
    Eigen::SelfAdjointEigenSolver<MatRe> es(sd.H);

    // Find the lowest-energy eigenvector with Sz_total = 0
    // (popcount = N/2). Walk eigenvectors from the lowest energy
    // and pick the first one with non-trivial weight on the Sz=0
    // subspace — at our problem sizes this is just the lowest one.
    const int N = p.N;
    const int half = N / 2;

    DenseDecomp out;
    out.eigs = es.eigenvalues();
    out.vecs = es.eigenvectors();
    out.gs_sz0 = VecCx::Zero(sd.H.rows());

    for (int lvl = 0; lvl < out.vecs.cols(); ++lvl) {
        // Compute weight on Sz=0 sector.
        double weight = 0.0;
        for (Eigen::Index s = 0; s < sd.H.rows(); ++s) {
            if (__builtin_popcountll(static_cast<std::uint64_t>(s))
                == static_cast<unsigned>(half)) {
                weight += out.vecs(s, lvl) * out.vecs(s, lvl);
            }
        }
        if (weight > 0.999) {
            for (Eigen::Index s = 0; s < sd.H.rows(); ++s) {
                out.gs_sz0(s) = out.vecs(s, lvl);
            }
            break;
        }
    }
    return out;
}

// Time-evolve a dense state by t using the precomputed eigendecomp.
//   ψ(t) = U exp(-i t E) U† ψ(0)
VecCx evolve_dense(VecCx const& psi0, DenseDecomp const& dd, double t) {
    // U† ψ_0 (real because ψ_0 may have come complex; H eigenvectors
    // are real for our real-symmetric H).
    const VecCx coeffs = dd.vecs.adjoint() * psi0;
    VecCx coeffs_t = coeffs;
    for (Eigen::Index k = 0; k < coeffs.size(); ++k) {
        coeffs_t(k) *= std::exp(Cplx(0.0, -dd.eigs(k) * t));
    }
    return dd.vecs * coeffs_t;
}

// ⟨σ^z_n⟩ on a dense complex state.
std::vector<double> sigma_z_profile_dense(VecCx const& psi, int N) {
    std::vector<double> out(static_cast<std::size_t>(N), 0.0);
    for (Eigen::Index s = 0; s < psi.size(); ++s) {
        const double prob = std::norm(psi(s));
        if (prob == 0.0) continue;
        for (int n = 1; n <= N; ++n) {
            out[static_cast<std::size_t>(n - 1)] +=
                prob * sigma_z(static_cast<std::uint64_t>(s), n, N);
        }
    }
    return out;
}

std::vector<double> L_profile_from_sz(std::vector<double> const& sz,
                                      double L0) {
    const int N = static_cast<int>(sz.size());
    std::vector<double> L(static_cast<std::size_t>(N - 1), 0.0);
    double cum = 0.0;
    for (int n = 1; n <= N - 1; ++n) {
        cum += sz[static_cast<std::size_t>(n - 1)];
        const double c_n = L0 + ((n % 2 == 0) ? 0.0 : -0.5);
        L[static_cast<std::size_t>(n - 1)] = c_n - 0.5 * cum;
    }
    return L;
}

bool case_test(int N, int i0, int d, double m,
               double T, double dt, int snapshotEvery,
               double tol_profile) {
    std::cout << "\nN=" << N << " m=" << m
              << " i0=" << i0 << " d=" << d
              << " T=" << T << " dt=" << dt << "\n";

    SchwingerParams p;
    p.N = N; p.a = 1.0; p.g = 1.0; p.m = m; p.L0 = 0.0;

    // Dense reference path
    auto dd = dense_decomp_with_gs(p);
    auto psi_dense = apply_quench_dense(dd.gs_sz0, N, i0, d);
    if (std::abs(psi_dense.norm() - 1.0) > 1e-10) {
        std::cout << "  FAIL: dense quench produced norm "
                  << psi_dense.norm() << "\n";
        return false;
    }

    // TDVP path through the production runner
    TDVPConfig cfg;
    cfg.N = N; cfg.a = p.a; cfg.g = p.g; cfg.m = p.m; cfg.L0 = p.L0;
    cfg.dmrgMaxBondDim = 1 << N;   // generously above max possible
    cfg.dmrgNSweeps = 14;
    cfg.dmrgCutoff = 1e-14;
    cfg.i0 = i0; cfg.d = d;
    cfg.dt = dt; cfg.T = T;
    cfg.maxBondDim = 1 << (N / 2);
    cfg.cutoff = 1e-12;
    cfg.krylovDim = 14;
    cfg.snapshotEvery = snapshotEvery;
    cfg.quiet = true;
    cfg.conserveQns = true;

    auto result = SchwingerQuench{cfg}.evolve();

    // Compare each TDVP snapshot to the dense evolution at the same time.
    bool ok = true;
    double max_dz = 0.0, max_dL = 0.0;
    for (auto const& snap : result.snapshots) {
        VecCx psi_t = evolve_dense(psi_dense, dd, snap.time);
        auto sz_dense = sigma_z_profile_dense(psi_t, N);
        auto L_dense  = L_profile_from_sz(sz_dense, p.L0);

        for (int n = 1; n <= N; ++n) {
            const double dZ = std::abs(
                snap.zProfile[static_cast<std::size_t>(n - 1)]
                - sz_dense[static_cast<std::size_t>(n - 1)]);
            if (dZ > max_dz) max_dz = dZ;
            if (dZ > tol_profile) ok = false;
        }
        for (int n = 1; n <= N - 1; ++n) {
            const double dL = std::abs(
                snap.lProfile[static_cast<std::size_t>(n - 1)]
                - L_dense[static_cast<std::size_t>(n - 1)]);
            if (dL > max_dL) max_dL = dL;
            if (dL > tol_profile) ok = false;
        }
    }
    std::cout << "  snapshots=" << result.snapshots.size()
              << "  max|ΔZ|=" << max_dz
              << "  max|ΔL|=" << max_dL
              << "  " << (ok ? "PASS" : "FAIL")
              << " (tol " << tol_profile << ")\n";
    return ok;
}

} // namespace

int main() {
    std::cout << std::setprecision(8);
    std::cout << "TDVP vs dense-ED real-time evolution\n";
    std::cout << "------------------------------------\n";
    bool ok = true;

    // Heavy-quark flux tube — same physics as test_tdvp_string.cpp but
    // dense-ED ground-truthed at N=6.
    ok &= case_test(/*N=*/6, /*i0=*/1, /*d=*/3, /*m=*/20.0,
                    /*T=*/2.0, /*dt=*/0.05, /*snap=*/4,
                    /*tol=*/1e-4);

    // Light-quark regime — string breaks, ⟨L_n⟩ profile evolves
    // non-trivially. Tighter tolerance because TDVP truncation is
    // negligible at this size.
    ok &= case_test(/*N=*/6, /*i0=*/1, /*d=*/3, /*m=*/0.5,
                    /*T=*/1.0, /*dt=*/0.05, /*snap=*/4,
                    /*tol=*/1e-4);

    // Massless — strongest test (most mixing, largest entanglement growth).
    ok &= case_test(/*N=*/6, /*i0=*/1, /*d=*/3, /*m=*/0.0,
                    /*T=*/1.0, /*dt=*/0.05, /*snap=*/4,
                    /*tol=*/1e-4);

    // Larger N — confirms scaling.
    ok &= case_test(/*N=*/8, /*i0=*/1, /*d=*/3, /*m=*/2.0,
                    /*T=*/1.0, /*dt=*/0.05, /*snap=*/4,
                    /*tol=*/1e-4);

    std::cout << (ok ? "\nALL PASS\n" : "\nFAILURES\n");
    return ok ? 0 : 1;
}
