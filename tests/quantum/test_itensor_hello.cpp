// Phase 0 acceptance: confirm the ITensor build links and runs by reproducing
// a small Heisenberg ground state through AutoMPO + DMRG. We compare against
// a dense Eigen diagonalization of the same Hamiltonian so the test stays
// self-contained — no external reference numbers to maintain. If this fails
// the bug is almost certainly in the ITensor build wiring (CMake, BLAS/LAPACK
// linkage, config.h generation), not in the Schwinger-specific code, so this
// test runs first to isolate.

#include <itensor/all.h>

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>

using namespace itensor;

namespace {

// Spin-1/2 Heisenberg AFM on N sites with open boundary conditions:
//
//   H = Σ_{n=1..N-1} S_n · S_{n+1}
//     = Σ_{n=1..N-1} [ (1/2) (S+_n S-_{n+1} + S-_n S+_{n+1}) + Sz_n Sz_{n+1} ]
//
// We build the dense matrix directly on the 2^N computational basis with
// the same bit convention as the Schwinger dense builder: site 1 is the
// most-significant bit, |Up⟩ = bit 0 (σ^z=+1), |Dn⟩ = bit 1 (σ^z=−1).
Eigen::MatrixXd heisenberg_dense(int N) {
    const std::size_t dim = 1ull << N;
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(dim),
                                              static_cast<Eigen::Index>(dim));
    auto bit = [&](std::size_t s, int n) {
        return static_cast<int>((s >> (N - n)) & 1ull);
    };
    auto flip = [&](std::size_t s, int n) {
        return s ^ (1ull << (N - n));
    };
    for (std::size_t s = 0; s < dim; ++s) {
        double diag = 0.0;
        for (int n = 1; n <= N - 1; ++n) {
            const double zn = bit(s, n) == 0 ? +1.0 : -1.0;
            const double zm = bit(s, n + 1) == 0 ? +1.0 : -1.0;
            diag += 0.25 * zn * zm;  // Sz Sz = (1/4) σ^z σ^z
            if (bit(s, n) != bit(s, n + 1)) {
                std::size_t s2 = flip(flip(s, n), n + 1);
                // (1/2) S+S- + (1/2) S-S+ contributes element 1/2 between
                // the swapped configurations.
                H(static_cast<Eigen::Index>(s2),
                  static_cast<Eigen::Index>(s)) += 0.5;
            }
        }
        H(static_cast<Eigen::Index>(s),
          static_cast<Eigen::Index>(s)) += diag;
    }
    return H;
}

// Same Heisenberg Hamiltonian via ITensor's AutoMPO + DMRG. We use:
//   • ConserveQNs=true     — total Sz is a good QN; pinning the sector via
//                            a Néel initial state keeps DMRG focused.
//   • Néel |↑↓↑↓…⟩ initial — total Sz = 0, the sector that contains the
//                            global GS for AFM Heisenberg with even N.
//   • Sweep schedule       — bond-dim ramp 20→40→max_bond → max_bond, with
//                            a small noise term in early sweeps to avoid
//                            local minima. Cutoff 1e-12 is well below
//                            double-precision noise on 256-state problems.
double dmrg_heisenberg(int N, int max_bond, int n_sweeps) {
    auto sites = SpinHalf(N, {"ConserveQNs=", true});
    auto ampo = AutoMPO(sites);
    for (int n = 1; n <= N - 1; ++n) {
        ampo += 0.5, "S+", n, "S-", n + 1;
        ampo += 0.5, "S-", n, "S+", n + 1;
        ampo +=      "Sz", n, "Sz", n + 1;
    }
    auto H = toMPO(ampo);

    auto state = InitState(sites);
    for (int i = 1; i <= N; ++i) {
        state.set(i, (i % 2 == 1) ? "Up" : "Dn");  // Néel, total Sz = 0
    }
    auto psi0 = MPS(state);

    auto sweeps = Sweeps(n_sweeps);
    sweeps.maxdim() = 20, 40, max_bond, max_bond, max_bond;
    sweeps.cutoff() = 1e-12;
    sweeps.niter() = 4;
    sweeps.noise() = 1e-7, 1e-8, 0.0;

    // Silent=true suppresses ITensor's default per-sweep diagnostic output;
    // we only care about the final energy here.
    auto [energy, psi] = dmrg(H, psi0, sweeps, {"Silent=", true});
    return energy;
}

} // namespace

int main() {
    // N=8 keeps the dense matrix at 256×256 (instant Eigen ED) while still
    // exercising a nontrivial DMRG run with truncation pressure on the
    // central bond.
    constexpr int N = 8;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(heisenberg_dense(N));
    const double e_dense = es.eigenvalues()(0);
    const double e_dmrg  = dmrg_heisenberg(N, /*max_bond=*/64, /*n_sweeps=*/8);

    const double diff = std::abs(e_dense - e_dmrg);
    std::cout << "Heisenberg N=" << N
              << "  dense=" << e_dense
              << "  dmrg="  << e_dmrg
              << "  |Δ|="   << diff << "\n";

    // 1e-6 is the bar in PLAN.md §5 Phase 0; we typically see ~1e-14
    // because DMRG converges to machine precision on a problem this small.
    constexpr double tol = 1e-6;
    if (diff > tol) {
        std::cerr << "FAIL: DMRG-vs-dense disagree by more than " << tol << "\n";
        return 1;
    }
    std::cout << "PASS\n";
    return 0;
}
