// Implementation of the Schwinger MPO and the dense reference Hamiltonian.
//
// ─── L_n² expansion (the only non-trivial algebra in this file) ───────────
//
// Define
//     T_n = (1/2) Σ_{k=1..n} σ^z_k    (operator part of L_n)
//     L_n = c_n − T_n                 (where c_n is a c-number, see header)
//
// Then
//     L_n² = c_n²  −  2 c_n T_n  +  T_n²
//
// Expand T_n²:
//     T_n² = (1/4) Σ_{j,k=1..n} σ^z_j σ^z_k
//          = (1/4) [ Σ_k (σ^z_k)²  +  2 Σ_{j<k≤n} σ^z_j σ^z_k ]
//          = n/4   +  (1/2) Σ_{j<k≤n} σ^z_j σ^z_k          [(σ^z)² = I]
//
// And cross-term:
//     −2 c_n T_n = − c_n Σ_{k=1..n} σ^z_k
//
// Sum L_n² over n = 1..N-1 (the H_E sum) and split into:
//
//   (a) c-number  ─ Σ_n c_n²  +  Σ_n (n/4)
//
//   (b) linear σ^z ─ −Σ_n c_n Σ_{k≤n} σ^z_k
//                  = − Σ_{k=1..N-1} σ^z_k · Σ_{n=k..N-1} c_n
//                  = − Σ_{k=1..N-1} A_k σ^z_k                where A_k = Σ_{n=k..N-1} c_n
//                  (note: σ^z_N never appears, since n only goes up to N-1)
//
//   (c) σ^z σ^z   ─ (1/2) Σ_{n=1..N-1} Σ_{j<k≤n} σ^z_j σ^z_k
//                  = (1/2) Σ_{1≤j<k≤N-1} σ^z_j σ^z_k · #{n : k ≤ n ≤ N-1}
//                  = (1/2) Σ_{1≤j<k≤N-1} (N−k) σ^z_j σ^z_k
//
// Multiplied by the H_E prefactor (g²a/2):
//
//   H_E = E_const + Σ_k coef_k σ^z_k + Σ_{j<k} coef_{jk} σ^z_j σ^z_k
//
// with
//   E_const   = (g²a/2) Σ_{n=1..N-1} (c_n² + n/4)
//   coef_k    = − (g²a/2) A_k          for k = 1..N-1
//   coef_{jk} = (g²a/4) (N−k)          for 1 ≤ j < k ≤ N-1
//
// The MPO goes through AutoMPO using "Sz" (= ½ σ^z) as the operator name,
// so the coefficients absorb factors of 2:
//   "Sz"_k          coefficient  =  2 · coef_k          = − (g²a) A_k
//   "Sz"_j "Sz"_k   coefficient  =  4 · coef_{jk}       =   (g²a) (N−k)
//
// The mass term is handled the same way: the dimensional H_m has coefficient
// (m/2)(-1)^n on σ^z_n, which becomes m · (-1)^n on "Sz"_n.
//
// And the hopping (1/(4a))(XX + YY) = (1/(2a))(σ⁺σ⁻ + σ⁻σ⁺) goes in directly
// since ITensor's "S+", "S-" are bare σ⁺, σ⁻ (no factor of 1/2).

#include "quantum/schwinger_model.hpp"

#include <itensor/mps/autompo.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace caset::quantum {

namespace {

// c_n = L₀ + ((-1)^n − 1)/4. Closed form: −1/2 if n is odd, 0 if even
// (plus the L₀ shift). This is the c-number part of L_n in 1-based indexing.
inline double c_n(int n, double L0) {
    return L0 + ((n % 2 == 0) ? 0.0 : -0.5);
}

// A_k = Σ_{n=k..N-1} c_n — the tail sum of c_n's that multiplies σ^z_k in
// the linear part of H_E (see derivation block at top of file).
inline double tail_sum_c(int k, int N, double L0) {
    double s = 0.0;
    for (int n = k; n <= N - 1; ++n) s += c_n(n, L0);
    return s;
}

} // namespace

double schwinger_energy_constant(SchwingerParams const& p) {
    // E_const = (g²a/2) Σ_{n=1..N-1} (c_n² + n/4)
    // First term tracks Σ c_n², second comes from the (σ^z)² = I diagonal of T_n².
    double s = 0.0;
    for (int n = 1; n <= p.N - 1; ++n) {
        const double c = c_n(n, p.L0);
        s += c * c + n / 4.0;
    }
    return 0.5 * p.g * p.g * p.a * s;
}

SchwingerMPO build_schwinger_mpo(SchwingerParams const& p, bool conserve_qns) {
    if (p.N < 2)   throw std::invalid_argument("SchwingerParams.N must be >= 2");
    if (p.a <= 0)  throw std::invalid_argument("SchwingerParams.a must be positive");
    // We deliberately do NOT reject g = 0: that's the free-Dirac limit (gauge
    // field decouples), the formulas below stay finite, and it gives us a
    // useful analytic-reference test point in test_schwinger_limits.cpp.

    using namespace itensor;

    // SpinHalf SiteSet. ConserveQNs=true makes the bond indices carry total
    // Sz quantum numbers — equivalent to U(1) total-charge conservation
    // after the Jordan-Wigner mapping. ITensor's SpinHalf operators in this
    // SiteSet are normalized as:
    //     "Sz"   = (1/2) σ^z   (eigenvalues ±1/2)
    //     "S+"   = σ⁺          (raises σ^z)
    //     "S-"   = σ⁻          (lowers σ^z)
    auto sites = SpinHalf(p.N, {"ConserveQNs=", conserve_qns});

    auto ampo = AutoMPO(sites);

    // ── Hopping: (1/(4a)) Σ (X X + Y Y) = (1/(2a)) Σ (σ⁺σ⁻ + σ⁻σ⁺)
    {
        const double t = 0.5 / p.a;
        for (int n = 1; n <= p.N - 1; ++n) {
            ampo += t, "S+", n, "S-", n + 1;
            ampo += t, "S-", n, "S+", n + 1;
        }
    }

    // ── Mass: (m/2) Σ_n (-1)^n σ^z_n  =  m Σ_n (-1)^n · "Sz"_n
    // Note 1-based (-1)^n: at n=1 the sign is −, at n=2 it's +, etc.
    if (p.m != 0.0) {
        for (int n = 1; n <= p.N; ++n) {
            const double sign = (n % 2 == 0) ? +1.0 : -1.0;
            ampo += p.m * sign, "Sz", n;
        }
    }

    // ── Electric-field operator part: see derivation at top of file.
    //
    //   H_E_op  =  − Σ_{k=1..N-1} (g²a A_k) · "Sz"_k                  (linear)
    //            + Σ_{1≤j<k≤N-1} (g²a (N−k)) · "Sz"_j "Sz"_k          (pair)
    //
    // (Coefficients have absorbed the factor of 2 / 4 from "Sz" = (1/2) σ^z.)
    {
        const double Eg = p.g * p.g * p.a;

        for (int k = 1; k <= p.N - 1; ++k) {
            const double Ak = tail_sum_c(k, p.N, p.L0);
            if (Ak != 0.0) ampo += -Eg * Ak, "Sz", k;
        }

        for (int k = 2; k <= p.N - 1; ++k) {
            const double w = Eg * static_cast<double>(p.N - k);
            if (w == 0.0) continue;
            for (int j = 1; j < k; ++j) {
                ampo += w, "Sz", j, "Sz", k;
            }
        }
    }

    SchwingerMPO out;
    out.params = p;
    out.sites = sites;
    out.H = toMPO(ampo);
    out.constant = schwinger_energy_constant(p);
    return out;
}

// ─── Dense reference Hamiltonian ──────────────────────────────────────────
//
// We build the same operator-valued H on the full 2^N computational basis,
// no symmetry reduction. Bit conventions:
//
//   • basis index s (an integer 0..2^N−1) decomposes into N bits.
//   • Site n (1-based) corresponds to bit at position (N − n), so site 1
//     is the most-significant bit. This makes the lexicographic order of
//     basis states (UU…UU, UU…UD, …) match increasing s.
//   • Bit value 0 ↔ |Up⟩ (σ^z = +1); bit value 1 ↔ |Dn⟩ (σ^z = −1). This
//     is opposite to "1 = present" in fermion-number language but is
//     internally consistent — only the spectrum matters for our tests.
//
// Hopping is the only non-diagonal term: σ⁺σ⁻ + σ⁻σ⁺ flips a (Up, Dn) or
// (Dn, Up) neighbouring pair to its swap. The matrix element in either
// direction equals 1/(2a). The other terms (mass + linear σ^z + pair σ^z σ^z)
// are all diagonal in the computational basis.

namespace {

inline int bit_at(std::size_t state, int n /*1-based*/, int N) {
    return static_cast<int>((state >> (N - n)) & 1ull);
}

inline std::size_t flip_bit(std::size_t state, int n, int N) {
    return state ^ (1ull << (N - n));
}

inline double sigma_z(std::size_t state, int n, int N) {
    return bit_at(state, n, N) == 0 ? +1.0 : -1.0;
}

} // namespace

SchwingerDense build_schwinger_dense(SchwingerParams const& p) {
    if (p.N < 2)   throw std::invalid_argument("SchwingerParams.N must be >= 2");
    // Hard cap at 16 to keep us from accidentally allocating a multi-GB
    // matrix; in practice the MPO/DMRG path takes over above N ~ 12.
    if (p.N > 16)  throw std::invalid_argument("SchwingerDense: N>16 is unreasonable");

    const std::size_t dim = 1ull << p.N;
    SchwingerDense out;
    out.params = p;
    out.H = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(dim),
                                  static_cast<Eigen::Index>(dim));
    out.constant = schwinger_energy_constant(p);

    // Precompute A_k tail sums once; they get hit O(N) times below per state.
    std::vector<double> A(p.N + 1, 0.0);
    for (int k = 1; k <= p.N - 1; ++k) A[k] = tail_sum_c(k, p.N, p.L0);

    const double t_hop = 0.5 / p.a;     // coefficient on σ⁺σ⁻ + σ⁻σ⁺
    const double Eg = p.g * p.g * p.a;  // common g²a prefactor

    // Walk every basis state. The outer loop hits each row of H twice
    // (once as the "from" state writing into H[s2, s], once as the "to"
    // state writing into H[s, s2'] for some other s2'); the matrix ends up
    // real-symmetric without an explicit symmetrize step.
    for (std::size_t s = 0; s < dim; ++s) {
        // ── Hopping (off-diagonal): flip an adjacent (Up, Dn) or (Dn, Up).
        for (int n = 1; n <= p.N - 1; ++n) {
            const int b_n = bit_at(s, n, p.N);
            const int b_m = bit_at(s, n + 1, p.N);
            if (b_n != b_m) {
                const std::size_t s2 = flip_bit(flip_bit(s, n, p.N), n + 1, p.N);
                out.H(static_cast<Eigen::Index>(s2),
                      static_cast<Eigen::Index>(s)) += t_hop;
            }
        }

        // ── All remaining terms are diagonal in the computational basis.
        double diag = 0.0;

        // Mass: (m/2) Σ (-1)^n σ^z_n
        if (p.m != 0.0) {
            for (int n = 1; n <= p.N; ++n) {
                const double sign = (n % 2 == 0) ? +1.0 : -1.0;
                diag += 0.5 * p.m * sign * sigma_z(s, n, p.N);
            }
        }

        // Linear σ^z piece of H_E: −(g²a/2) Σ_{k=1..N-1} A_k σ^z_k
        for (int k = 1; k <= p.N - 1; ++k) {
            diag += -0.5 * Eg * A[k] * sigma_z(s, k, p.N);
        }

        // Pairwise σ^z σ^z piece of H_E: (g²a/4) Σ_{j<k≤N-1} (N−k) σ^z_j σ^z_k
        for (int k = 2; k <= p.N - 1; ++k) {
            const double w = 0.25 * Eg * static_cast<double>(p.N - k);
            if (w == 0.0) continue;
            const double zk = sigma_z(s, k, p.N);
            for (int j = 1; j < k; ++j) {
                diag += w * sigma_z(s, j, p.N) * zk;
            }
        }

        out.H(static_cast<Eigen::Index>(s),
              static_cast<Eigen::Index>(s)) += diag;
    }

    return out;
}

} // namespace caset::quantum
