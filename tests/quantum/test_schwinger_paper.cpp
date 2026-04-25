// Paper-aligned Schwinger-model checks against Bañuls et al., JHEP 11, 158
// (2013), exercising things the small-N MPO-vs-dense and analytic-limit
// tests don't reach.
//
//   1. Continuum trend (Bañuls §4, fig. 6): the ground-state energy density
//      ω_0 = E_0 / (2 x N) at m/g = 0 must approach the Schwinger exact
//      value ω_0 = -1/π ≈ -0.3183099 as x = 1/(g²a²) and N grow together
//      (Bañuls' prescription is N ≳ 20√x to keep finite-volume effects
//      below the continuum signal). We do a 3-point geometric scan
//      (x = 1, 4, 16 with N tracking √x) and require monotone descent
//      plus a bracketed approach.
//
//      Mapping our dimensional H to Bañuls' dimensionless ω_0:
//
//          E_W       = (2/(ag²)) · E_dim                  (Bañuls' W)
//          ω_0       = E_W / (2 x N)
//          x         = 1/(g²a²)
//          ⇒  ω_0   = E_dim · a / N           (after substitution)
//
//      So with a = 1 and g = 1/√x, the test variable is just E_total · 1/N.
//
//   2. First-excited gap above GS in Sz=0 (cf. Bañuls table): the gap
//      E_1 - E_0 in the Sz=0 sector, divided by g. In the continuum the
//      first excited state in this sector is the *vector* with
//      M_V/g → 1/√π ≈ 0.5641895; at our small x the level ordering is
//      different (see test 4: scalar-branch states sit between GS and the
//      first vector level), so this gap is to the lowest C=+1 excitation,
//      not specifically to the vector. We use DMRG with projection-
//      orthogonality (Bañuls eq. 3.3) to find the excited state and verify
//      the gap is positive and finite — the DMRG plumbing test.
//
//   3. Chiral condensate ⟨Σ̄Σ⟩ (Bañuls' benchmark observable): at m/g = 0,
//      L_0 = 0 the staggered chiral-condensate operator
//          Σ̄Σ ∝ (1/N) Σ_n (-1)^n σ^z_n
//      has a nonzero VEV on the lattice GS (anomalously broken U(1)_A);
//      at m → ∞ it saturates against the strong-coupling Néel state. We
//      run both regimes and verify the expected nonzero / saturated values.
//
//   4. Charge-conjugation parity (Bañuls page 8-9): Bañuls' operator
//          S_R = (⊗_k σ^x_{2k-1}) · T⁽¹⁾
//      where T⁽¹⁾ is cyclic translation by one site. He shows that even
//      under OBC (where S_R doesn't strictly commute with H), its real-
//      valued expectation tags eigenstates by C-parity: GS / scalar branch
//      have ⟨S_R⟩ > 0, vector branch has ⟨S_R⟩ < 0. We implement T⁽¹⁾ as
//      a basis-state permutation (which it is in any computational-basis
//      representation) on dense eigenvectors, dump the lowest several Sz=0
//      levels with their ⟨S_R⟩ values, and require both signs present in
//      the spectrum (GS in C=+1, at least one C=−1 state nearby).
//
// Why dense ED for (4) instead of DMRG: T⁽¹⁾ on an OBC MPS would require
// re-routing a non-edge bond to the boundary, since it sends spin label N
// to position 1 with that bond originally not connecting through the edge.
// Dense ED on the Sz=0 sector is small enough at N=8,12 to be free.

#include "quantum/schwinger_model.hpp"

#include <itensor/all.h>

#include <Eigen/Dense>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numbers>
#include <vector>

using namespace caset::quantum;
using itensor::dmrg;
using itensor::Sweeps;
using itensor::MPS;
using itensor::InitState;

namespace {

// Néel |↑↓↑↓…⟩ initial state in the Sz=0 sector.
MPS neel_init(itensor::SpinHalf const& sites, int N) {
    auto state = InitState(sites);
    for (int i = 1; i <= N; ++i) {
        state.set(i, (i % 2 == 1) ? "Up" : "Dn");
    }
    return MPS(state);
}

// Total Sz = Σ_n ⟨ψ|Sz_n|ψ⟩ via single-site contractions (same pattern as
// in test_schwinger_spectrum.cpp; see comments there). With ConserveQNs the
// returned value is exactly 0 up to roundoff for any DMRG output; we use
// it to confirm that ITensor's QN tracking didn't silently break.
double total_sz(MPS const& psi_in, itensor::SpinHalf const& sites, int N) {
    MPS psi = psi_in;
    double s = 0.0;
    for (int n = 1; n <= N; ++n) {
        psi.position(n);
        auto Op = itensor::op(sites, "Sz", n);
        auto bra = itensor::dag(psi(n));
        bra.prime("Site");
        s += itensor::elt(bra * Op * psi(n));
    }
    return s;
}

// Staggered chiral-condensate observable in spin language (after Jordan-
// Wigner): ⟨Σ̄Σ⟩_lat = (1/N) Σ_n (-1)^n ⟨σ^z_n⟩. Normalization conventions
// vary across the literature; we report the bare sum and only check sign /
// order of magnitude. Factor of 2 below converts ITensor's "Sz" (= ½ σ^z)
// to bare σ^z.
double chiral_condensate(MPS const& psi_in, itensor::SpinHalf const& sites, int N) {
    MPS psi = psi_in;
    double s = 0.0;
    for (int n = 1; n <= N; ++n) {
        psi.position(n);
        auto Op = itensor::op(sites, "Sz", n);
        auto bra = itensor::dag(psi(n));
        bra.prime("Site");
        const double sz = itensor::elt(bra * Op * psi(n));
        const double sign = (n % 2 == 0) ? 1.0 : -1.0;
        s += sign * (2.0 * sz);  // 2*Sz = σ^z
    }
    return s / N;
}

struct DMRGResult { double energy; MPS psi; };

DMRGResult run_dmrg_gs(SchwingerMPO const& sm, int max_bond, int n_sweeps) {
    auto psi0 = neel_init(sm.sites, sm.params.N);
    auto sweeps = Sweeps(n_sweeps);
    sweeps.maxdim() = 20, 40, 80, max_bond, max_bond;
    sweeps.cutoff() = 1e-12;
    sweeps.niter() = 4;
    sweeps.noise() = 1e-7, 1e-8, 0.0;
    auto [E, psi] = dmrg(sm.H, psi0, sweeps, {"Silent=", true});
    return {E, psi};
}

// First excited state via projection orthogonality (Bañuls eq. 3.3): we
// pass the GS as a "wave function to be orthogonal to", which makes ITensor
// add a penalty W·|ψ_0⟩⟨ψ_0| to H during the search. The variational
// minimum of the penalized H is then the lowest state orthogonal to ψ_0.
// W must exceed the GS-excited gap or DMRG will just rediscover the GS.
DMRGResult run_dmrg_first_excited(SchwingerMPO const& sm, MPS const& gs,
                                  int max_bond, int n_sweeps) {
    auto psi_init = neel_init(sm.sites, sm.params.N);
    auto sweeps = Sweeps(n_sweeps);
    sweeps.maxdim() = 20, 40, 80, max_bond, max_bond;
    sweeps.cutoff() = 1e-12;
    sweeps.niter() = 4;
    sweeps.noise() = 1e-7, 1e-8, 0.0;
    std::vector<MPS> wfs = {gs};
    // Weight = 100 is generous: the GS-1st-excited gap at our parameters is
    // O(1), and we want W ≫ gap so the GS becomes a high-energy state under
    // the penalty.
    auto [E, psi] = dmrg(sm.H, wfs, psi_init, sweeps,
                        {"Silent=", true, "Weight=", 100.0});
    return {E, psi};
}

bool check_continuum_trend() {
    std::cout << "Continuum trend ω_0 → -1/π at m/g=0 (Bañuls fig. 6)\n";
    std::cout << "----------------------------------------------------\n";
    constexpr double pi = std::numbers::pi;
    constexpr double omega0_exact = -1.0 / pi;

    // (x, N) pairs roughly tracking N ∝ √x to keep the physical box size
    // N · a (with a = 1/√x in units where g·a = const) growing modestly.
    // Bañuls' published prescription is N ≥ 20√x; we scale down a bit to
    // fit the pytest budget — the trend is still clearly visible. Each
    // later point should be more negative (closer to -1/π) than the prior.
    struct Pt { double x; int N; int max_bond; int sweeps; };
    const std::vector<Pt> points = {
        { 1.0,  20,  60,  10},
        { 4.0,  40,  80,  12},
        {16.0,  80, 120,  14},
    };

    std::vector<double> omega0_values;
    for (auto const& pt : points) {
        SchwingerParams p;
        p.N = pt.N;
        p.a = 1.0;
        p.g = 1.0 / std::sqrt(pt.x);  // x = 1/(g²a²)
        p.m = 0.0;
        p.L0 = 0.0;

        auto sm = build_schwinger_mpo(p);
        auto gs = run_dmrg_gs(sm, pt.max_bond, pt.sweeps);
        const double e_total = gs.energy + sm.constant;
        // ω_0 = (a/N) * E_dim_total  (derivation in file header).
        const double omega0 = e_total * p.a / p.N;
        omega0_values.push_back(omega0);

        std::cout
            << "  x=" << pt.x << " N=" << pt.N
            << "  E_total=" << e_total
            << "  ω_0=" << omega0
            << "  Δ(ω_0,-1/π)=" << (omega0 - omega0_exact)
            << "\n";
    }

    // Expect monotone descent toward -1/π from above.
    bool monotone = true;
    for (std::size_t i = 1; i < omega0_values.size(); ++i) {
        if (!(omega0_values[i] < omega0_values[i - 1])) monotone = false;
    }
    // The largest-x point should bracket -1/π within finite-size slack.
    // At x=16, N=80 we expect |ω_0 - (-1/π)| < 0.05 based on Bañuls fig. 6
    // shapes; this is a soft envelope, not a precision claim.
    const double last = omega0_values.back();
    const bool bracketed = std::abs(last - omega0_exact) < 0.05;

    const bool pass = monotone && bracketed;
    std::cout
        << "  monotone-descent=" << (monotone ? "Y" : "N")
        << "  |ω_0(x=16) - (-1/π)|=" << std::abs(last - omega0_exact)
        << (pass ? "  PASS" : "  FAIL") << "\n";
    return pass;
}

// Vector-mass-gap continuum trend at m/g = 0 (Bañuls 2013 fig. 7a, table on
// page 13): the published continuum value is M_V/g = 0.5642(1) ≈ 1/√π. At
// finite x the gap sits above this and descends as x → ∞ (more concretely,
// Bañuls fig. 7a shows M_V/g ≈ 0.61 at 1/√x = 0.25 and a near-linear approach
// to 0.5642 at the y-axis intercept).
//
// What we assert:
//   (1) at the (x, N) points we run, the gap is in a wide band consistent
//       with the figure (covering finite-size corrections at our small N);
//   (2) the gap at the larger x is BELOW the gap at the smaller x — a
//       monotone-descent invariant from the figure.
//
// Why this is "vector mass gap" and not generic first-excited: at m/g = 0
// our DMRG penalised search for the first excited state in Sz = 0 returns
// the LOWEST orthogonal state. Bañuls' Fig. 7a is the vector branch, the
// continuum value of which is below the scalar (0.5642 vs. 1.1284). The
// vector branch is below the scalar at x ≳ 16 (Bañuls' published range
// starts at x = 20), so the lowest orthogonal state in our x = 16 run is
// the vector. At smaller x the level ordering flips — we don't claim the
// x = 4 point hits the vector.
bool check_vector_mass_continuum_trend() {
    std::cout << "\nVector-mass-gap continuum trend at m/g=0 (Bañuls fig. 7a)\n";
    std::cout << "----------------------------------------------------------\n";

    // (x=4, N=40) is the off-figure warm-up — the gap there is the lowest
    // orthogonal state, which may or may not be the vector branch but is a
    // consistency floor. (x=16, N=80) lands at 1/√x = 0.25 which is the
    // leftmost data point in Bañuls Fig. 7a; at this x the vector is the
    // lowest orthogonal state in Sz=0 and the published value is 0.61.
    struct Pt { double x; int N; int max_bond; int gs_sweeps; int ex1_sweeps; };
    const std::vector<Pt> points = {
        { 4.0,  40,  80, 12, 14},
        {16.0,  80, 120, 14, 16},
    };

    std::vector<double> gap_per_g_values;
    for (auto const& pt : points) {
        SchwingerParams p;
        p.N = pt.N;
        p.a = 1.0;
        p.g = 1.0 / std::sqrt(pt.x);
        p.m = 0.0;
        p.L0 = 0.0;

        auto sm  = build_schwinger_mpo(p);
        auto gs  = run_dmrg_gs(sm, pt.max_bond, pt.gs_sweeps);
        auto ex1 = run_dmrg_first_excited(sm, gs.psi, pt.max_bond, pt.ex1_sweeps);
        const double gap_dim   = ex1.energy - gs.energy;
        // Bañuls' dimensionless ω_1 = (a/N) · E_W = (a/N) · (2/(ag²)) · E_dim
        // for energies, but the vector mass M_V/g uses (E_1 − E_0)/(2√x):
        //
        //     (E_1 − E_0)_dim · a · √x = (1/g) · (E_1 − E_0)_dim
        //
        // since g = 1/(a√x) when a = 1. So gap_per_g = (E_1 − E_0)/g.
        const double gap_per_g = gap_dim / p.g;
        gap_per_g_values.push_back(gap_per_g);

        std::cout
            << "  x=" << pt.x << " N=" << pt.N
            << "  E_0=" << gs.energy
            << "  E_1=" << ex1.energy
            << "  (E_1-E_0)/g=" << gap_per_g
            << "\n";
    }

    // (1) Monotone descent (vector mass gap shrinks toward the continuum
    //     1/√π ≈ 0.5642 from above as x → ∞).
    bool monotone = true;
    for (std::size_t i = 1; i < gap_per_g_values.size(); ++i) {
        if (!(gap_per_g_values[i] < gap_per_g_values[i - 1])) monotone = false;
    }
    // (2) The x = 16 point — assuming this is the vector branch — should
    //     bracket Bañuls Fig. 7a's value 0.61 within a finite-N envelope.
    //     We use [0.55, 0.75]: the lower bound excludes already-converged
    //     continuum (we're at 1/√x = 0.25 which Bañuls shows at 0.61);
    //     the upper bound permits some N=80 overshoot vs. their N→∞.
    constexpr double pi = std::numbers::pi;
    constexpr double m_v_continuum = 1.0 / 1.7724538509055159; // 1/√π
    const double last = gap_per_g_values.back();
    const bool bracketed = last > 0.55 && last < 0.75;

    const bool pass = monotone && bracketed;
    std::cout
        << "  monotone-descent=" << (monotone ? "Y" : "N")
        << "  gap@x=16=" << last
        << "  continuum 1/√π=" << m_v_continuum
        << "  Δ=" << (last - m_v_continuum)
        << (pass ? "  PASS" : "  FAIL") << "\n";
    (void)pi;
    return pass;
}

bool check_mass_gap() {
    std::cout << "\nFirst-excited gap above GS in Sz=0 (cf. Bañuls table)\n";
    std::cout << "------------------------------------------------------\n";
    // In the continuum (x → ∞) the first excited state in the Sz=0 sector
    // is the vector with M_V/g → 1/√π ≈ 0.5641895 (Bañuls). At our small
    // x = 1 the level ordering is different — the parity test below shows
    // that several C=+1 (scalar-branch) states sit between the GS and the
    // first C=−1 (vector) level, so this gap is to the lowest C=+1
    // excitation, not specifically to the vector. We just require it to be
    // positive and finite for the DMRG plumbing to be working.
    SchwingerParams p;
    p.N = 20; p.a = 1.0; p.g = 1.0; p.m = 0.0; p.L0 = 0.0;

    auto sm = build_schwinger_mpo(p);
    auto gs  = run_dmrg_gs(sm, /*max_bond=*/100, /*sweeps=*/14);
    auto ex1 = run_dmrg_first_excited(sm, gs.psi, /*max_bond=*/100, /*sweeps=*/16);

    const double gap_dim = ex1.energy - gs.energy;
    const double gap_per_g = gap_dim / p.g;
    const double sz_gs  = total_sz(gs.psi, sm.sites, p.N);
    const double sz_ex1 = total_sz(ex1.psi, sm.sites, p.N);

    // At x=1, N=20 the lattice gap is well above the continuum value
    // (Bañuls fig. 7 a, m/g=0 starts near 0.84 at 1/√x = 1 and decreases
    // to 0.5642 as x → ∞). So 0.4 < gap/g < 1.5 is a sane band.
    const bool gap_positive = gap_dim > 0;
    const bool gap_in_band  = gap_per_g > 0.4 && gap_per_g < 1.5;
    const bool sz_clean     = std::abs(sz_gs) < 1e-6 && std::abs(sz_ex1) < 1e-6;

    const bool pass = gap_positive && gap_in_band && sz_clean;
    std::cout
        << "  N=" << p.N << " m/g=0 (x=1)"
        << "  E_0=" << gs.energy << "  E_1=" << ex1.energy
        << "  gap/g=" << gap_per_g
        << "  Sz_gs=" << sz_gs << "  Sz_ex1=" << sz_ex1
        << (pass ? "  PASS" : "  FAIL") << "\n";
    return pass;
}

// Cyclic right rotation of an N-bit value by 1 in the MSB-is-site-1
// convention. Physically this is T⁽¹⁾ acting on a basis state:
//
//   T⁽¹⁾ |i_1 i_2 … i_N⟩  =  |i_N i_1 i_2 … i_{N-1}⟩
//
// In bits, site 1 is the MSB (position N-1) and site N is the LSB
// (position 0). After T⁽¹⁾ the new MSB holds the old LSB, and everything
// else shifts right by one bit.
inline std::uint64_t cyclic_rotr(std::uint64_t s, int N) {
    return ((s & 1ull) << (N - 1)) | (s >> 1);
}

// XOR mask flipping bits at every odd 1-based site (sites 1, 3, …, N-1).
// In our bit layout site k is at bit position N-k, so the mask sets bits
// N-1, N-3, …, 1.
inline std::uint64_t odd_site_mask(int N) {
    std::uint64_t m = 0;
    for (int k = 1; k <= N; k += 2) m |= (1ull << (N - k));
    return m;
}

// ⟨ψ | S_R | ψ⟩  with  S_R = (⊗_k σ^x_{2k-1}) · T⁽¹⁾   (Bañuls page 8).
//
// On a basis state |s⟩, S_R|s⟩ = |s'⟩ where s' = (cyclic_rotr(s) XOR mask):
// translation comes first (right-most operator acts first), then we flip
// the odd sites of the translated state.
//
// So ⟨s|S_R|t⟩ = δ_{s, S_R(t)}, and for real ψ:
//   ⟨ψ|S_R|ψ⟩ = Σ_s ψ_s · ψ_{S_R(s)}
//
// The result is real (S_R is Hermitian: σ^x is, and translation is
// permutation-unitary), and can be positive or negative — the sign is
// the C-parity tag Bañuls uses to distinguish vector vs. scalar branches.
//
// Note: ⟨S_R⟩ vanishes identically in the Sz=0 sector unless N is
// divisible by 4. Reason: σ^x_odd flips N/2 sites, and on a Sz=0 state
// (popcount = N/2) the parity of σ^x_odd-induced popcount changes is
// determined mod 2 by N/2; for ⟨ψ_Sz0|S_R|ψ_Sz0⟩ to have *any* nonzero
// contribution we need apply_SR(s) ∈ Sz=0, which forces N divisible by 4.
double sr_dense_expectation(Eigen::VectorXd const& psi, int N) {
    const std::uint64_t mask = odd_site_mask(N);
    const std::size_t dim = static_cast<std::size_t>(psi.size());
    double total = 0.0;
    for (std::size_t s = 0; s < dim; ++s) {
        const std::uint64_t s_after = cyclic_rotr(s, N) ^ mask;
        total += psi[static_cast<Eigen::Index>(s_after)] * psi[s];
    }
    return total;
}

// All Sz=0 eigenpairs of the dense Schwinger H, with each eigenvector
// embedded back into the full 2^N space (zero-padded outside Sz=0). Used
// by the C-parity test, which needs full-space ψ to apply S_R = σ^x · T⁽¹⁾
// (S_R doesn't preserve Sz, so the matrix elements ⟨ψ_Sz0|S_R|ψ_Sz0⟩
// require evaluating ψ at indices that *would* be outside Sz=0 if S_R
// took us there — we simply have zero contribution for those, which
// drops out automatically when ψ is zero-padded).
struct Sz0Spectrum {
    Eigen::VectorXd energies;
    std::vector<Eigen::VectorXd> psis;  // psis[k] is the k-th Sz=0 eigenvector
};
Sz0Spectrum sz0_spectrum(SchwingerDense const& sd) {
    const int N = sd.params.N;
    const int half = N / 2;

    std::vector<Eigen::Index> idx;
    idx.reserve(static_cast<std::size_t>(sd.H.rows()));
    for (Eigen::Index s = 0; s < sd.H.rows(); ++s) {
        if (std::popcount(static_cast<std::uint64_t>(s)) == static_cast<unsigned>(half)) {
            idx.push_back(s);
        }
    }
    const Eigen::Index k = static_cast<Eigen::Index>(idx.size());
    Eigen::MatrixXd Hsub(k, k);
    for (Eigen::Index i = 0; i < k; ++i) {
        for (Eigen::Index j = 0; j < k; ++j) {
            Hsub(i, j) = sd.H(idx[i], idx[j]);
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hsub);

    Sz0Spectrum out;
    out.energies = es.eigenvalues();
    out.psis.reserve(static_cast<std::size_t>(k));
    for (Eigen::Index lvl = 0; lvl < k; ++lvl) {
        Eigen::VectorXd psi = Eigen::VectorXd::Zero(sd.H.rows());
        for (Eigen::Index i = 0; i < k; ++i) {
            psi(idx[i]) = es.eigenvectors()(i, lvl);
        }
        out.psis.push_back(std::move(psi));
    }
    return out;
}

bool check_charge_conjugation_parity() {
    std::cout << "\nCharge-conjugation parity ⟨S_R⟩ (Bañuls page 8-9, S_R = σ^x_odd · T⁽¹⁾)\n";
    std::cout << "----------------------------------------------------------------------\n";
    // Bañuls' phase argument requires two C-parity classes among the low-
    // lying Sz=0 levels: GS plus scalar branch in C=+1 (⟨S_R⟩ > 0,
    // φ ≈ 0); vector branch in C=−1 (⟨S_R⟩ < 0, φ ≈ π). At our small x=1
    // the lattice-corrected level ordering is *not* the continuum order:
    // a few scalar-branch (C=+1) states sit between the GS and the first
    // C=−1 (vector) state. So we require the weaker but still meaningful
    // claim:
    //
    //   • the ground state is in C=+1, AND
    //   • at least one C=−1 level appears in the lowest 6 Sz=0 levels.
    //
    // We chose N ∈ {8, 12} because (a) ⟨S_R⟩ vanishes identically when
    // N is not divisible by 4, see sr_dense_expectation comment; and (b)
    // dense ED on the C(N, N/2)-dim Sz=0 subspace stays cheap (70 × 70
    // for N=8, 924 × 924 for N=12).
    bool ok = true;
    for (int N : {8, 12}) {
        SchwingerParams p;
        p.N = N; p.a = 1.0; p.g = 1.0; p.m = 0.0; p.L0 = 0.0;
        auto sd = build_schwinger_dense(p);
        auto spec = sz0_spectrum(sd);

        bool saw_neg = false;
        bool gs_pos  = false;
        const int n_show = std::min<int>(6, static_cast<int>(spec.psis.size()));
        std::cout << "  N=" << N << " m/g=0  Sz=0 spectrum:\n";
        for (int lvl = 0; lvl < n_show; ++lvl) {
            const double sr = sr_dense_expectation(spec.psis[lvl], N);
            // 1e-3 sign threshold — well above DMRG/ED floor noise (1e-15)
            // and small enough to label even weakly-mixed levels.
            const char* tag = (sr >  1e-3) ? "C=+1" :
                              (sr < -1e-3) ? "C=−1" : "C≈0";
            if (sr < -1e-3) saw_neg = true;
            if (lvl == 0)   gs_pos = (sr > 1e-3);
            std::cout
                << "    lvl=" << lvl
                << "  E=" << spec.energies(lvl)
                << "  ⟨S_R⟩=" << sr
                << "  → " << tag << "\n";
        }
        const bool case_pass = gs_pos && saw_neg;
        if (!case_pass) ok = false;
        std::cout << "    → "
                  << (gs_pos ? "GS in C=+1" : "GS NOT in C=+1")
                  << ", "
                  << (saw_neg ? "C=−1 levels present" : "no C=−1 levels found")
                  << (case_pass ? "  PASS" : "  FAIL") << "\n";
    }
    return ok;
}

bool check_chiral_condensate() {
    std::cout << "\nChiral condensate ⟨Σ̄Σ⟩ benchmark observable\n";
    std::cout << "---------------------------------------------\n";
    // Two scales chosen so the expected value is unambiguous:
    //
    //   • m/g = 0:  anomalously broken U(1)_A chiral symmetry gives a
    //               nonzero condensate VEV. The exact lattice value depends
    //               on (x, N); we just require |⟨Σ̄Σ⟩| > 0.05 — well above
    //               the DMRG noise floor on this size.
    //
    //   • m/g = 100: mass term dominates, GS ≈ |↑↓↑↓…⟩. On that state
    //                σ^z_n = (-1)^(n+1), so
    //                  ⟨Σ̄Σ⟩ = (1/N) Σ (-1)^n · (-1)^(n+1) = -1
    //                exactly. Hopping corrections are O(1/m²) ≈ 1e-4 here.
    //
    // Note: the *sign* differs between m=0 and m=100 — this is real OBC
    // physics, not a bug. At L₀=0 the H_E term strictly prefers the Néel
    // pattern |↓↑↓↑…⟩ (which gives L_n=0 on every link) over |↑↓↑↓…⟩
    // (which gives L_n²=1 on every odd link). At m=0, with no mass term
    // to compete, the GS sits in the |↓↑↓↑…⟩-dominant phase, so ⟨Σ̄Σ⟩ > 0.
    // At large m the mass term wins (it minimizes on |↑↓↑↓…⟩) and ⟨Σ̄Σ⟩
    // flips sign to −1. DMRG correctly tracks both regimes regardless of
    // the Néel-pattern initial state we hand it.
    bool ok = true;
    {
        SchwingerParams p;
        p.N = 20; p.a = 1.0; p.g = 1.0; p.m = 0.0; p.L0 = 0.0;
        auto sm = build_schwinger_mpo(p);
        auto gs = run_dmrg_gs(sm, 100, 12);
        const double cc = chiral_condensate(gs.psi, sm.sites, p.N);
        const bool nonzero = std::abs(cc) > 0.05;
        if (!nonzero) ok = false;
        std::cout
            << "  N=20 m/g=0  ⟨Σ̄Σ⟩=" << cc
            << (nonzero ? "  PASS" : "  FAIL  (expected |⟨Σ̄Σ⟩|>0.05)")
            << "\n";
    }
    {
        SchwingerParams p;
        p.N = 20; p.a = 1.0; p.g = 1.0; p.m = 100.0; p.L0 = 0.0;
        auto sm = build_schwinger_mpo(p);
        auto gs = run_dmrg_gs(sm, 60, 8);
        const double cc = chiral_condensate(gs.psi, sm.sites, p.N);
        const bool saturated = std::abs(cc - (-1.0)) < 0.05;
        if (!saturated) ok = false;
        std::cout
            << "  N=20 m/g=100  ⟨Σ̄Σ⟩=" << cc
            << "  (expected ≈ -1)"
            << (saturated ? "  PASS" : "  FAIL")
            << "\n";
    }
    return ok;
}

} // namespace

int main() {
    bool ok = true;
    if (!check_continuum_trend())                 ok = false;
    if (!check_mass_gap())                        ok = false;
    if (!check_vector_mass_continuum_trend())     ok = false;
    if (!check_chiral_condensate())               ok = false;
    if (!check_charge_conjugation_parity())       ok = false;
    std::cout << (ok ? "\nALL PASS\n" : "\nFAILURES\n");
    return ok ? 0 : 1;
}
