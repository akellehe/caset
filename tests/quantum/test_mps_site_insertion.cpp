// Spike: validate the MPS site-insertion + 3-site-gate mechanics that the
// InteractionSimulation needs.
//
// An interaction event creates a fresh AB worldline — a new site spliced
// into the MPS — and then acts a 3-body unitary on (X, AB, Y). This test
// proves the two operations in isolation:
//
//   1. insertSite: splice a product-state |Up> site between two adjacent
//      MPS sites. Norm, the inserted site's purity, and the pre-existing
//      correlations across the insertion point must all be preserved.
//
//   2. apply3SiteGate: contract three adjacent site tensors, multiply by
//      a 3-qubit gate, and SVD back into the MPS. A non-local gate must
//      generate mutual information; the identity gate must be a no-op.
//
// If this fails, the bug is in the ITensor splice/SVD plumbing, not in
// the simulation logic — so it runs standalone.

#include <itensor/all.h>

#include "quantum/mutual_information.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <iostream>

using namespace itensor;
using tessera::quantum::MutualInformation;

namespace {

// Splice a fresh product-state |Up> site between MPS positions p and p+1.
// Returns the (L+1)-site MPS; the new site lands at position p+1 and
// carries the supplied site index. Pre-existing tensors keep their own
// site indices — no SiteSet rebuild.
MPS insertSite(MPS psi, int p, Index const& sAB) {
    const int L = length(psi);
    if (p < 1 || p >= L)
        throw std::invalid_argument("insertSite: p must be in [1, L-1]");

    psi.position(p);

    const Index lL = leftLinkIndex(psi, p);    // null when p == 1
    const Index sX = siteIndex(psi, p);
    const Index sY = siteIndex(psi, p + 1);
    const Index lR = rightLinkIndex(psi, p + 1);  // null when p+1 == L

    // 2-site wavefunction ⊗ a product-state |Up> at the new site.
    ITensor ab(sAB);
    ab.set(sAB = 1, 1.0);
    ITensor wf3 = psi(p) * psi(p + 1) * ab;  // {lL?, sX, sAB, sY, lR?}

    // Split wf3 → U1 . U2 . tail over two fresh bonds.
    IndexSet leftInds = lL ? IndexSet(lL, sX) : IndexSet(sX);
    auto [U1, S1, V1] = svd(wf3, leftInds,
                            {"Cutoff=", 1e-15, "LeftTags=", "Link,l=ins0"});
    ITensor rest = S1 * V1;
    const Index link1 = commonIndex(U1, S1);

    auto [U2, S2, V2] = svd(rest, IndexSet(link1, sAB),
                            {"Cutoff=", 1e-15, "LeftTags=", "Link,l=ins1"});
    ITensor abTensor = U2;
    ITensor yTensor = S2 * V2;  // orthogonality centre

    MPS out(L + 1);
    for (int k = 1; k < p; ++k) out.set(k, psi(k));
    out.set(p, U1);
    out.set(p + 1, abTensor);
    out.set(p + 2, yTensor);
    for (int k = p + 2; k <= L; ++k) out.set(k + 1, psi(k));
    // The tensors are in canonical form (left-orth up to p+1, OC at p+2)
    // but set() does not update the MPS's ortho-limit bookkeeping —
    // re-establish it explicitly.
    out.leftLim(p + 1);
    out.rightLim(p + 3);
    return out;
}

// Apply a 3-qubit gate to adjacent MPS sites (p, p+1, p+2). gate is a
// rank-6 ITensor over (prime(s_p), prime(s_{p+1}), prime(s_{p+2}),
// s_p, s_{p+1}, s_{p+2}).
void apply3SiteGate(MPS& psi, int p, ITensor const& gate) {
    psi.position(p);
    const Index lL = leftLinkIndex(psi, p);
    const Index sP = siteIndex(psi, p);
    const Index sQ = siteIndex(psi, p + 1);

    ITensor wf = psi(p) * psi(p + 1) * psi(p + 2);
    wf *= gate;
    wf.noPrime();

    IndexSet leftInds = lL ? IndexSet(lL, sP) : IndexSet(sP);
    auto [U1, S1, V1] = svd(wf, leftInds,
                            {"Cutoff=", 1e-14, "LeftTags=", "Link,l=g0"});
    ITensor rest = S1 * V1;
    const Index link1 = commonIndex(U1, S1);
    auto [U2, S2, V2] = svd(rest, IndexSet(link1, sQ),
                            {"Cutoff=", 1e-14, "LeftTags=", "Link,l=g1"});
    psi.set(p, U1);
    psi.set(p + 1, U2);
    psi.set(p + 2, S2 * V2);
    // OC is at p+2, sites <= p+1 left-orthogonal — refresh the bookkeeping.
    psi.leftLim(p + 1);
    psi.rightLim(p + 3);
}

bool approx(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) < tol;
}

// Dense rank-6 identity over (a',b',c', a,b,c). Built by setting the
// diagonal explicitly so the storage is dense — a product of delta
// tensors is diagonal storage, which does not contract against itself.
ITensor denseIdentity3(Index const& a, Index const& b, Index const& c) {
    ITensor id(prime(a), prime(b), prime(c), a, b, c);
    for (int i = 1; i <= 2; ++i)
        for (int j = 1; j <= 2; ++j)
            for (int k = 1; k <= 2; ++k)
                id.set(prime(a) = i, prime(b) = j, prime(c) = k,
                       a = i, b = j, c = k, 1.0);
    return id;
}

} // namespace

int main() {
    const double LN2 = std::log(2.0);
    int failures = 0;

    // ── Build a 4-site MPS with a Bell pair on the adjacent sites (2,3),
    //    so the correlation straddles the insertion point. Product state
    //    + a Bell-prep gate via svdBond keeps every tensor dense. ───────
    auto sites = SpinHalf(4, {"ConserveQNs=", false});
    auto init = InitState(sites);
    for (int i = 1; i <= 4; ++i) init.set(i, "Up");
    MPS psi(init);
    {
        psi.position(2);
        const Index s2 = siteIndex(psi, 2), s3 = siteIndex(psi, 3);
        const double s2inv = 1.0 / std::sqrt(2.0);
        // U|↑↑> = (|↑↑> + |↓↓>)/√2, completed to a unitary.
        ITensor G(prime(s2), prime(s3), s2, s3);
        G.set(prime(s2) = 1, prime(s3) = 1, s2 = 1, s3 = 1, s2inv);
        G.set(prime(s2) = 2, prime(s3) = 2, s2 = 1, s3 = 1, s2inv);
        G.set(prime(s2) = 1, prime(s3) = 2, s2 = 1, s3 = 2, s2inv);
        G.set(prime(s2) = 2, prime(s3) = 1, s2 = 1, s3 = 2, s2inv);
        G.set(prime(s2) = 1, prime(s3) = 1, s2 = 2, s3 = 1, s2inv);
        G.set(prime(s2) = 2, prime(s3) = 2, s2 = 2, s3 = 1, -s2inv);
        G.set(prime(s2) = 1, prime(s3) = 2, s2 = 2, s3 = 2, s2inv);
        G.set(prime(s2) = 2, prime(s3) = 1, s2 = 2, s3 = 2, -s2inv);
        ITensor wf = psi(2) * psi(3);
        wf *= G;
        wf.noPrime();
        psi.svdBond(2, wf, Fromleft);
    }

    const double normBefore = norm(psi);
    const double mi23Before = MutualInformation::siteSite(psi, 2, 3);
    std::cout << "[before] L=" << length(psi)
              << "  norm=" << normBefore
              << "  I(2:3)=" << mi23Before << std::endl;

    // ── 1. Insert a fresh |Up> site between positions 2 and 3 ──────────
    Index sAB(2, "Site,AB");
    MPS psi5 = insertSite(psi, 2, sAB);

    const double normAfter = norm(psi5);
    const double mi24After = MutualInformation::siteSite(psi5, 2, 4);
    auto rhoAB = MutualInformation::oneSiteReducedDensity(psi5, 3);
    const double sAB_entropy = MutualInformation::vonNeumannEntropy(rhoAB);

    std::cout << "[insert] L=" << length(psi5)
              << "  norm=" << normAfter
              << "  I(2:4)=" << mi24After
              << "  S(AB)=" << sAB_entropy
              << "  rhoAB[Up,Up]=" << std::real(rhoAB(0, 0)) << std::endl;

    if (length(psi5) != 5) {
        std::cerr << "FAIL: expected L=5 after insertion\n";
        ++failures;
    }
    if (!approx(normAfter, 1.0)) {
        std::cerr << "FAIL: norm not preserved by insertion\n";
        ++failures;
    }
    if (!approx(sAB_entropy, 0.0) || !approx(std::real(rhoAB(0, 0)), 1.0)) {
        std::cerr << "FAIL: inserted site is not a pure |Up>\n";
        ++failures;
    }
    if (!approx(mi24After, mi23Before)) {
        std::cerr << "FAIL: insertion changed the (2,3)->(2,4) correlation "
                  << "(" << mi23Before << " -> " << mi24After << ")\n";
        ++failures;
    }

    // ── 2. Identity 3-site gate is a no-op ────────────────────────────
    {
        MPS phi = psi5;
        const Index a = siteIndex(phi, 2), b = siteIndex(phi, 3),
                    c = siteIndex(phi, 4);
        apply3SiteGate(phi, 2, denseIdentity3(a, b, c));
        const double miId = MutualInformation::siteSite(phi, 2, 4);
        std::cout << "[identity] norm=" << norm(phi)
                  << "  I(2:4)=" << miId << std::endl;
        if (!approx(norm(phi), 1.0) || !approx(miId, mi24After)) {
            std::cerr << "FAIL: identity 3-site gate was not a no-op\n";
            ++failures;
        }
    }

    // ── 3. A genuine 3-site gate must create mutual information ───────
    {
        MPS phi = psi5;
        const Index a = siteIndex(phi, 2), b = siteIndex(phi, 3),
                    c = siteIndex(phi, 4);
        // Dense rank-6 gate: Hadamard on AB(b), then CNOT AB->X(a),
        // identity on Y(c). This unambiguously entangles the fresh AB
        // site with X regardless of the X-Y Bell state it starts in.
        const double s2inv = 1.0 / std::sqrt(2.0);
        auto H = [&](int bo, int bi) {  // <bo|H|bi>
            return (bi == 1) ? s2inv : (bo == 1 ? s2inv : -s2inv);
        };
        ITensor gate(prime(a), prime(b), prime(c), a, b, c);
        for (int xi = 1; xi <= 2; ++xi)
            for (int bi = 1; bi <= 2; ++bi)
                for (int yi = 1; yi <= 2; ++yi)
                    for (int bo = 1; bo <= 2; ++bo) {
                        // after H, AB is in basis state bo; CNOT flips X
                        // when bo == 2 (Dn).
                        const int xo = (bo == 2) ? (3 - xi) : xi;
                        gate.set(prime(a) = xo, prime(b) = bo,
                                 prime(c) = yi, a = xi, b = bi, c = yi,
                                 H(bo, bi));
                    }
        apply3SiteGate(phi, 2, gate);

        const double miXAB = MutualInformation::siteSite(phi, 2, 3);
        const double miABY = MutualInformation::siteSite(phi, 3, 4);
        const double miXY = MutualInformation::siteSite(phi, 2, 4);
        std::cout << "[entangler] norm=" << norm(phi)
                  << "  I(X:AB)=" << miXAB
                  << "  I(AB:Y)=" << miABY
                  << "  I(X:Y)=" << miXY << std::endl;
        if (!approx(norm(phi), 1.0)) {
            std::cerr << "FAIL: entangler did not preserve the norm\n";
            ++failures;
        }
        if (miXAB < 0.1) {
            std::cerr << "FAIL: entangler produced no X-AB mutual "
                         "information\n";
            ++failures;
        }
        (void)LN2;
    }

    if (failures == 0) {
        std::cout << "PASS\n";
        return 0;
    }
    std::cerr << failures << " check(s) failed\n";
    return 1;
}
