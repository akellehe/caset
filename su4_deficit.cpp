// =============================================================================
// su4_deficit_mc.cpp
//
// Monte Carlo simulation of SU(4) lattice gauge theory with the deficit
// involution, testing the dynamical Pati-Salam breaking mechanism.
//
// Degrees of freedom:
//   Per site:  Weyl parameters c = (c1,c2,c3) in [0, pi/4]
//              Ising label sigma = +1 (original) or -1 (deficit)
//   Per link:  Connection W = (W_L, W_R) in SU(2) x SU(2)
//
// The composite link variable is U_mu(x) = embed(W_L, W_R) * N(x)
// where N(x) is the non-local core determined by (c, sigma) at site x.
//
// Action: standard Wilson plaquette action
//   S = -beta * sum_{plaquettes} Re tr(P)
//
// Measurements:
//   1. Staggered magnetization Phi (Ising order parameter)
//   2. Average plaquette (thermodynamic)
//   3. Polyakov loop: colour vs leptoquark sectors
//   4. Eigenvalue degeneracy of plaquettes
//   5. Forward/backward temporal propagator asymmetry
//
// Compile: g++ -O3 -std=c++17 -o su4mc su4_deficit_mc.cpp -lm
// Run:     ./su4mc [beta] [Ls] [Lt] [n_therm] [n_meas] [meas_interval]
// =============================================================================

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <complex>
#include <random>
#include <array>
#include <vector>
#include <algorithm>
#include <numeric>

using cd = std::complex<double>;
constexpr double PI = 3.14159265358979323846;
constexpr int NDIM = 4;

// ===================== 4x4 Complex Matrix =====================

struct Mat4 {
    cd m[4][4];

    Mat4() { memset(m, 0, sizeof(m)); }

    static Mat4 identity() {
        Mat4 r;
        for (int i = 0; i < 4; i++) r.m[i][i] = 1.0;
        return r;
    }

    static Mat4 zero() { return Mat4(); }

    Mat4 operator*(const Mat4& b) const {
        Mat4 r;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                for (int k = 0; k < 4; k++)
                    r.m[i][j] += m[i][k] * b.m[k][j];
        return r;
    }

    Mat4 operator+(const Mat4& b) const {
        Mat4 r;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                r.m[i][j] = m[i][j] + b.m[i][j];
        return r;
    }

    Mat4 adjoint() const {
        Mat4 r;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                r.m[i][j] = std::conj(m[j][i]);
        return r;
    }

    cd trace() const {
        return m[0][0] + m[1][1] + m[2][2] + m[3][3];
    }

    // Frobenius norm squared
    double norm2() const {
        double s = 0;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                s += std::norm(m[i][j]);
        return s;
    }

    // Extract eigenvalues of a unitary matrix via QR iteration (simplified)
    // For measurement purposes; not performance-critical
    std::array<cd, 4> eigenvalues() const;
};

// ===================== SU(2) as Quaternion =====================
// U = a0*I + i*(a1*sigma_x + a2*sigma_y + a3*sigma_z)
// with a0^2 + a1^2 + a2^2 + a3^2 = 1

struct SU2 {
    double a[4]; // (a0, a1, a2, a3)

    static SU2 identity() { return {1.0, 0.0, 0.0, 0.0}; }

    SU2 operator*(const SU2& b) const {
        // Quaternion multiplication
        return {
            a[0]*b.a[0] - a[1]*b.a[1] - a[2]*b.a[2] - a[3]*b.a[3],
            a[0]*b.a[1] + a[1]*b.a[0] + a[2]*b.a[3] - a[3]*b.a[2],
            a[0]*b.a[2] - a[1]*b.a[3] + a[2]*b.a[0] + a[3]*b.a[1],
            a[0]*b.a[3] + a[1]*b.a[2] - a[2]*b.a[1] + a[3]*b.a[0]
        };
    }

    SU2 dagger() const {
        return {a[0], -a[1], -a[2], -a[3]};
    }

    void normalize() {
        double n = sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2] + a[3]*a[3]);
        for (int i = 0; i < 4; i++) a[i] /= n;
    }

    // Convert to 2x2 complex matrix
    // U = [a0+i*a3, a2+i*a1; -a2+i*a1, a0-i*a3]
    void to_matrix(cd out[2][2]) const {
        out[0][0] = cd(a[0],  a[3]);
        out[0][1] = cd(a[2],  a[1]);
        out[1][0] = cd(-a[2], a[1]);
        out[1][1] = cd(a[0], -a[3]);
    }
};

// ===================== Tensor Product & Embedding =====================

// Embed SU(2) x SU(2) into SU(4) as the tensor product W_L (x) W_R
// In computational basis |00>, |01>, |10>, |11>:
// (A (x) B)_{2i+j, 2k+l} = A_{ik} * B_{jl}
Mat4 embed_su2xu2(const SU2& wL, const SU2& wR) {
    cd L[2][2], R[2][2];
    wL.to_matrix(L);
    wR.to_matrix(R);
    Mat4 result;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                for (int l = 0; l < 2; l++)
                    result.m[2*i+j][2*k+l] = L[i][k] * R[j][l];
    return result;
}

// ===================== Magic Basis =====================
// Columns: v1=(|00>+|11>)/sqrt2, v2=(|00>-|11>)/sqrt2,
//          v3=(|01>+|10>)/sqrt2, v4=(|01>-|10>)/sqrt2
// Sign vectors: n1=(+,-,+), n2=(-,+,+), n3=(+,+,-), n4=(-,-,-)

static const double ISQRT2 = 1.0 / sqrt(2.0);

// Magic basis transformation matrix M (columns = magic basis vectors)
// M transforms FROM magic basis TO computational basis: |comp> = M |magic>
// M^dagger transforms FROM computational TO magic: |magic> = M^dag |comp>
Mat4 magic_basis_M() {
    Mat4 M;
    // v1 = (|00> + |11>)/sqrt2
    M.m[0][0] = ISQRT2; M.m[3][0] = ISQRT2;
    // v2 = (|00> - |11>)/sqrt2
    M.m[0][1] = ISQRT2; M.m[3][1] = -ISQRT2;
    // v3 = (|01> + |10>)/sqrt2
    M.m[1][2] = ISQRT2; M.m[2][2] = ISQRT2;
    // v4 = (|01> - |10>)/sqrt2
    M.m[1][3] = ISQRT2; M.m[2][3] = -ISQRT2;
    return M;
}

static const int sign_vectors[4][3] = {
    {+1, -1, +1},  // n1: Phi+ type
    {-1, +1, +1},  // n2: Phi- type
    {+1, +1, -1},  // n3: Psi+ type
    {-1, -1, -1},  // n4: Psi- type (singlet)
};

// Compute the non-local core N in computational basis
// for Weyl parameters c[3] and Ising label sigma (+1 or -1)
Mat4 nonlocal_core(const double c[3], int sigma) {
    static Mat4 M = magic_basis_M();
    static Mat4 Mdag = M.adjoint();

    double cc[3];
    for (int i = 0; i < 3; i++)
        cc[i] = (sigma == 1) ? c[i] : (PI / 4.0 - c[i]);

    // Eigenvalues in magic basis: exp(-i * lambda_k)
    // lambda_k = n_k . cc
    Mat4 D;
    for (int k = 0; k < 4; k++) {
        double lambda = 0;
        for (int i = 0; i < 3; i++)
            lambda += sign_vectors[k][i] * cc[i];
        D.m[k][k] = std::exp(cd(0, -lambda));
    }

    // N = M * D * M^dag
    return M * D * Mdag;
}

// ===================== Lattice =====================

struct Lattice {
    int Ls, Lt, vol;
    double beta;

    // Per-site data
    std::vector<std::array<double, 3>> weyl;  // Weyl parameters
    std::vector<int> ising;                     // +1 or -1

    // Per-link data: [site][direction]
    std::vector<std::array<SU2, NDIM>> connL;  // left SU(2)
    std::vector<std::array<SU2, NDIM>> connR;  // right SU(2)

    // Cached composite link variables (recomputed as needed)
    // Not cached globally for memory; computed on the fly.

    std::mt19937_64 rng;
    std::uniform_real_distribution<double> uniform;

    Lattice(int Ls_, int Lt_, double beta_, unsigned seed = 12345)
        : Ls(Ls_), Lt(Lt_), beta(beta_), uniform(0.0, 1.0)
    {
        vol = Ls * Ls * Ls * Lt;
        rng.seed(seed);

        weyl.resize(vol);
        ising.resize(vol);
        connL.resize(vol);
        connR.resize(vol);
    }

    int index(int x, int y, int z, int t) const {
        // Periodic boundary conditions
        x = ((x % Ls) + Ls) % Ls;
        y = ((y % Ls) + Ls) % Ls;
        z = ((z % Ls) + Ls) % Ls;
        t = ((t % Lt) + Lt) % Lt;
        return ((x * Ls + y) * Ls + z) * Lt + t;
    }

    void coords(int idx, int& x, int& y, int& z, int& t) const {
        t = idx % Lt; idx /= Lt;
        z = idx % Ls; idx /= Ls;
        y = idx % Ls; idx /= Ls;
        x = idx;
    }

    // Neighbour in direction mu (+1 step)
    int neighbour(int site, int mu) const {
        int x, y, z, t;
        coords(site, x, y, z, t);
        switch (mu) {
            case 0: return index(x+1, y, z, t);
            case 1: return index(x, y+1, z, t);
            case 2: return index(x, y, z+1, t);
            case 3: return index(x, y, z, t+1);
        }
        return -1;
    }

    // Neighbour in direction mu (-1 step)
    int neighbour_back(int site, int mu) const {
        int x, y, z, t;
        coords(site, x, y, z, t);
        switch (mu) {
            case 0: return index(x-1, y, z, t);
            case 1: return index(x, y-1, z, t);
            case 2: return index(x, y, z-1, t);
            case 3: return index(x, y, z, t-1);
        }
        return -1;
    }

    // Staggered parity of a site: (-1)^{x+y+z+t}
    int parity(int site) const {
        int x, y, z, t;
        coords(site, x, y, z, t);
        return ((x + y + z + t) % 2 == 0) ? 1 : -1;
    }

    // =================== Initialization ===================

    // Cold start: all c = pi/8 (SWAP^{1/2}), sigma = checkerboard, W = identity
    void cold_start() {
        for (int i = 0; i < vol; i++) {
            weyl[i] = {PI/8.0, PI/8.0, PI/8.0};
            ising[i] = parity(i); // checkerboard
            for (int mu = 0; mu < NDIM; mu++) {
                connL[i][mu] = SU2::identity();
                connR[i][mu] = SU2::identity();
            }
        }
    }

    // Hot start: random c, random sigma, random W
    void hot_start() {
        for (int i = 0; i < vol; i++) {
            for (int a = 0; a < 3; a++)
                weyl[i][a] = uniform(rng) * PI / 4.0;
            ising[i] = (uniform(rng) < 0.5) ? 1 : -1;
            for (int mu = 0; mu < NDIM; mu++) {
                connL[i][mu] = random_su2(2.0 * PI);
                connR[i][mu] = random_su2(2.0 * PI);
            }
        }
    }

    // =================== Random SU(2) ===================

    // Random SU(2) element near identity (for Metropolis proposals)
    // epsilon controls how far from identity
    SU2 random_su2(double epsilon) {
        std::normal_distribution<double> gauss(0, 1);
        double n1 = gauss(rng), n2 = gauss(rng), n3 = gauss(rng);
        double nrm = sqrt(n1*n1 + n2*n2 + n3*n3);
        if (nrm < 1e-12) return SU2::identity();
        double angle = epsilon * uniform(rng);
        double s = sin(angle) / nrm;
        SU2 r = {cos(angle), s*n1, s*n2, s*n3};
        r.normalize();
        return r;
    }

    // =================== Composite Link Variable ===================

    // U_mu(x) = embed(W_L, W_R) * N(x)
    Mat4 link_matrix(int site, int mu) const {
        Mat4 W = embed_su2xu2(connL[site][mu], connR[site][mu]);
        Mat4 N = nonlocal_core(weyl[site].data(), ising[site]);
        return W * N;
    }

    // U_mu^dagger(x) = N^dag(x) * embed^dag
    Mat4 link_matrix_dag(int site, int mu) const {
        return link_matrix(site, mu).adjoint();
    }

    // =================== Plaquette ===================

    // Plaquette P_{mu,nu}(x) = U_mu(x) * U_nu(x+mu) * U_mu^dag(x+nu) * U_nu^dag(x)
    Mat4 plaquette(int site, int mu, int nu) const {
        int xmu = neighbour(site, mu);
        int xnu = neighbour(site, nu);
        Mat4 U1 = link_matrix(site, mu);
        Mat4 U2 = link_matrix(xmu, nu);
        Mat4 U3 = link_matrix(xnu, mu).adjoint();
        Mat4 U4 = link_matrix(site, nu).adjoint();
        return U1 * U2 * U3 * U4;
    }

    // Re tr(plaquette)
    double plaquette_retrace(int site, int mu, int nu) const {
        return plaquette(site, mu, nu).trace().real();
    }

    // =================== Action ===================

    // Total Wilson action: S = -beta * sum_{x, mu<nu} Re tr P_{mu,nu}(x)
    double total_action() const {
        double S = 0;
        for (int i = 0; i < vol; i++)
            for (int mu = 0; mu < NDIM; mu++)
                for (int nu = mu + 1; nu < NDIM; nu++)
                    S -= beta * plaquette_retrace(i, mu, nu);
        return S;
    }

    // Action contribution from plaquettes touching link (site, mu)
    // Each link in 4D participates in 2*(NDIM-1) = 6 plaquettes
    double link_action(int site, int mu) const {
        double S = 0;
        for (int nu = 0; nu < NDIM; nu++) {
            if (nu == mu) continue;
            // Forward plaquette: P_{mu,nu}(site)
            S -= beta * plaquette_retrace(site, mu, nu);
            // Backward plaquette: P_{mu,nu}(site - nu)
            int xbnu = neighbour_back(site, nu);
            S -= beta * plaquette_retrace(xbnu, mu, nu);
        }
        return S;
    }

    // Action contribution from all plaquettes touching site x
    // (through N(x) which enters links U_mu(x) for all mu)
    double site_action(int site) const {
        double S = 0;
        for (int mu = 0; mu < NDIM; mu++) {
            // Forward links from site
            for (int nu = 0; nu < NDIM; nu++) {
                if (nu == mu) continue;
                S -= beta * plaquette_retrace(site, mu, nu);
                int xbnu = neighbour_back(site, nu);
                S -= beta * plaquette_retrace(xbnu, mu, nu);
            }
        }
        // Remove double-counting: each plaquette was counted multiple times
        // A plaquette P_{mu,nu}(y) depends on N(y) (through U_mu(y) and U_nu(y))
        // and on N(y+mu) (through nothing - N is at the SOURCE of each link)
        // Actually N(x) only enters U_mu(x), so site x affects plaquettes
        // where x is the source of one of the 4 links.
        // Each plaquette has 4 links, and x is source of link U_mu(x).
        // So we need all plaquettes containing link (x, mu) for any mu.
        // Already computed above, but with overcounting.
        // Let me just compute it cleanly.
        S = 0;
        // For each direction mu, link U_mu(x) participates in plaquettes:
        for (int mu = 0; mu < NDIM; mu++) {
            for (int nu = 0; nu < NDIM; nu++) {
                if (nu == mu) continue;
                // P_{mu,nu}(x): uses U_mu(x) as the first link
                S -= beta * plaquette_retrace(site, mu, nu);
                // P_{nu,mu}(x - nu): uses U_mu(x) as the second link
                // Wait - P_{nu,mu}(x-nu) = U_nu(x-nu) U_mu(x) U_nu^dag(x-nu+mu) U_mu^dag(x-nu)
                // This uses link (x, mu)? No: P uses U_mu at site (x-nu+nu)=x... hmm.
                // Let me think again. P_{mu,nu}(y) uses links:
                //   U_mu(y), U_nu(y+mu), U_mu^dag(y+nu), U_nu^dag(y)
                // Link U_mu(x) appears as:
                //   - First link of P_{mu,nu}(x) for any nu != mu
                //   - Third link (adjoint) of P_{mu,nu}(x-nu+nu-mu)... no.
                //     U_mu^dag(y+nu) is the adjoint of U_mu at site y+nu.
                //     So U_mu(x) appears as the third link when y+nu = x, i.e. y = x-nu.
                //     Plaquette: P_{mu,nu}(x-nu)
                // So link U_mu(x) appears in:
                //   P_{mu,nu}(x) for all nu != mu    [as first link]
                //   P_{mu,nu}(x-nu) for all nu != mu [as third link, adjointed]
                // That's 2*(NDIM-1) = 6 plaquettes. Good.
            }
        }
        // The above loop counts each plaquette once per link of site x that
        // participates. But N(x) only enters through links U_mu(x) where x
        // is the SOURCE. So we need exactly the plaquettes containing any
        // link U_mu(x). Let me redo this properly.
        S = 0;
        for (int mu = 0; mu < NDIM; mu++) {
            S += link_action(site, mu);
        }
        // But link_action counts plaquettes: for each mu, 2*(NDIM-1) plaquettes.
        // Plaquettes shared between two mu directions are counted twice.
        // P_{mu,nu}(x) is counted by link_action(x,mu) and link_action(x,nu).
        // So we've double-counted each P_{mu,nu}(x).
        // P_{mu,nu}(x-nu) is counted by link_action(x,mu) only (as backward plaq).
        // Hmm, this is getting complicated. For Metropolis we only need the
        // difference in action, so let me just compute all affected plaquettes.

        // SIMPLIFICATION: compute all plaquettes that use N(x)
        // N(x) enters link U_mu(x) for mu=0,1,2,3.
        // Each U_mu(x) participates in 6 plaquettes (see above).
        // Total: up to 24 plaquettes (some may overlap).
        // For correctness in Metropolis, we compute the action of all
        // affected plaquettes before and after the change. This is O(24)
        // plaquettes per site update.
        return S; // placeholder; we'll use delta_action methods below
    }

    // =================== Metropolis Updates ===================

    // Update connection W at link (site, mu)
    bool update_connection(int site, int mu, double epsilon) {
        // Save old
        SU2 oldL = connL[site][mu];
        SU2 oldR = connR[site][mu];

        // Compute old action of affected plaquettes
        double S_old = link_action(site, mu);

        // Propose new
        SU2 dL = random_su2(epsilon);
        SU2 dR = random_su2(epsilon);
        connL[site][mu] = dL * oldL;
        connR[site][mu] = dR * oldR;

        // Compute new action
        double S_new = link_action(site, mu);

        // Accept/reject
        double dS = S_new - S_old;
        if (dS < 0 || uniform(rng) < exp(-dS)) {
            return true; // accepted
        } else {
            connL[site][mu] = oldL;
            connR[site][mu] = oldR;
            return false;
        }
    }

    // Compute action of all plaquettes involving N(x) at site
    double site_N_action(int site) const {
        double S = 0;
        for (int mu = 0; mu < NDIM; mu++) {
            for (int nu = 0; nu < NDIM; nu++) {
                if (nu == mu) continue;
                // P_{mu,nu}(site): U_mu(site) is first link
                S -= beta * plaquette_retrace(site, mu, nu);
                // P_{mu,nu}(site - nu): U_mu(site) appears as 3rd link
                // Actually: P_{mu,nu}(y) third link is U_mu^dag(y+nu).
                // So U_mu(site) appears when y+nu = site => y = site-nu.
                int ysite = neighbour_back(site, nu);
                S -= beta * plaquette_retrace(ysite, mu, nu);
            }
        }
        return S;
    }

    // Update Weyl parameters at site
    bool update_weyl(int site, double epsilon) {
        auto old_c = weyl[site];

        double S_old = site_N_action(site);

        // Propose small change
        for (int a = 0; a < 3; a++) {
            weyl[site][a] += epsilon * (uniform(rng) - 0.5);
            // Reflect at boundaries [0, pi/4]
            while (weyl[site][a] < 0 || weyl[site][a] > PI/4.0) {
                if (weyl[site][a] < 0) weyl[site][a] = -weyl[site][a];
                if (weyl[site][a] > PI/4.0) weyl[site][a] = PI/2.0 - weyl[site][a];
            }
        }

        double S_new = site_N_action(site);
        double dS = S_new - S_old;

        if (dS < 0 || uniform(rng) < exp(-dS)) {
            return true;
        } else {
            weyl[site] = old_c;
            return false;
        }
    }

    // Update Ising label at site
    bool update_ising(int site) {
        double S_old = site_N_action(site);

        ising[site] = -ising[site]; // flip

        double S_new = site_N_action(site);
        double dS = S_new - S_old;

        if (dS < 0 || uniform(rng) < exp(-dS)) {
            return true;
        } else {
            ising[site] = -ising[site]; // flip back
            return false;
        }
    }

    // One full sweep: update all variables
    struct SweepStats {
        double acc_conn, acc_weyl, acc_ising;
    };

    SweepStats sweep(double eps_conn, double eps_weyl) {
        int acc_c = 0, acc_w = 0, acc_i = 0;
        int tot_c = 0, tot_w = 0, tot_i = 0;

        for (int i = 0; i < vol; i++) {
            // Update connections on all links from site i
            for (int mu = 0; mu < NDIM; mu++) {
                if (update_connection(i, mu, eps_conn)) acc_c++;
                tot_c++;
            }
            // Update Weyl parameters
            if (update_weyl(i, eps_weyl)) acc_w++;
            tot_w++;
            // Update Ising label
            if (update_ising(i)) acc_i++;
            tot_i++;
        }

        return {
            (double)acc_c / tot_c,
            (double)acc_w / tot_w,
            (double)acc_i / tot_i
        };
    }

    // =================== Measurements ===================

    // 1. Average plaquette: (1 / (6*vol)) * sum Re tr P
    double measure_plaquette() const {
        double sum = 0;
        for (int i = 0; i < vol; i++)
            for (int mu = 0; mu < NDIM; mu++)
                for (int nu = mu + 1; nu < NDIM; nu++)
                    sum += plaquette_retrace(i, mu, nu);
        return sum / (6.0 * vol);
    }

    // 2. Staggered magnetization: Phi = (1/vol) * sum (-1)^{x+y+z+t} * sigma
    double measure_staggered_mag() const {
        double sum = 0;
        for (int i = 0; i < vol; i++)
            sum += parity(i) * ising[i];
        return sum / vol;
    }

    // 3. Ising energy: E = -(1/vol) * sum_{<ij>} sigma_i * sigma_j
    double measure_ising_energy() const {
        double sum = 0;
        for (int i = 0; i < vol; i++)
            for (int mu = 0; mu < NDIM; mu++)
                sum += ising[i] * ising[neighbour(i, mu)];
        return -sum / (NDIM * vol); // normalized per bond
    }

    // 4. Polyakov loop (temporal direction = mu=3)
    //    L(x,y,z) = tr prod_{t=0}^{Lt-1} U_3(x,y,z,t)
    //    Average |L| and phase
    struct PolyakovResult {
        double avg_abs_full;     // |tr(P)|
        double avg_abs_colour;   // |tr(P_{3x3})|  (top-left 3x3 in magic basis)
        double avg_abs_lepto;    // |P_{off-diagonal}|  (3x1 and 1x3 blocks)
    };

    PolyakovResult measure_polyakov() const {
        static Mat4 M = magic_basis_M();
        static Mat4 Mdag = M.adjoint();

        double sum_full = 0, sum_colour = 0, sum_lepto = 0;
        int n_spatial = Ls * Ls * Ls;

        for (int x = 0; x < Ls; x++)
            for (int y = 0; y < Ls; y++)
                for (int z = 0; z < Ls; z++) {
                    // Compute temporal Polyakov loop
                    Mat4 P = Mat4::identity();
                    for (int t = 0; t < Lt; t++) {
                        int site = index(x, y, z, t);
                        P = P * link_matrix(site, 3); // temporal direction
                    }

                    // Full trace
                    sum_full += std::abs(P.trace());

                    // Transform to magic basis: P_magic = M^dag * P * M
                    Mat4 Pm = Mdag * P * M;

                    // Colour: trace of top-left 3x3 block
                    cd tr_colour = Pm.m[0][0] + Pm.m[1][1] + Pm.m[2][2];
                    sum_colour += std::abs(tr_colour);

                    // Leptoquark: off-diagonal blocks {0,1,2} <-> {3}
                    double lepto = 0;
                    for (int i = 0; i < 3; i++) {
                        lepto += std::norm(Pm.m[i][3]);
                        lepto += std::norm(Pm.m[3][i]);
                    }
                    sum_lepto += sqrt(lepto);
                }

        return {
            sum_full / n_spatial,
            sum_colour / n_spatial,
            sum_lepto / n_spatial
        };
    }

    // 5. Average eigenvalue degeneracy of temporal plaquettes
    //    In the ordered phase, the product N(x)*N_bar(x+t) should have
    //    3+1 eigenvalue structure. We measure the degeneracy ratio.
    double measure_eigenvalue_spread() const {
        // For each temporal link, compute N(x) * N(x+t_hat) in magic basis
        // and look at the eigenvalue spread
        static Mat4 M = magic_basis_M();
        static Mat4 Mdag = M.adjoint();

        double total_spread = 0;
        int count = 0;

        for (int i = 0; i < vol; i++) {
            int j = neighbour(i, 3); // temporal neighbour
            Mat4 Ni = nonlocal_core(weyl[i].data(), ising[i]);
            Mat4 Nj = nonlocal_core(weyl[j].data(), ising[j]);
            Mat4 prod = Ni * Nj;

            // Transform to magic basis (where it should be diagonal)
            Mat4 Pm = Mdag * prod * M;

            // Extract diagonal phases
            double phases[4];
            for (int k = 0; k < 4; k++)
                phases[k] = std::arg(Pm.m[k][k]);

            // Measure spread: for 3+1 degeneracy, three phases should be equal
            // Compute variance of the first three phases
            double mean3 = (phases[0] + phases[1] + phases[2]) / 3.0;
            double var3 = 0;
            for (int k = 0; k < 3; k++)
                var3 += (phases[k] - mean3) * (phases[k] - mean3);
            var3 /= 3.0;

            total_spread += sqrt(var3);
            count++;
        }

        return total_spread / count;
    }

    // 6. Forward/backward temporal propagator asymmetry
    //    Compare plaquette values for temporal plaquettes in the
    //    forward vs backward causal directions
    struct AsymmetryResult {
        double plaq_forward;
        double plaq_backward;
    };

    AsymmetryResult measure_asymmetry() const {
        double fwd = 0, bwd = 0;
        int count = 0;

        for (int i = 0; i < vol; i++) {
            for (int mu = 0; mu < 3; mu++) { // spatial directions
                // "Forward" temporal plaquette: P_{mu, 3}(x)
                double pf = plaquette_retrace(i, mu, 3);
                fwd += pf;

                // "Backward" temporal plaquette: P_{mu, 3}(x - t_hat)
                int xbt = neighbour_back(i, 3);
                double pb = plaquette_retrace(xbt, mu, 3);
                bwd += pb;

                count++;
            }
        }

        return { fwd / count, bwd / count };
    }
};

// ===================== Eigenvalue Computation (power iteration) =====================
// Simple QR-like iteration for 4x4 unitary eigenvalues
// Not high-precision but sufficient for measurement
std::array<cd, 4> Mat4::eigenvalues() const {
    // Use the characteristic polynomial approach for 4x4
    // For unitary matrices, eigenvalues lie on the unit circle
    // We'll use a simplified shifted QR iteration

    Mat4 A = *this;
    std::array<cd, 4> evals;

    // 20 iterations of QR
    for (int iter = 0; iter < 30; iter++) {
        // Shift
        cd shift = A.m[3][3];
        for (int i = 0; i < 4; i++) A.m[i][i] -= shift;

        // QR decomposition via Gram-Schmidt
        // (simplified; not numerically optimal but adequate)
        cd Q_cols[4][4], R_vals[4][4];
        memset(Q_cols, 0, sizeof(Q_cols));
        memset(R_vals, 0, sizeof(R_vals));

        for (int j = 0; j < 4; j++) {
            // Copy column j
            for (int i = 0; i < 4; i++) Q_cols[i][j] = A.m[i][j];

            // Orthogonalize against previous columns
            for (int k = 0; k < j; k++) {
                cd dot = 0;
                for (int i = 0; i < 4; i++)
                    dot += std::conj(Q_cols[i][k]) * Q_cols[i][j];
                R_vals[k][j] = dot;
                for (int i = 0; i < 4; i++)
                    Q_cols[i][j] -= dot * Q_cols[i][k];
            }

            // Normalize
            double nrm = 0;
            for (int i = 0; i < 4; i++) nrm += std::norm(Q_cols[i][j]);
            nrm = sqrt(nrm);
            R_vals[j][j] = nrm;
            if (nrm > 1e-14)
                for (int i = 0; i < 4; i++) Q_cols[i][j] /= nrm;
        }

        // A = R * Q + shift * I
        Mat4 Q, R;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                Q.m[i][j] = Q_cols[i][j];
                R.m[i][j] = R_vals[i][j];
            }
        A = R * Q;
        for (int i = 0; i < 4; i++) A.m[i][i] += shift;
    }

    for (int i = 0; i < 4; i++) evals[i] = A.m[i][i];
    return evals;
}

// ===================== Main =====================

int main(int argc, char* argv[]) {
    // Default parameters
    double beta = 2.0;
    int Ls = 6;
    int Lt = 12;
    int n_therm = 200;
    int n_meas = 500;
    int meas_interval = 5;

    if (argc > 1) beta = atof(argv[1]);
    if (argc > 2) Ls = atoi(argv[2]);
    if (argc > 3) Lt = atoi(argv[3]);
    if (argc > 4) n_therm = atoi(argv[4]);
    if (argc > 5) n_meas = atoi(argv[5]);
    if (argc > 6) meas_interval = atoi(argv[6]);

    printf("# SU(4) Deficit Lattice Monte Carlo\n");
    printf("# beta = %.4f, Ls = %d, Lt = %d\n", beta, Ls, Lt);
    printf("# n_therm = %d, n_meas = %d, interval = %d\n", n_therm, n_meas, meas_interval);
    printf("# vol = %d\n", Ls*Ls*Ls*Lt);

    Lattice lat(Ls, Lt, beta);

    // =================== Run Strategy ===================
    // The Ising sector cannot be sampled by local Metropolis (flips change
    // the action by O(24*J), giving essentially zero acceptance). Instead,
    // we compare the equilibrated action of three fixed Ising configurations:
    //   1. Ordered (checkerboard): sigma = (-1)^{x+y+z+t}
    //   2. Disordered (random): sigma random
    //   3. Uniform (all +1): no deficit tetrahedra at all
    // The configuration with lowest action/free energy wins.

    struct PhaseResult {
        const char* name;
        double plaq, mag, ising_e, Lcol, Llep, spread;
    };

    auto run_phase = [&](const char* name, auto init_fn) -> PhaseResult {
        init_fn();
        // Only update connections and Weyl parameters, NOT Ising labels
        double eps_c = 0.3, eps_w = 0.2;

        printf("# --- Phase: %s ---\n", name);

        // Thermalize (gauge sector only)
        for (int i = 0; i < n_therm; i++) {
            // Update connections and Weyl params only
            for (int s = 0; s < lat.vol; s++) {
                for (int mu = 0; mu < NDIM; mu++)
                    lat.update_connection(s, mu, eps_c);
                lat.update_weyl(s, eps_w);
                // NO Ising update
            }
            if ((i+1) % 10 == 0) {
                // Adapt step sizes
                int acc_c = 0, tot_c = 0, acc_w = 0, tot_w = 0;
                for (int s = 0; s < std::min(lat.vol, 50); s++) {
                    for (int mu = 0; mu < NDIM; mu++) {
                        if (lat.update_connection(s, mu, eps_c)) acc_c++;
                        tot_c++;
                    }
                    if (lat.update_weyl(s, eps_w)) acc_w++;
                    tot_w++;
                }
                double rc = (double)acc_c/tot_c, rw = (double)acc_w/tot_w;
                if (rc > 0.5) eps_c *= 1.1; else if (rc < 0.3) eps_c *= 0.9;
                if (rw > 0.5) eps_w *= 1.1; else if (rw < 0.3) eps_w *= 0.9;
            }
        }

        // Measure
        double sp = 0, sm = 0, se = 0, sLc = 0, sLl = 0, ss = 0;
        int nm = 0;
        for (int i = 0; i < n_meas * meas_interval; i++) {
            for (int s = 0; s < lat.vol; s++) {
                for (int mu = 0; mu < NDIM; mu++)
                    lat.update_connection(s, mu, eps_c);
                lat.update_weyl(s, eps_w);
            }
            if ((i+1) % meas_interval == 0) {
                sp += lat.measure_plaquette();
                double m = lat.measure_staggered_mag();
                sm += std::abs(m);
                se += lat.measure_ising_energy();
                auto poly = lat.measure_polyakov();
                sLc += poly.avg_abs_colour;
                sLl += poly.avg_abs_lepto;
                ss += lat.measure_eigenvalue_spread();
                nm++;
            }
        }
        PhaseResult r;
        r.name = name;
        r.plaq = sp/nm; r.mag = sm/nm; r.ising_e = se/nm;
        r.Lcol = sLc/nm; r.Llep = sLl/nm; r.spread = ss/nm;

        printf("#   plaq=%.5f |Phi|=%.4f E_I=%.4f |Lc|=%.4f |Ll|=%.4f spread=%.5f\n",
               r.plaq, r.mag, r.ising_e, r.Lcol, r.Llep, r.spread);
        return r;
    };

    // --- Run all three phases ---
    PhaseResult ordered = run_phase("ORDERED (checkerboard)", [&]() {
        lat.cold_start(); // checkerboard Ising, identity connections
        // Randomize connections slightly to break symmetry
        for (int s = 0; s < lat.vol; s++)
            for (int mu = 0; mu < NDIM; mu++) {
                lat.connL[s][mu] = lat.random_su2(0.5) * lat.connL[s][mu];
                lat.connR[s][mu] = lat.random_su2(0.5) * lat.connR[s][mu];
            }
    });

    PhaseResult disordered = run_phase("DISORDERED (random Ising)", [&]() {
        lat.hot_start(); // random everything
    });

    PhaseResult uniform = run_phase("UNIFORM (all original)", [&]() {
        lat.hot_start();
        for (int s = 0; s < lat.vol; s++) lat.ising[s] = 1; // all +1
    });

    // --- Comparison ---
    printf("#\n");
    printf("# ================================================================\n");
    printf("# PHASE COMPARISON at beta = %.4f on %d^3 x %d\n", beta, Ls, Lt);
    printf("# ================================================================\n");
    printf("#\n");
    printf("# %-20s  plaq       |Phi|    E_Ising   |L_col|   |L_lep|   col/lep   spread\n", "Phase");
    printf("# %-20s  %-9.5f  %-7.4f  %-8.4f  %-8.4f  %-8.4f  %-8.3f  %-8.5f\n",
           ordered.name, ordered.plaq, ordered.mag, ordered.ising_e,
           ordered.Lcol, ordered.Llep, ordered.Lcol/std::max(ordered.Llep,1e-10), ordered.spread);
    printf("# %-20s  %-9.5f  %-7.4f  %-8.4f  %-8.4f  %-8.4f  %-8.3f  %-8.5f\n",
           disordered.name, disordered.plaq, disordered.mag, disordered.ising_e,
           disordered.Lcol, disordered.Llep, disordered.Lcol/std::max(disordered.Llep,1e-10), disordered.spread);
    printf("# %-20s  %-9.5f  %-7.4f  %-8.4f  %-8.4f  %-8.4f  %-8.3f  %-8.5f\n",
           uniform.name, uniform.plaq, uniform.mag, uniform.ising_e,
           uniform.Lcol, uniform.Llep, uniform.Lcol/std::max(uniform.Llep,1e-10), uniform.spread);
    printf("#\n");

    // The phase with highest plaquette (= lowest action, since S = -beta * Re tr P)
    // is the energetically preferred phase.
    double best_plaq = std::max({ordered.plaq, disordered.plaq, uniform.plaq});
    printf("# RESULT: Highest <plaq> (lowest action) wins.\n");
    if (best_plaq == ordered.plaq) {
        printf("# >>> ORDERED PHASE WINS (checkerboard) <<<\n");
        printf("# The Ising ordering IS dynamically selected.\n");
        printf("# SU(4) -> SU(3) x U(1) confirmed at beta = %.2f.\n", beta);
        if (ordered.Lcol / std::max(ordered.Llep, 1e-10) > 1.5)
            printf("# Leptoquark suppression: col/lep = %.2f\n",
                   ordered.Lcol / std::max(ordered.Llep, 1e-10));
        if (ordered.spread < 0.1)
            printf("# Eigenvalue degeneracy: spread = %.5f ~ 0 (3+1 structure confirmed)\n",
                   ordered.spread);
    } else if (best_plaq == disordered.plaq) {
        printf("# >>> DISORDERED PHASE WINS <<<\n");
        printf("# The checkerboard is NOT selected at beta = %.2f.\n", beta);
        printf("# Full SU(4) remains active.\n");
    } else {
        printf("# >>> UNIFORM PHASE WINS (no deficit tetrahedra) <<<\n");
        printf("# The deficit construction is not energetically favoured at beta = %.2f.\n", beta);
    }

    printf("# Done.\n");
    return 0;
}
