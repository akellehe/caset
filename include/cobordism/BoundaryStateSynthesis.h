// MIT License
// Copyright (c) 2025 Andrew Kelleher
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TESSERA_COBORDISM_BOUNDARYSTATESYNTHESIS_H
#define TESSERA_COBORDISM_BOUNDARYSTATESYNTHESIS_H

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "mesh/ForwardDeclarations.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;

/// # BoundaryStateSynthesis
///
/// The §4b **cone-and-retry synthesis loop**: given a target qubit
/// \f$ \psi = (c_0, c_1) \f$, find the *simplest* simplicial complex whose
/// \f$ k=0 \f$ Hodge Laplacian \f$ L = D - A \f$ has \f$ \psi \f$ (padded with
/// zero-amplitude auxiliary vertices) as an eigenvector, and return that
/// minimal complex together with its Hermitian edge weights/phases and the
/// realized eigenvalue \f$ \lambda \f$ — the geometric image \f$ \mathrm{geo}(\psi) \f$.
///
/// This is the growth stage built on top of the fixed-complex
/// `EigenstateSynthesis` (#133, the residual + Rayleigh + parameter core) and
/// the pre-geometric vertex-insertion of the Pachner family (#112). It does
/// **not** modify `EigenstateSynthesis`: it constructs a fresh one over the
/// current complex on every optimize pass and calls its public residual /
/// `apply` / parameter surface.
///
/// ## Embedding (§4b.1)
///
/// The qubit is carried by the two **logical** vertices (the two smallest
/// vertex ids of the seed); every other vertex is **auxiliary** and carries
/// zero amplitude, existing only to supply combinatorial freedom:
/// \f$ \psi = (c_0, c_1, 0, \dots, 0) \f$ in the operator's sorted-vertex-id
/// order.
///
/// ## Why the loop is needed (§4b.2)
///
/// On a single edge (\f$ |V| = 2 \f$) the only eigenvectors are the **balanced**
/// \f$ \tfrac{1}{\sqrt2}(e^{i\theta}, \pm 1) \f$, so a general-amplitude qubit
/// (\f$ |c_0| \neq |c_1| \f$) cannot be realized — its residual floors at
/// \f$ w_{\min}^2(|c_0|^2-|c_1|^2)^2 > 0 \f$. Auxiliary vertices lift this
/// obstruction; the minimal number needed is the state's **combinatorial
/// complexity**, reported as \f$ (|V|, |E|) \f$.
///
/// ## The loop (§4b.4)
///
/// 1. Run the non-convex, multi-restart optimizer over the per-edge magnitudes
///    \f$ \{w_{ij}\} \f$ and U(1) phases \f$ \{\theta_{ij}\} \f$, minimizing the
///    eigenvalue-agnostic residual \f$ r(\psi) = \|(I-\psi\psi^\dagger)L\psi\|^2 \f$.
///    (A Levenberg–Marquardt least-squares solver on the residual vector
///    \f$ L\psi - \lambda\psi \f$, with random restarts — the landscape is
///    non-convex.)
/// 2. If no restart drives \f$ r < \epsilon \f$, **cone in one vertex** and
///    re-optimize. The cone joins a fresh apex to the current top simplex,
///    enlarging the parameter space; for the contractible boundary-state
///    complexes of §4b this preserves the homotopy type (verified by the
///    Betti numbers), supplying exactly the auxiliary freedom of §4b.2 without
///    the topology-trivializing collapse the spec's "full cone" warning is
///    about (that warning bites only for the topologically nontrivial degree-k
///    states of §5).
/// 3. Accept the first complex reaching \f$ r < \epsilon \f$. Its \f$ (|V|,|E|) \f$
///    is the combinatorial complexity; the realized eigenvalue is the Rayleigh
///    quotient \f$ \lambda = \psi^\dagger L\psi \f$.
class BoundaryStateSynthesis {
  public:
    /// The geometric image \f$ \mathrm{geo}(\psi) \f$ of an accepted synthesis:
    /// the minimal complex's size, its realized edge parameters, and \f$ \lambda \f$.
    struct Geo {
      /// True iff the loop reached \f$ r < \epsilon \f$ within the cone budget.
      bool converged{false};
      /// The best residual \f$ r \f$ achieved on the accepted complex.
      double residual{0.0};
      /// The realized eigenvalue \f$ \lambda \f$ (Rayleigh quotient) of \f$ \psi \f$.
      double eigenvalue{0.0};
      /// \f$ |V| \f$ of the accepted complex (combinatorial complexity, with `numEdges`).
      std::size_t numVertices{0};
      /// \f$ |E| \f$ of the accepted complex.
      std::size_t numEdges{0};
      /// Number of vertices coned in to reach the accepted complex.
      int conesApplied{0};
      /// The accepted complex's edge magnitudes \f$ \{w_{ij}\} \f$ (EdgeList order).
      std::vector<double> weights{};
      /// The accepted complex's edge phases \f$ \{\theta_{ij}\} \f$ (EdgeList order).
      std::vector<double> phases{};
    };

    /// Construct the loop over a seed complex (§4b.4 seeds it on a 4-simplex
    /// \f$ \Delta^4 \f$; a single edge is the minimal seed that exhibits the
    /// §4b.2 two-vertex floor). The two smallest-id vertices become the logical
    /// pair; coned-in apices take larger ids, so the logical pair stays the
    /// \f$ \psi \f$ head. The held `shared_ptr` keeps the (growing) complex alive.
    explicit BoundaryStateSynthesis(std::shared_ptr<Spacetime> seed);

    /// Run the cone-and-retry loop for the qubit \f$ (c_0, c_1) \f$ and return
    /// \f$ \mathrm{geo}(\psi) \f$. Optimizes the current complex; if it cannot
    /// reach \f$ r < \epsilon \f$, cones in one vertex and retries, up to
    /// `maxCones` times. Leaves the complex realized at the accepted optimum
    /// (or the best found if it never converged).
    /// @param epsilon  acceptance threshold on the residual.
    /// @param restarts random restarts per optimize pass.
    /// @param maxCones maximum auxiliary vertices to cone in.
    /// @param seed     RNG seed for the restart draws (reproducible).
    Geo synthesize(std::complex<double> c0, std::complex<double> c1,
                   double epsilon = 1e-9, int restarts = 64,
                   int maxCones = 5, std::uint64_t seed = 0);

    /// Optimize the **current** complex only (no coning): multi-restart
    /// Levenberg–Marquardt minimizing \f$ r(\psi) \f$. Leaves the complex at the
    /// best parameters found and returns that best residual. This is the §4b.2
    /// floor probe (e.g. on the two-vertex seed) and the per-pass core of
    /// `synthesize`.
    double optimize(std::complex<double> c0, std::complex<double> c1,
                    int restarts = 64, std::uint64_t seed = 0);

    /// Cone in one auxiliary vertex: join a fresh apex (a new largest-id, hence
    /// auxiliary, vertex) to every vertex of the current top simplex, growing
    /// \f$ K_n \to K_{n+1} \f$. Topology-preserving for the contractible §4b
    /// complexes. Returns false (without growing) if the simplex has reached the
    /// Fingerprint vertex capacity.
    bool coneInVertex();

    /// \f$ |V| \f$ of the current complex.
    [[nodiscard]] std::size_t numVertices() const;

    /// \f$ |E| \f$ of the current complex (weight-carrying edges).
    [[nodiscard]] std::size_t numEdges() const;

    /// The current (growing) complex.
    [[nodiscard]] std::shared_ptr<Spacetime> spacetime() const { return st_; }

  private:
    std::shared_ptr<Spacetime> st_;
    std::uint64_t logicalId0_{0};  // smallest vertex id  -> amplitude c0
    std::uint64_t logicalId1_{0};  // 2nd smallest id     -> amplitude c1
    // Vertices of the current top simplex (the one coning extends). Raw
    // pointers owned by the VertexList; valid for the complex's lifetime.
    ::tessera::mesh::VertexPtrs topVerts_{};

    // psi = (c0, c1, 0, ..., 0) in the operator's sorted-vertex-id order, length
    // == current |V|.
    [[nodiscard]] std::vector<std::complex<double>> embed(
        std::complex<double> c0, std::complex<double> c1) const;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_BOUNDARYSTATESYNTHESIS_H
