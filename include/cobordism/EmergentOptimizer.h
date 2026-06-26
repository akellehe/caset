// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_EMERGENTOPTIMIZER_H
#define TESSERA_COBORDISM_EMERGENTOPTIMIZER_H

#include <complex>
#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <utility>
#include <vector>

namespace tessera::spacetime { class Spacetime; }

namespace tessera::cobordism {
using ::tessera::spacetime::Spacetime;

/// # EmergentOptimizer
///
/// The C++ source-of-truth port of `examples/cobordism/emergent_optimizer.py`
/// (epic #457 / T5, #491): the merge as a **fully emergent** optimization — no
/// prescribed topology, no hand-placed register. From a bare host it grows the
/// register by **gated surgical moves** under the objective and reads the register
/// **dynamically** off `getBoundary` at a **user-defined degree k**.
///
/// Objective (the four-term `F`, extremize δS=0 — never minimize |S|):
/// \f[ F = \lVert\nabla S_{\text{Regge}}\rVert^2
///        + \Gamma\,\big( r_U(\text{output}) + \textstyle\sum_i r_U(\text{input}_i) \big) \f]
/// summed over the register `degrees`. `‖∇S‖²` is the **full complex**
/// `Σ_e |actionGradientExact_e|²`; each `r_U` is the relabeling-invariant,
/// zero-filled `residualForPeriods` over the emergent holes (the whole's holes for
/// the output; each input sub-complex's own holes for the inputs).
///
/// Two stages, exactly as the reference:
///   * **Stage 1 (combinatorial):** greedy best-ΔF single random moves
///     `{add,remove,flip,iflip,cone_out,cone_in}`, each gated by `dualComplexValid`
///     and "no input vertex removed", committed only if ΔF < 0; re-seed on stall.
///   * **Stage 2 (geometric):** relax every (complex) edge `ℓ²` toward a stationary
///     point of `β‖∇S‖² + Γ·r_U` (Wirtinger steepest descent, backtracking line
///     search), re-opening the scale DOF.
class EmergentOptimizer {
 public:
  /// An emergent input: the vertex SET whose own sub-complex `L_k` must carry
  /// `target`, and the target period vector.
  struct Input {
    std::set<std::uint64_t> verts;
    std::vector<std::complex<double>> target;
  };

  EmergentOptimizer(std::shared_ptr<Spacetime> host,
                    std::vector<std::vector<std::complex<double>>> inputTargets,
                    std::vector<std::complex<double>> outputTarget,
                    std::vector<int> degrees = {3}, double gamma = 1.0,
                    std::uint64_t seed = 0);

  // ---- module-level helpers (static; the reference's free functions) ----
  /// Betti numbers (combinatorial, geometry-free).
  [[nodiscard]] static std::vector<int> betti(const Spacetime &st);
  /// The emergent k-register, read off `getBoundary`: the `(k+2)`-vertex tuples
  /// all of whose drop-one facets are boundary facets. Nothing placed.
  [[nodiscard]] static std::vector<std::vector<std::uint64_t>> emergentHoles(
      const Spacetime &st, int k);
  /// `Σ_e |actionGradientExact_e|²` — the full-complex Regge extremization term.
  [[nodiscard]] static double gradNorm2(const std::shared_ptr<Spacetime> &st);
  /// The relabeling-invariant, zero-filled residual of `target` against the
  /// `L_k` harmonic of `st` over its emergent holes (`r_state` in the reference).
  [[nodiscard]] static double rState(
      const std::shared_ptr<Spacetime> &st, int k,
      const std::vector<std::complex<double>> &target);

  // ---- objective ----
  /// The three-term register residual summed over `degrees_`: `r_state(output) +
  /// Σ_i r_input(input_i)`.
  [[nodiscard]] double rU(const std::shared_ptr<Spacetime> &st) const;
  /// `F = gradNorm2 + gamma * rU`.
  [[nodiscard]] double objective() const;

  // ---- the two stages + input construction ----
  void constructInputs(const std::vector<std::uint64_t> &seeds, int rounds = 24);
  std::vector<double> runStage1(int maxSteps = 200, int nCandidates = 12,
                                int patience = 8);
  std::vector<double> relaxStage2(double beta = 1.0, int maxIters = 40,
                                  double alpha0 = 0.05);

  [[nodiscard]] std::shared_ptr<Spacetime> spacetime() const { return st_; }
  [[nodiscard]] const std::vector<Input> &inputs() const { return inputs_; }

 private:
  using Snapshot =
      std::pair<std::vector<std::vector<std::uint64_t>>,
                std::map<std::pair<std::uint64_t, std::uint64_t>,
                         std::complex<double>>>;
  using MoveSpec = std::pair<std::string, std::vector<std::uint64_t>>;

  [[nodiscard]] std::shared_ptr<Spacetime> subOf(
      const std::shared_ptr<Spacetime> &st, const std::set<std::uint64_t> &verts)
      const;
  [[nodiscard]] double rInput(const Input &inp,
                              const std::shared_ptr<Spacetime> &st) const;
  [[nodiscard]] std::set<std::uint64_t> inputVerts() const;

  [[nodiscard]] Snapshot snapshotOf(const Spacetime &st) const;
  [[nodiscard]] Snapshot snapshot() const;
  [[nodiscard]] std::shared_ptr<Spacetime> build(const Snapshot &snap) const;

  [[nodiscard]] MoveSpec randomSpec(const Spacetime &st);
  [[nodiscard]] bool applySpec(const std::shared_ptr<Spacetime> &st,
                               const MoveSpec &spec);
  [[nodiscard]] double deltaF(const std::shared_ptr<Spacetime> &cand,
                              double baseRu,
                              const std::set<std::vector<std::uint64_t>> &baseCells)
      const;
  double step(int nCandidates);

  std::shared_ptr<Spacetime> st_;
  std::vector<std::vector<std::complex<double>>> inputTargets_;
  std::vector<std::complex<double>> outputTarget_;
  std::vector<int> degrees_;
  int gateK_;
  double gamma_;
  std::mt19937_64 rng_;
  double tol_ = 1e-9;
  std::vector<Input> inputs_;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_EMERGENTOPTIMIZER_H
