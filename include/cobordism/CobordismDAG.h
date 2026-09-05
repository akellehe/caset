// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_COBORDISMDAG_H
#define TESSERA_COBORDISM_COBORDISMDAG_H

#include <complex>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "cobordism/PencilLayer.h"

namespace tessera::spacetime { class Spacetime; }

namespace tessera::cobordism {
using ::tessera::spacetime::Spacetime;

/// # CobordismDAG
///
/// Chain emergent merges (`MultiCobordism`) into a DAG: the **output of one
/// cobordism is an input to the next** (#491). Generalizes the
/// `proton_merge_sequence` compose — `merge(q,q)→diquark`, then
/// `merge(diquark,q)→proton` — to an arbitrary acyclic graph, on the C++
/// source-of-truth engine.
///
/// Each node is one merge: a bare host, literal input targets, edges that pipe
/// upstream nodes' outputs into further input slots, and a prescribed
/// `outputTarget` (the MultiCobordism semantics — the output is scored by its
/// own `r_U`). `run()` executes the nodes in topological order, assembling each
/// node's input targets from its literals plus the resolved upstream outputs,
/// running both stages, and recording the node's output (its verified
/// `outputTarget`) and its final realizability residual `r_U`.
class CobordismDAG {
 public:
  /// Add a node — one co-optimized `MultiCobordism` system. `host` is a bare
  /// emergent host. The node's input targets are `literalInputs` followed by, for
  /// each `(nodeId, outputIndex)` in `upstream`, that upstream node's
  /// `outputIndex`-th output. `outputTargets` is the list of output boundary
  /// blocks (one for a merge, two for a 2→2 recombination). `degrees` is the
  /// user-defined register degree(s) k. Returns the node id. Coupled interactions
  /// (e.g. the recombination's two pairs) MUST be one node; uncoupled ones (the
  /// proton/antiproton legs) are separate nodes.
  int addNode(
      std::shared_ptr<Spacetime> host,
      const std::vector<std::vector<std::complex<double>>> &literalInputs,
      const std::vector<std::pair<int, int>> &upstream,
      const std::vector<std::vector<std::complex<double>>> &outputTargets,
      const std::vector<int> &degrees = {3}, double gamma = 1.0,
      std::uint64_t seed = 0);

  /// Run every node in topological order. Stage-1 (combinatorial) and stage-2
  /// (geometric) parameters are shared across nodes. Raises on a cycle.
  void run(int stage1MaxSteps = 30, int stage1CandidateMoves = 8,
           double stage2Beta = 1.0, int stage2MaxIters = 40);

  /// The node's `outputIndex`-th output (its verified target), valid after run.
  [[nodiscard]] std::vector<std::complex<double>> output(int node,
                                                         int outputIndex) const;
  /// Number of outputs of a node.
  [[nodiscard]] int numOutputs(int node) const;
  /// The node's final realizability residual `r_U` (≈0 ⇒ realizable), after run.
  [[nodiscard]] double residual(int node) const;
  [[nodiscard]] std::size_t size() const { return nodes_.size(); }

  /// Pipe fibers (#916): after each node runs, read the fiber form of every
  /// output block at `degree` (`MultiCobordism::readOutputFiber` on the
  /// harmonic contour) and attach it to the downstream input block the edge
  /// names, beside the period target. A read that refuses leaves the slot
  /// empty and records the reason (`fiberRefusal`). Requires the
  /// process-wide Whitney pencil metric source when the DAG runs.
  /// \p scoreBlocksByFiber makes every node score its piped input blocks by
  /// the fiber residual (`MultiCobordism::useFiberResiduals`, #940) instead of
  /// the period residual; off by default.
  void setFiberPiping(bool enabled, int degree = 1, bool scoreBlocksByFiber = false);
  [[nodiscard]] bool scoresBlocksByFiber() const noexcept { return scoreBlocksByFiber_; }
  [[nodiscard]] bool fiberPiping() const noexcept { return pipeFibers_; }
  /// The fiber form of a node's output, valid after run; throws when absent.
  [[nodiscard]] const BoundaryFiber &outputFiber(int node, int outputIndex) const;
  [[nodiscard]] bool hasOutputFiber(int node, int outputIndex) const;
  /// Why a node's output fiber was not read (empty when it was).
  [[nodiscard]] std::string fiberRefusal(int node) const;
  /// How many upstream fibers were attached to a node's input blocks when it ran.
  [[nodiscard]] int pipedInputCount(int node) const;

 private:
  struct Node {
    std::shared_ptr<Spacetime> host;
    std::vector<std::vector<std::complex<double>>> literalInputs;
    std::vector<std::pair<int, int>> upstream;  // (nodeId, outputIndex)
    std::vector<std::vector<std::complex<double>>> outputTargets;
    std::vector<int> degrees;
    double gamma;
    std::uint64_t seed;
  };
  std::vector<Node> nodes_;
  std::vector<std::vector<std::vector<std::complex<double>>>> outputs_;  // per node
  std::vector<double> residuals_;
  std::vector<bool> done_;
  bool pipeFibers_{false};
  int fiberDegree_{1};
  bool scoreBlocksByFiber_{false};
  std::vector<std::vector<std::optional<BoundaryFiber>>> outputFibers_;  // per node
  std::vector<std::string> fiberRefusals_;
  std::vector<int> pipedInputs_;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_COBORDISMDAG_H
