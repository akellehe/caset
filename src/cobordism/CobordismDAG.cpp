// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/CobordismDAG.h"

#include <stdexcept>

#include "cobordism/EmergentOptimizer.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

using cd = std::complex<double>;

int CobordismDAG::addNode(std::shared_ptr<Spacetime> host,
                          std::vector<std::vector<cd>> literalInputs,
                          std::vector<int> upstream,
                          std::vector<cd> outputTarget, std::vector<int> degrees,
                          double gamma, std::uint64_t seed) {
  nodes_.push_back(Node{std::move(host), std::move(literalInputs),
                        std::move(upstream), std::move(outputTarget),
                        std::move(degrees), gamma, seed});
  return static_cast<int>(nodes_.size()) - 1;
}

void CobordismDAG::run(int stage1MaxSteps, int stage1Candidates,
                       int stage1Patience, double stage2Beta,
                       int stage2MaxIters) {
  const std::size_t n = nodes_.size();
  outputs_.assign(n, {});
  residuals_.assign(n, 0.0);
  done_.assign(n, false);

  std::size_t completed = 0;
  while (completed < n) {
    bool progressed = false;
    for (std::size_t i = 0; i < n; ++i) {
      if (done_[i]) continue;
      bool ready = true;
      for (int u : nodes_[i].upstream)
        if (u < 0 || u >= static_cast<int>(n) || !done_[u]) {
          ready = false;
          break;
        }
      if (!ready) continue;

      const Node &nd = nodes_[i];
      // Assemble input targets: literals, then each upstream node's output.
      std::vector<std::vector<cd>> inputs = nd.literalInputs;
      for (int u : nd.upstream) inputs.push_back(outputs_[u]);

      EmergentOptimizer opt(nd.host, inputs, nd.outputTarget, nd.degrees,
                            nd.gamma, nd.seed);
      // One construct_inputs seed vertex per input (the first |inputs| ids).
      std::vector<std::uint64_t> seeds;
      for (const auto *v : nd.host->getVertexList()->toVector()) {
        if (seeds.size() >= inputs.size()) break;
        seeds.push_back(v->getId());
      }
      opt.constructInputs(seeds, /*rounds=*/12);
      opt.runStage1(stage1MaxSteps, stage1Candidates, stage1Patience);
      opt.relaxStage2(stage2Beta, stage2MaxIters);

      residuals_[i] = opt.rU(opt.spacetime());
      outputs_[i] = nd.outputTarget;  // the verified output, threaded downstream
      done_[i] = true;
      ++completed;
      progressed = true;
    }
    if (!progressed)
      throw std::runtime_error("CobordismDAG::run: cycle (no runnable node)");
  }
}

std::vector<cd> CobordismDAG::output(int node) const {
  if (node < 0 || node >= static_cast<int>(outputs_.size()))
    throw std::out_of_range("CobordismDAG::output: node id out of range");
  return outputs_[node];
}

double CobordismDAG::residual(int node) const {
  if (node < 0 || node >= static_cast<int>(residuals_.size()))
    throw std::out_of_range("CobordismDAG::residual: node id out of range");
  return residuals_[node];
}

}  // namespace tessera::cobordism
