// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/CobordismDAG.h"

#include <stdexcept>

#include "cobordism/MultiCobordism.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

using complexd = std::complex<double>;

int CobordismDAG::addNode(std::shared_ptr<Spacetime> host,
                          const std::vector<std::vector<complexd>> &literalInputs,
                          const std::vector<std::pair<int, int>> &upstream,
                          const std::vector<std::vector<complexd>> &outputTargets,
                          const std::vector<int> &degrees, double gamma,
                          std::uint64_t seed) {
  nodes_.push_back(Node{std::move(host), literalInputs, upstream, outputTargets,
                        degrees, gamma, seed});
  return static_cast<int>(nodes_.size()) - 1;
}

void CobordismDAG::run(int stage1MaxSteps, int stage1CandidateMoves,
                       double stage2Beta, int stage2MaxIters) {
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
      for (const auto &e : nodes_[i].upstream)
        if (e.first < 0 || e.first >= static_cast<int>(n) || !done_[e.first]) {
          ready = false;
          break;
        }
      if (!ready) continue;

      const Node &nd = nodes_[i];
      // Assemble input targets: literals, then each upstream node's chosen output.
      std::vector<std::vector<complexd>> inputs = nd.literalInputs;
      for (const auto &e : nd.upstream) {
        const auto &up = outputs_[e.first];
        if (e.second < 0 || e.second >= static_cast<int>(up.size()))
          throw std::out_of_range("CobordismDAG::run: bad upstream output index");
        inputs.push_back(up[e.second]);
      }

      MultiCobordism opt(nd.host, inputs, nd.outputTargets, nd.degrees,
                            nd.gamma, nd.seed);
      // One construct seed vertex per block (the first |inputs|+|outputs| ids).
      const auto verts = nd.host->getVertexList()->toVector();
      std::vector<std::uint64_t> inSeeds, outSeeds;
      std::size_t v = 0;
      for (; v < verts.size() && inSeeds.size() < inputs.size(); ++v)
        inSeeds.push_back(verts[v]->getId());
      for (; v < verts.size() && outSeeds.size() < nd.outputTargets.size(); ++v)
        outSeeds.push_back(verts[v]->getId());
      opt.seedInputs(inSeeds);
      opt.seedOutputs(outSeeds);
      opt.runStage1(stage1MaxSteps, stage1CandidateMoves);
      opt.runStage2(stage2Beta, stage2MaxIters);

      residuals_[i] = opt.rU(opt.spacetime());
      outputs_[i] = nd.outputTargets;  // verified outputs, threaded downstream
      done_[i] = true;
      ++completed;
      progressed = true;
    }
    if (!progressed)
      throw std::runtime_error("CobordismDAG::run: cycle (no runnable node)");
  }
}

std::vector<complexd> CobordismDAG::output(int node, int outputIndex) const {
  if (node < 0 || node >= static_cast<int>(outputs_.size()))
    throw std::out_of_range("CobordismDAG::output: node id out of range");
  const auto &outs = outputs_[node];
  if (outputIndex < 0 || outputIndex >= static_cast<int>(outs.size()))
    throw std::out_of_range("CobordismDAG::output: output index out of range");
  return outs[outputIndex];
}

int CobordismDAG::numOutputs(int node) const {
  if (node < 0 || node >= static_cast<int>(outputs_.size()))
    throw std::out_of_range("CobordismDAG::numOutputs: node id out of range");
  return static_cast<int>(outputs_[node].size());
}

double CobordismDAG::residual(int node) const {
  if (node < 0 || node >= static_cast<int>(residuals_.size()))
    throw std::out_of_range("CobordismDAG::residual: node id out of range");
  return residuals_[node];
}

}  // namespace tessera::cobordism
