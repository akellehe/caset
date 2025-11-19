//
// Created by andrew on 10/23/25.
//

#include <torch/torch.h>
#include "Logger.h"
#include "spacetime/Spacetime.h"


namespace caset {

void Spacetime::embedEuclidean(int dimensions, double epsilon) {
  if (vertexList->size() == 0) return;
  if (edgeList->size() == 0) return;

  const int N = vertexList->size();
  const int E = edgeList->size();
  double lr = 10e-3;

  std::vector<std::shared_ptr<Edge>> edgeVector = edgeList->toVector();
  std::vector<std::shared_ptr<Vertex>> vertexVector = vertexList->toVector();

  std::unordered_map<std::uint64_t, int> id2idx;
  id2idx.reserve(vertexVector.size());
  if (vertexVector.empty()) {
    CASET_LOG(WARN_LEVEL, "No vertices to embed!");
    return;
  }
  if (edgeVector.empty()) {
    CASET_LOG(WARN_LEVEL, "No edges to embed!");
    return;
  }

  CASET_LOG(INFO_LEVEL, "Embedding a ", dimensions, "-d Euclidean space with ", N, " vertices and ", E, " edges.");
  for (int i = 0; i < static_cast<int>(vertexVector.size()); ++i) {
    id2idx[vertexVector[i]->getId()] = i;
  }

  torch::Tensor positions = torch::randn({N, dimensions}, torch::TensorOptions().dtype(torch::kDouble)).set_requires_grad(true);
  std::vector<int64_t> src_idx_vec(E);
  std::vector<int64_t> tgt_idx_vec(E);
  std::vector<double>  target_sq_vec(E);

  for (int e = 0; e < E; ++e) {
    const auto& edge = edgeVector[e];
    auto it_s = id2idx.find(edge->getSourceId());
    auto it_t = id2idx.find(edge->getTargetId());
    if (it_s == id2idx.end() || it_t == id2idx.end()) {
      throw std::runtime_error("Edge refers to unknown vertex id");
    }

    src_idx_vec[e] = it_s->second;
    tgt_idx_vec[e] = it_t->second;

    double L = edge->getSquaredLength();
    // If you have Minkowski lengths and want magnitude-only, use std::abs(L).
    target_sq_vec[e] = std::abs(L)
      ? std::abs(L)
      : epsilon; // Avoid zero target distances which can cause issues in optimization;
  }

  auto src_idx = torch::from_blob(src_idx_vec.data(), {E}, torch::TensorOptions().dtype(torch::kLong)).clone();
  auto tgt_idx = torch::from_blob(tgt_idx_vec.data(), {E}, torch::TensorOptions().dtype(torch::kLong)).clone();
  auto target_sq = torch::from_blob(target_sq_vec.data(), {E}, torch::TensorOptions().dtype(torch::kDouble)).clone();

  // 4. Set up optimizer (Adam is simple and robust)
  torch::optim::Adam optimizer({positions}, torch::optim::AdamOptions(lr));

  auto previousLoss = torch::tensor({0});
  auto loss = torch::tensor({0});
  auto iter = 0;
  auto epsilonTensor = torch::tensor({epsilon}, torch::TensorOptions().dtype(torch::kDouble));
  while (iter == 0 || ((loss - previousLoss).abs() > epsilonTensor).item<bool>()) {
    iter++;
    optimizer.zero_grad();

    // 5. Compute predicted squared distances for all edges
    auto src = positions.index_select(0, src_idx);
    auto tgt = positions.index_select(0, tgt_idx);
    auto diff = src - tgt;                        // (E, dim)
    auto sqdist = diff.pow(2).sum(-1);            // (E,)

    // 6. Loss: match squared distances
    auto residual = sqdist - target_sq;
    previousLoss = loss;
    loss = residual.pow(2).mean();

    loss.backward();
    optimizer.step();

    // Optional: early stopping / logging
    if (iter % 100 == 0) {
      std::cout << "[embedEuclidean] iter " << iter
                << " loss = " << loss.item<double>() << std::endl;
    }
  }

  // 7. Write back into Vertex coordinates
  auto pos_cpu = positions.detach().cpu();
  auto pos_accessor = pos_cpu.accessor<double, 2>();

  for (int i = 0; i < N; ++i) {
    std::vector<double> coords(dimensions);
    for (int d = 0; d < dimensions; ++d) {
      coords[d] = pos_accessor[i][d];
    }
    // You don't have a setter now; add one:
    // void Vertex::setCoordinates(const std::vector<double>& coords);
    vertexVector[i]->setCoordinates(coords);
  }
  CASET_LOG(INFO_LEVEL, "Iteration: ", iter, " Loss: ", loss.item<double>(), " Previous Loss: ", previousLoss.item<double>());
}

} // caset