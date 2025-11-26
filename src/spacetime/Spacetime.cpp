//
// Created by andrew on 10/23/25.
//

#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include "Logger.h"
#include "spacetime/Spacetime.h"


namespace caset {

void Spacetime::embedEuclidean(int dimensions=4, double epsilon=1e-8) {
  pybind11::gil_scoped_release no_gil;
  if (vertexList->size() == 0) return;
  if (edgeList->size() == 0) return;

  const int N = vertexList->size();
  const int E = edgeList->size();
  double lr = 10e-3;

  std::vector<std::shared_ptr<Edge>> edgeVector = edgeList->toVector();
  std::vector<std::shared_ptr<Vertex>> vertexVector = vertexList->toVector();

  if (vertexVector.empty()) {
    CASET_LOG(WARN_LEVEL, "No vertices to embed!");
    return;
  }
  if (edgeVector.empty()) {
    CASET_LOG(WARN_LEVEL, "No edges to embed!");
    return;
  }

  CASET_LOG(INFO_LEVEL, "Embedding a ", dimensions, "-d Euclidean space with ", N, " vertices and ", E, " edges.");
  std::unordered_map<std::uint64_t, int> vertexIdToIndex;
  vertexIdToIndex.reserve(vertexVector.size());
  std::unordered_map<std::uint64_t, double> vertexIdToTime;
  vertexIdToTime.reserve(vertexVector.size());
  for (int i = 0; i < static_cast<int>(vertexVector.size()); ++i) {
    vertexIdToIndex[vertexVector[i]->getId()] = i;
    vertexIdToTime[vertexVector[i]->getId()] = vertexVector[i]->getTime();
  }

  std::vector<int64_t> edgeIdxToSourceIndex(E);
  std::vector<int64_t> edgeIdxToTargetIndex(E);
  std::vector<double> edgeIdxToSourceTime(E);
  std::vector<double> edgeIdxToTargetTime(E);
  std::vector<double>  edgeIdxToAbsoluteSquaredLength(E);

  for (int e = 0; e < E; ++e) {
    const auto& edge = edgeVector[e];
    auto sourceIndexIterator = vertexIdToIndex.find(edge->getSourceId());
    auto targetIndexIterator = vertexIdToIndex.find(edge->getTargetId());
    if (sourceIndexIterator == vertexIdToIndex.end() || targetIndexIterator == vertexIdToIndex.end()) {
      throw std::runtime_error("Edge refers to unknown vertex id");
    }
    auto sourceTimeIterator = vertexIdToTime.find(edge->getSourceId());
    auto targetTimeIterator = vertexIdToTime.find(edge->getTargetId());

    edgeIdxToSourceIndex[e] = sourceIndexIterator->second;
    edgeIdxToTargetIndex[e] = targetIndexIterator->second;
    edgeIdxToSourceTime[e] = sourceTimeIterator->second;
    edgeIdxToTargetTime[e] = targetTimeIterator->second;

    double L = edge->getSquaredLength();
    // If you have Minkowski lengths and want magnitude-only, use std::abs(L).
    edgeIdxToAbsoluteSquaredLength[e] = std::abs(L)
      ? std::abs(L)
      : epsilon; // Avoid zero target distances which can cause issues in optimization;
  }

  auto edgeIdxToSourceIdxTensor = torch::from_blob(edgeIdxToSourceIndex.data(), {E}, torch::TensorOptions().dtype(torch::kLong)).clone();
  auto edgeIdxToTargetIdxTensor = torch::from_blob(edgeIdxToTargetIndex.data(), {E}, torch::TensorOptions().dtype(torch::kLong)).clone();
  auto edgeIdxToSourceTimeTensor = torch::from_blob(edgeIdxToSourceTime.data(), {E}, torch::TensorOptions().dtype(torch::kDouble)).clone();
  auto edgeIdxToTargetTimeTensor = torch::from_blob(edgeIdxToTargetTime.data(), {E}, torch::TensorOptions().dtype(torch::kDouble)).clone();

  auto edgeIdxToAbsoluteSquaredLengthTensor = torch::from_blob(edgeIdxToAbsoluteSquaredLength.data(), {E}, torch::TensorOptions().dtype(torch::kDouble)).clone();

  // 4. Set up optimizer (Adam is simple and robust)
  torch::Tensor positions = torch::randn({N, dimensions}, torch::TensorOptions()
    .dtype(torch::kDouble))
    .set_requires_grad(true);

  torch::Tensor vertexTimesTensor = torch::zeros(
    {N},
    torch::TensorOptions().dtype(torch::kDouble)
    );
  for (int i = 0; i < N; ++i) {
    vertexTimesTensor[i] = vertexVector[i]->getTime();
  }

  torch::optim::Adam optimizer({positions}, torch::optim::AdamOptions(lr));

  auto previousLoss = torch::tensor({0});
  auto loss = torch::tensor({0});
  auto iter = 0;
  auto epsilonTensor = torch::tensor({epsilon}, torch::TensorOptions().dtype(torch::kDouble));
  while (iter == 0 || ((loss - previousLoss).abs() > epsilonTensor).item<bool>()) {
    iter++;
    optimizer.zero_grad();

    // 5. Compute predicted squared distances for all edges
    auto srcPositions = positions.index_select(0, edgeIdxToSourceIdxTensor);  // (E, dim)
    auto tgtPositions = positions.index_select(0, edgeIdxToTargetIdxTensor);  // (E, dim)

    auto expectedSrcTimes = vertexTimesTensor.index_select(0, edgeIdxToSourceIdxTensor);
    auto expectedTgtTimes = vertexTimesTensor.index_select(0, edgeIdxToTargetIdxTensor);
    auto expectedTimes = (expectedSrcTimes + expectedTgtTimes) / 2.;                             // (E,)
    auto observedLengths = srcPositions - tgtPositions;                        // (E, dim - 1)

    auto sqdist = observedLengths.pow(2).sum(-1);            // (E,)

    // The observed time is the 0th element of the coordinate vector
    auto observedSrcTimes = srcPositions.index({torch::arange(0, E), 0});
    auto observedTgtTimes = tgtPositions.index({torch::arange(0, E), 0});
    auto observedTimes = (observedSrcTimes + observedTgtTimes) / 2.;                             // (E,)

    auto sqtime = (observedTimes - expectedTimes).pow(2);                     // (E,)

    // 6. Loss: match squared distances
    auto residual = sqdist - edgeIdxToAbsoluteSquaredLengthTensor + (sqtime * dimensions);
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
  auto posCpu = positions.detach().cpu();
  auto posAccessor = posCpu.accessor<double, 2>();

  for (int i = 0; i < N; ++i) {
    std::vector<double> coords(dimensions);
    coords[0] = vertexVector[i]->getTime();
    for (int d = 1; d < dimensions; ++d) {
      coords[d] = posAccessor[i][d];
    }
    vertexVector[i]->setCoordinates(coords);
  }
  CASET_LOG(INFO_LEVEL, "Iteration: ", iter, " Loss: ", loss.item<double>(), " Previous Loss: ", previousLoss.item<double>());
}

} // caset