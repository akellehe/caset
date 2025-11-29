# MIT License
# Copyright (c) 2025 Andrew Kelleher
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse

import torch
from caset import Spacetime, Simplex

import collections
import timeit
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # just to register 3D projection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from caset import Spacetime, Simplex


def project4_to_3(t, x, y, z, alpha=0.7, beta=0.7):
    # Normalize so alpha^2 + beta^2 ~= 1 if you care:
    norm = (alpha ** 2 + beta ** 2) ** 0.5
    alpha /= norm
    beta /= norm
    x_p = x
    y_p = y
    z_p = alpha * z + beta * t
    return x_p, y_p, z_p


def embed_euclidean(st, dimensions=4, epsilon=1e-10):
    """
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

    """
    N = st.getVertexList().size()
    E = st.getEdgeList().size()
    lr = 10e-3

    edgeVector = st.getEdgeList().toVector()
    vertexVector = st.getVertexList().toVector()

    vertexIdToIndex = {}
    vertexIdToTime = {}
    for (i, vertex) in enumerate(vertexVector):
        vertexIdToIndex[vertex.getId()] = i
        vertexIdToTime[vertex.getId()] = vertex.getTime()

    edgeIdxToSourceIndex = [0] * E
    edgeIdxToTargetIndex = [0] * E
    edgeIdxToSourceTime = [0.0] * E
    edgeIdxToTargetTime = [0.0] * E
    edgeIdxToAbsoluteSquaredLength = [0.0] * E
    for e, edge in enumerate(edgeVector):
        sourceIndex = vertexIdToIndex[edge.getSourceId()]
        targetIndex = vertexIdToIndex[edge.getTargetId()]
        sourceTime = vertexIdToTime[edge.getSourceId()]
        targetTime = vertexIdToTime[edge.getTargetId()]

        edgeIdxToSourceIndex[e] = sourceIndex
        edgeIdxToTargetIndex[e] = targetIndex
        edgeIdxToSourceTime[e] = sourceTime
        edgeIdxToTargetTime[e] = targetTime

        L = edge.getSquaredLength()
        edgeIdxToAbsoluteSquaredLength[e] = abs(L) if abs(L) > 0 else epsilon

    edgeIdxToSourceIdxTensor = torch.tensor(edgeIdxToSourceIndex, dtype=torch.long)
    edgeIdxToTargetIdxTensor = torch.tensor(edgeIdxToTargetIndex, dtype=torch.long)
    edgeIdxToSourceTimeTensor = torch.tensor(edgeIdxToSourceTime, dtype=torch.double)
    edgeIdxToTargetTimeTensor = torch.tensor(edgeIdxToTargetTime, dtype=torch.double)
    edgeIdxToAbsoluteSquaredLengthTensor = torch.tensor(edgeIdxToAbsoluteSquaredLength, dtype=torch.double)
    # 4. Set up optimizer (Adam is simple and robust)
    positions = torch.randn((N, dimensions), dtype=torch.double, requires_grad=True)
    vertexTimesTensor = torch.zeros((N,), dtype=torch.double)
    for i, vertex in enumerate(vertexVector):
        vertexTimesTensor[i] = vertex.getTime()
    optimizer = torch.optim.Adam([positions], lr=lr)
    previousLoss = torch.tensor(0.0)
    loss = torch.tensor(0.0)
    iter = 0
    epsilonTensor = torch.tensor(epsilon, dtype=torch.double)
    while iter == 0 or (loss - previousLoss).abs().item() > epsilon:
        iter += 1
        optimizer.zero_grad()
        # 5. Compute predicted squared distances for all edges
        srcPositions = positions.index_select(0, edgeIdxToSourceIdxTensor)  # (E, dim)
        tgtPositions = positions.index_select(0, edgeIdxToTargetIdxTensor)  # (E, dim)

        expectedSrcTimes = vertexTimesTensor.index_select(0, edgeIdxToSourceIdxTensor)
        expectedTgtTimes = vertexTimesTensor.index_select(0, edgeIdxToTargetIdxTensor)
        expectedTimes = (expectedSrcTimes + expectedTgtTimes) / 2.0
        observedLengths = srcPositions - tgtPositions  # (E, dim - 1)
        sqdist = observedLengths.pow(2).sum(-1)  # (E,)

        # The observed time is the 0th element of the coordinate vector
        observedSrcTimes = srcPositions[:, 0]
        observedTgtTimes = tgtPositions[:, 0]
        observedTimes = (observedSrcTimes + observedTgtTimes) / 2.
        sqtime = (observedTimes - expectedTimes).pow(2)  # (E,)

        # 6. Loss: match squared distances
        residual = sqdist - edgeIdxToAbsoluteSquaredLengthTensor + (sqtime * dimensions)
        previousLoss = loss
        loss = residual.pow(2).mean()
        loss.backward()
        optimizer.step()

        # Optional: early stopping / logging
        if iter % 100 == 0:
            print(f"[embedEuclidean] iter {iter} loss = {loss.item():.6f}")

    # 7. Write back into Vertex coordinates
    posCpu = positions.detach().cpu()
    for i, vertex in enumerate(vertexVector):
        coords = [0.0] * dimensions
        coords[0] = vertex.getTime()
        for d in range(1, dimensions):
            coords[d] = posCpu[i, d].item()
        vertex.setCoordinates(coords)

    print(f"[embedEuclidean] Iteration: {iter} Loss: {loss.item():.6f} Previous Loss: {previousLoss.item():.6f}")


# Label Vertices:
def label_vertices(st, ax):
    vertex_positions = {}  # id -> (x, y, z)
    for vertex in st.getVertexList().toVector():
        x, y, z = project4_to_3(*vertex.getCoordinates())
        ax.text(
            x, y, z,
            f"{vertex.getId()};t={vertex.getTime():.2f}",  # label
            color="black",
            fontsize=7,
            ha="center",
            va="center"
        )


def label_edges(st, ax):
    for edge in st.getEdgeList().toVector():
        src = st.getVertexList().get(edge.getSourceId())
        tgt = st.getVertexList().get(edge.getTargetId())

        srcX = project4_to_3(*src.getCoordinates())
        tgtX = project4_to_3(*tgt.getCoordinates())

        # Midpoint
        m = [0.5 * (s + t) + .02 for s, t in zip(srcX, tgtX)]

        label = f"l={edge.getSquaredLength():.1f}"

        ax.text(
            *m,
            label,
            color="gray",
            fontsize=7
        )


def get_orientations(dimensions):
    if dimensions == 4:
        return [(1, 4), (2, 3)]
    if dimensions == 3:
        return [(1, 2), (2, 1)]
    else:
        raise ValueError("Only 3D and 4D supported.")


def build(args):
    st = Spacetime()

    # Glue:
    orientations = [(1, 2), (2, 1)]
    unglued = collections.deque()
    for i in range(args.n_simplices):
        newSimplex = st.createSimplex(orientations[i % 2])
        print("----------------------------------------------")
        print("New simplex: ", newSimplex, newSimplex.getOrientation());
        if i == 0:
            continue

        leftFace, rightFace = st.chooseSimplexFacesToGlue(newSimplex)
        print("left Face: ", leftFace, "right face: ", rightFace)

        complex, succeeded = st.causallyAttachFaces(leftFace, rightFace)
        if complex:
            print(complex.getCofaces(), 'Glued: ', succeeded)
        else:
            unglued.append((newSimplex, 0))
            print("Gluing failed.")
            continue

        if unglued:
            print("Attempting to glue unglued simplices...")
            for i in range(len(unglued)):
                newSimplex, retries = unglued.popleft()
                if retries > 3:
                    print("Giving up on simplex after 3 retries: ", newSimplex)
                    continue
                try:
                    leftFace, rightFace = st.chooseSimplexFacesToGlue(newSimplex)
                    complex, succeeded = st.causallyAttachFaces(leftFace, rightFace)
                except RuntimeError:
                    continue
                if not succeeded:
                    unglued.append((newSimplex, retries + 1))
                    print("Gluing from unglued queue failed.")



    print("-------------------</Gluing>---------------------------")

    # Embed:
    embed_euclidean(st, dimensions=4, epsilon=10e-10)
    print("----------------------------------------------")

    ccs = st.getConnectedComponents()
    print(f"Number of connected components: {len(ccs)}", ccs)
    for edge in st.getEdgeList().toVector():
        print(edge)

    # Plot:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    timelike_lc, spacelike_lc = label_bounds_and_get_edges(st, ax)
    ax.add_collection(timelike_lc)
    ax.add_collection(spacelike_lc)

    label_vertices(st, ax)
    label_edges(st, ax)

    plt.show()


def label_bounds_and_get_edges(st, ax):
    vlist = st.getVertexList()
    timeEdges = []
    spaceEdges = []
    xmin, xmax = float("inf"), float("-inf")
    ymin, ymax = float("inf"), float("-inf")
    zmin, zmax = float("inf"), float("-inf")
    for edge in (st.getEdgeList().toVector()):
        source = vlist.get(edge.getSourceId())
        target = vlist.get(edge.getTargetId())

        x1, y1, z1 = project4_to_3(*source.getCoordinates())
        x2, y2, z2 = project4_to_3(*target.getCoordinates())

        xmin = min(xmin, x1, x2)
        xmax = max(xmax, x1, x2)
        ymin = min(ymin, y1, y2)
        ymax = max(ymax, y1, y2)
        zmin = min(zmin, z1, z2)
        zmax = max(zmax, z1, z2)

        print("Squared length: ", edge.getSquaredLength())
        if edge.getSquaredLength() < 0:
            timeEdges.append([(x1, y1, z1), (x2, y2, z2)])
        else:
            spaceEdges.append([(x1, y1, z1), (x2, y2, z2)])

    ax.set_xlim(xmin - 1, xmax + 1)
    ax.set_ylim(ymin - 1, ymax + 1)
    ax.set_zlim(zmin - 1, zmax + 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    timelike_lc = Line3DCollection(timeEdges, linewidths=.7, colors="blue")
    spacelike_lc = Line3DCollection(spaceEdges, linewidths=.7, colors="red")
    return timelike_lc, spacelike_lc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a 4D causal triangulation.")
    parser.add_argument("--n-simplices", type=int, default=4, help="Number of simplices to create.")
    parser.add_argument("--dimensions", type=int, default=4, help="Number of embedding dimensions.")
    args = parser.parse_args()

    build(args)
