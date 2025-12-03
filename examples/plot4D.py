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

import time
import torch
from caset import Spacetime, Simplex

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
    return x_p, y_p, t


def embed_euclidean(st, dimensions=4, epsilon=1e-10):
    """
    Embed vertices in (t, x_1, ..., x_{d-1}) with:
      - t fixed to vertex.getTime() in {0,1}
      - spatial coordinates learned to roughly match edge lengths
      - repulsive term to reduce overlap within the same time slice
    """
    assert dimensions >= 2, "Need at least 1 time + 1 spatial dimension"
    spatial_dims = dimensions - 1

    N = st.getVertexList().size()
    E = st.getEdgeList().size()
    lr = 1e-2
    max_iters = 10000  # safety cap

    edgeVector = st.getEdgeList().toVector()
    vertexVector = st.getVertexList().toVector()

    # --- Map vertex IDs to indices and times ---
    vertexIdToIndex = {}
    vertexTimes = []
    for i, vertex in enumerate(vertexVector):
        vid = vertex.getId()
        vertexIdToIndex[vid] = i
        vertexTimes.append(vertex.getTime())

    vertexTimesTensor = torch.tensor(vertexTimes, dtype=torch.double)  # (N,)

    # --- Edge index and length data ---
    edgeIdxToSourceIndex = [0] * E
    edgeIdxToTargetIndex = [0] * E
    edgeIdxToAbsoluteSquaredLength = [0.0] * E

    for e, edge in enumerate(edgeVector):
        s_id = edge.getSourceId()
        t_id = edge.getTargetId()

        s_idx = vertexIdToIndex[s_id]
        t_idx = vertexIdToIndex[t_id]

        edgeIdxToSourceIndex[e] = s_idx
        edgeIdxToTargetIndex[e] = t_idx

        L = edge.getSquaredLength()
        # Treat |L| as the "target" squared Euclidean edge length
        edgeIdxToAbsoluteSquaredLength[e] = abs(L) if abs(L) > 0 else epsilon

    edgeIdxToSourceIdxTensor = torch.tensor(edgeIdxToSourceIndex, dtype=torch.long)
    edgeIdxToTargetIdxTensor = torch.tensor(edgeIdxToTargetIndex, dtype=torch.long)
    targetSqLenTensor = torch.tensor(edgeIdxToAbsoluteSquaredLength, dtype=torch.double)

    # --- Learn ONLY spatial coordinates (time is fixed separately) ---
    # positions[i] is (x_1, ..., x_{d-1}) for vertex i
    positions = torch.randn((N, spatial_dims), dtype=torch.double, requires_grad=True)

    optimizer = torch.optim.Adam([positions], lr=lr)
    previousLoss = torch.tensor(float("inf"), dtype=torch.double)

    epsilonTensor = torch.tensor(epsilon, dtype=torch.double)
    repulsion_weight = 1.15  # tune this to taste

    iter = 0
    while True:
        iter += 1
        optimizer.zero_grad()

        # --- Edge length term ---
        srcPositions = positions.index_select(0, edgeIdxToSourceIdxTensor)  # (E, spatial_dims)
        tgtPositions = positions.index_select(0, edgeIdxToTargetIdxTensor)  # (E, spatial_dims)

        diff = srcPositions - tgtPositions
        sqdist = diff.pow(2).sum(-1)  # (E,)

        length_residual = sqdist - targetSqLenTensor
        length_loss = (length_residual.pow(2)).mean()

        # --- Repulsion term (mostly within same time slice) ---
        # pairwise distances (N x N)
        # Note: O(N^2), fine for a few thousand vertices, but you can replace with
        # sampling if this gets too big.
        pos_i = positions.unsqueeze(0)         # (1, N, spatial_dims)
        pos_j = positions.unsqueeze(1)         # (N, 1, spatial_dims)
        pair_diff = pos_i - pos_j              # (N, N, spatial_dims)
        pair_sqdist = pair_diff.pow(2).sum(-1) # (N, N)

        # Avoid self-interaction
        eye = torch.eye(N, dtype=torch.double)
        pair_sqdist = pair_sqdist + eye * 1e9

        # More repulsion for vertices with the same time
        same_time = (vertexTimesTensor.unsqueeze(0) == vertexTimesTensor.unsqueeze(1)).to(torch.double)

        # 1 / r^2 repulsion, small epsilon to avoid blow-ups
        repulsion_matrix = same_time * (1.0 / (pair_sqdist + 1e-6))

        # Normalise by number of same-time pairs so weight is scale-ish-independent
        same_time_count = same_time.sum().clamp(min=1.0)
        repulsion_loss = repulsion_matrix.sum() / same_time_count

        loss = length_loss + repulsion_weight * repulsion_loss

        loss.backward()
        optimizer.step()

        if iter % 200 == 0:
            print(
                f"[embedEuclidean-fixedTime] iter {iter} "
                f"loss={loss.item():.6f} length={length_loss.item():.6f} "
                f"rep={repulsion_loss.item():.6f}"
            )

        # convergence check
        if (previousLoss - loss).abs().item() <= epsilon:
            break
        if iter >= max_iters:
            print(f"[embedEuclidean-fixedTime] Hit max_iters={max_iters}, stopping.")
            break

        previousLoss = loss.detach()

    # --- Write back into Vertex coordinates ---
    posCpu = positions.detach().cpu()
    for i, vertex in enumerate(vertexVector):
        coords = [0.0] * dimensions
        # HARD-CONSTRAINED time coordinate:
        coords[0] = float(vertex.getTime())      # should already be 0 or 1
        for d in range(1, dimensions):
            coords[d] = posCpu[i, d - 1].item()
        vertex.setCoordinates(coords)

    print(
        f"[embedEuclidean-fixedTime] Iteration: {iter} "
        f"Final loss: {loss.item():.6f} Previous loss: {previousLoss.item():.6f}"
    )



# Label Vertices:
def label_vertices(st, ax):
    for vertex in st.getVertexList().toVector():
        x, y, z = project4_to_3(*vertex.getCoordinates())
        _t, _x, _y, _z = vertex.getCoordinates()
        ax.text(
            x, y, z,
            f"({_x:.1f},{_y:.1f},{_z:.1f},{_t:.1f})",
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
        ax.text(*m, label, color="gray", fontsize=7)


def get_orientations(dimensions):
    if dimensions == 4:
        return [(1, 4), (2, 3)]
    if dimensions == 3:
        return [(1, 2), (2, 1)]
    else:
        raise ValueError("Only 3D and 4D supported.")


def build(args):
    st = Spacetime()

    print("---------------Building Spacetime-------------")
    start = time.time() * 1000.
    st.build(args.n_simplices)
    end = time.time() * 1000.
    print("Elapsed time: ", end - start)

    # Embed:

    print("---------------Embedding Euclidean------------")
    start = time.time() * 1000.
    embed_euclidean(st, dimensions=4, epsilon=10e-10)
    end = time.time() * 1000.
    print("Elapsed time: ", end - start)
    print("----------------------------------------------")

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
