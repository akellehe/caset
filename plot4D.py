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
   norm = (alpha**2 + beta**2) ** 0.5
   alpha /= norm
   beta  /= norm
   x_p = x
   y_p = y
   z_p = alpha * z + beta * t
   return x_p, y_p, z_p

# Label Vertices:
def label_vertices(st, ax):
   vertex_positions = {}  # id -> (x, y, z)
   for vertex in st.getVertexList().toVector():
      x, y, z = project4_to_3(*vertex.getCoordinates())
      vertex_positions[vertex.getId()] = (x, y, z)
   for vid, (x, y, z) in vertex_positions.items():
      ax.text(
         x, y, z,
         f"{vid}",        # label
         color="black",
         fontsize=10,
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
      m = [0.5 * (s + t)  + .02 for s, t in zip(srcX, tgtX)]

      if src.getTime() == tgt.getTime():
         label = f"t={src.getTime():.1f},\nl={edge.getSquaredLength():.1f}"
      else:
         label = f"(t=[{src.getTime():.1f}, {tgt.getTime():.1f}],\nl={edge.getSquaredLength():.1f})"

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
   for i in range(args.n_simplices):
      newSimplex = st.createSimplex(orientations[i % 2])
      try:
         leftFace, rightFace = st.chooseSimplexToGlueTo(newSimplex)
      except TypeError:
         print("Failed to choose gluable simplex.")
         continue

      complex, succeeded = st.causallyAttachFaces(leftFace, rightFace)

   # Embed:
   st.embedEuclidean(dimensions=4, epsilon=10e-10)

   ccs = st.getConnectedComponents()
   print(f"Number of connected components: {len(ccs)}", ccs)

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


if  __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Plot a 4D causal triangulation.")
   parser.add_argument("--n-simplices", type=int, default=4, help="Number of simplices to create.")
   parser.add_argument("--dimensions", type=int, default=4, help="Number of embedding dimensions.")
   args = parser.parse_args()

   build(args)