import torch
from caset import Spacetime, Simplex
import collections
import timeit
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # just to register 3D projection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from caset import Spacetime, Simplex

st = Spacetime()
orientations = [(1, 3), (2, 2)]
for i in range(3):
   newSimplex = st.createSimplex(orientations[i % 2])
   try:
      leftFace, rightFace = st.chooseSimplexToGlueTo(newSimplex)
   except TypeError:
      print("Failed to choose gluable simplex.")
      continue

   complex, succeeded = st.causallyAttachFaces(leftFace, rightFace)


def project4_to_3(t, x, y, z, alpha=0.7, beta=0.7):
   # Normalize so alpha^2 + beta^2 ~= 1 if you care:
   norm = (alpha**2 + beta**2) ** 0.5
   alpha /= norm
   beta  /= norm
   x_p = x
   y_p = y
   z_p = alpha * z + beta * t
   return x_p, y_p, z_p

print("Embedding...")
st.embedEuclidean(dimensions=4, epsilon=10e-10)
vlist = st.getVertexList()
timeEdges = []
spaceEdges = []
xmin, xmax = float("inf"), float("-inf")
ymin, ymax = float("inf"), float("-inf")
zmin, zmax = float("inf"), float("-inf")

# Label Vertices:
def label_vertices():
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

def label_edges():
   vertex_positions = {}  # id -> (x, y, z)
   vlist = st.getVertexList().toVector()
   vertexById = {v.getId(): v for v in vlist}
   for vertex in vlist:
      x, y, z = project4_to_3(*vertex.getCoordinates())
      vertex_positions[vertex.getId()] = (x, y, z)

   for edge in st.getEdgeList().toVector():
      src = vertex_positions[edge.getSourceId()]
      tgt = vertex_positions[edge.getTargetId()]

      srcVertex = vertexById[edge.getSourceId()]
      tgtVertex = vertexById[edge.getTargetId()]

      # Midpoint
      mx = 0.5 * (src[0] + tgt[0])
      my = 0.5 * (src[1] + tgt[1])
      mz = 0.5 * (src[2] + tgt[2])

      # Optional small offset so text isn’t exactly on the line
      ox, oy, oz = 0.02, 0.02, 0.02
      mx += ox; my += oy; mz += oz

      if srcVertex.getTime() == tgtVertex.getTime():
            label = f"t={srcVertex.getTime():.1f},\nl={edge.getSquaredLength():.1f}"
      else:
            label = f"(t=[{srcVertex.getTime():.1f}, {tgtVertex.getTime():.1f}],\nl={edge.getSquaredLength():.1f})"

      ax.text(
         mx, my, mz,
         label,
         color="gray",
         fontsize=7
      )


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

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
timelike_lc = Line3DCollection(timeEdges, linewidths=.7, colors="blue")
spacelike_lc = Line3DCollection(spaceEdges, linewidths=.7, colors="red")
ax.add_collection(timelike_lc)
ax.add_collection(spacelike_lc)

ax.set_xlim(xmin - 1, xmax + 1)
ax.set_ylim(ymin - 1, ymax + 1)
ax.set_zlim(zmin - 1, zmax + 1)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

label_vertices()
label_edges()

plt.show()
