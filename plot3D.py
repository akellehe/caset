import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from caset import Spacetime, Simplex
st = Spacetime()
orientations = [(1, 2), (2, 1)]
st.createSimplex((1, 2))
rightSimplex = st.createSimplex((2, 1))

leftFace, rightFace = st.chooseSimplexToGlueTo(rightSimplex)

print("Embedding...")
st.embedEuclidean()

vlist = st.getVertexList()
timeEdges = []
spaceEdges = []

xmin, xmax = float("inf"), float("-inf")
ymin, ymax = float("inf"), float("-inf")
zmin, zmax = float("inf"), float("-inf")

for edge in (st.getEdgeList().toVector()):
   source = vlist.get(edge.getSourceId())
   target = vlist.get(edge.getTargetId())

   x1, y1, t1 = source.getCoordinates()
   x2, y2, t2 = target.getCoordinates()

   xmin = min(xmin, x1, x2)
   xmax = max(xmax, x1, x2)
   ymin = min(ymin, y1, y2)
   ymax = max(ymax, y1, y2)
   zmin = min(zmin, t1, t2)
   zmax = max(zmax, t1, t2)

   print("Squared length: ", edge.getSquaredLength())
   if edge.getSquaredLength() < 0:
       timeEdges.append([(x1, y1, t1), (x2, y2, t2)])
   else:
       spaceEdges.append([(x1, y1, t1), (x2, y2, t2)])

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

plt.show()
