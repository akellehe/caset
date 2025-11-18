from caset import Spacetime, Simplex
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # just to register 3D projection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

st = Spacetime()
simplex14 = st.createSimplex((1, 4))
simplex23 = st.createSimplex((2, 3))

left, right = None, None
facets14 = simplex14.getFacets()
for facet in facets14:
    if facet.isTimelike():
        left = facet
        break

facets23 = simplex23.getFacets()
for face in facets23:
    if face.isTimelike():
        right = face
        break

updated, succeeded = st.causallyAttachFaces(left, right)

st.embedEuclidean()
vlist = st.getVertexList()
edges = []
xmin, xmax = float("inf"), float("-inf")
ymin, ymax = float("inf"), float("-inf")
zmin, zmax = float("inf"), float("-inf")
for edge in (st.getEdgeList().toVector()):
    source = vlist.get(edge.getSourceId())
    target = vlist.get(edge.getTargetId())

    t1, x1, y1, z1 = source.getCoordinates()
    t2, x2, y2, z2 = target.getCoordinates()

    xmin = min(xmin, x1, x2)
    xmax = max(xmax, x1, x2)
    ymin = min(ymin, y1, y2)
    ymax = max(ymax, y1, y2)
    zmin = min(zmin, z1, z2)
    zmax = max(zmax, z1, z2)

    edges.append([(x1, y1, z1), (x2, y2, z2)])

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
lc = Line3DCollection(edges, linewidths=.5, colors="blue")
ax.add_collection(lc)

ax.set_xlim(xmin - 1, xmax + 1)
ax.set_ylim(ymin - 1, ymax + 1)
ax.set_zlim(zmin - 1, zmax + 1)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
