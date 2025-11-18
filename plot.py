from caset import Spacetime, Simplex
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # just to register 3D projection

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Two points
# x1, y1, z1 = 0, 0, 0
# x2, y2, z2 = 1, 2, 3

# Plot the line segment between them
# ax.plot([x1, x2], [y1, y2], [z1, z2])

# Optional: scatter the endpoints
# ax.scatter([x1, x2], [y1, y2], [z1, z2])


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

st.embedEuclidean(4, 0.0001, 10000)
vlist = st.getVertexList()
for edge in (st.getEdgeList().toVector()):
    source = vlist.get(edge.getSourceId())
    target = vlist.get(edge.getTargetId())
    t1, x1, y1, z1 = source.getCoordinates()
    t2, x2, y2, z2 = target.getCoordinates()

ax.plot([x1, x2], [y1, y2], [z1, z2])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
