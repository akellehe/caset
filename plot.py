from caset import Spacetime, Simplex
import collections
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # just to register 3D projection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

st = Spacetime()
print("Creating simplices...")

orientations = collections.deque([(1, 4), (2, 3)])
st.createSimplex((2, 3))
for i in range(100000):
    orientation = orientations.popleft()
    rightSimplex = st.createSimplex(orientation)
    leftFace, rightFace = st.chooseSimplexToGlueTo(rightSimplex)
    complex, succeeded = st.causallyAttachFaces(leftFace, rightFace)
    orientations.append(orientation)

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

plt.show()
