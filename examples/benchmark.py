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

from caset import Spacetime, Simplex
import collections
import timeit
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # just to register 3D projection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

print("Creating simplices in python...")

setup_python = """
from caset import Spacetime, Simplex
st = Spacetime()
orientations = [(1, 4), (2, 3)]
"""

stmt_python = """
st.createSimplex((2, 3))
for i in range(100000):
    rightSimplex = st.createSimplex(orientations[i % 2])
    leftFace, rightFace = st.chooseSimplexToGlueTo(rightSimplex)
    complex, succeeded = st.causallyAttachFaces(leftFace, rightFace)
"""


setup_cpp = """
from caset import Spacetime, Simplex
st = Spacetime()
orientations = [(1, 4), (2, 3)]
"""

stmt_cpp = """
st.build()
"""

print("Python: ", timeit.timeit(stmt_python, setup=setup_python, number=1))
print("C++   : ", timeit.timeit(stmt_cpp, setup=setup_cpp, number=1))

#print("Creating simplices in c++...")
#
#def project4_to_3(t, x, y, z, alpha=0.7, beta=0.7):
#    # Normalize so alpha^2 + beta^2 ~= 1 if you care:
#    norm = (alpha**2 + beta**2) ** 0.5
#    alpha /= norm
#    beta  /= norm
#    x_p = x
#    y_p = y
#    z_p = alpha * z + beta * t
#    return x_p, y_p, z_p
#
#print("Embedding...")
#st.embedEuclidean()
#vlist = st.getVertexList()
#timeEdges = []
#spaceEdges = []
#xmin, xmax = float("inf"), float("-inf")
#ymin, ymax = float("inf"), float("-inf")
#zmin, zmax = float("inf"), float("-inf")
#
#for edge in (st.getEdgeList().toVector()):
#    source = vlist.get(edge.getSourceId())
#    target = vlist.get(edge.getTargetId())
#
#    x1, y1, z1 = project4_to_3(*source.getCoordinates())
#    x2, y2, z2 = project4_to_3(*target.getCoordinates())
#
#    xmin = min(xmin, x1, x2)
#    xmax = max(xmax, x1, x2)
#    ymin = min(ymin, y1, y2)
#    ymax = max(ymax, y1, y2)
#    zmin = min(zmin, z1, z2)
#    zmax = max(zmax, z1, z2)
#
#    print("Squared length: ", edge.getSquaredLength())
#    if edge.getSquaredLength() < 0:
#        timeEdges.append([(x1, y1, z1), (x2, y2, z2)])
#    else:
#        spaceEdges.append([(x1, y1, z1), (x2, y2, z2)])
#
#fig = plt.figure(figsize=(8, 8))
#ax = fig.add_subplot(111, projection="3d")
#timelike_lc = Line3DCollection(timeEdges, linewidths=.7, colors="blue")
#spacelike_lc = Line3DCollection(spaceEdges, linewidths=.7, colors="red")
#ax.add_collection(timelike_lc)
#ax.add_collection(spacelike_lc)
#
#ax.set_xlim(xmin - 1, xmax + 1)
#ax.set_ylim(ymin - 1, ymax + 1)
#ax.set_zlim(zmin - 1, zmax + 1)
#ax.set_xlabel("X")
#ax.set_ylabel("Y")
#ax.set_zlabel("Z")
#
#plt.show()
