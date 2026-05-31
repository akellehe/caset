// MIT License
// Copyright (c) 2025 Andrew Kelleher
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "spacetime/topologies/Topology.h"
#include <vector>
#include <memory>
#include <iostream>

#include "spacetime/Spacetime.h"
#include "mesh/Vertex.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;
Topology::~Topology() = default;
// std::vector<std::shared_ptr<Constraint> > Topology::getConstraints() {return {};}
void Topology::build(Spacetime *spacetime, int numSimplices) {
  std::cout << "Building Topology (base)" << std::endl;
}

void Topology::buildExplicit(
    Spacetime *spacetime, std::size_t numVertices,
    const std::vector<std::vector<std::uint64_t>> &topSimplices) {
  std::vector<VertexPtr> verts;
  verts.reserve(numVertices);
  // Coordinate-free vertices: the triangulation is purely combinatorial here.
  for (std::size_t i = 0; i < numVertices; ++i) {
    verts.push_back(spacetime->createVertex(static_cast<std::uint64_t>(i)));
  }
  for (const auto &simplex : topSimplices) {
    VertexPtrs sv;
    sv.reserve(simplex.size());
    for (auto id : simplex) sv.push_back(verts.at(static_cast<std::size_t>(id)));
    spacetime->createSimplex(sv);
  }
}
} // namespace tessera::spacetime
