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
#include <stdexcept>

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

int Topology::dimension() const {
  // Only the fixed-triangulation fixtures have an intrinsic manifold dimension;
  // the dimension-parametric CDT topologies (Toroid, Sphere, Cylinder) take
  // theirs from the spacetime's signature at build() time and so do not
  // override this.
  throw std::logic_error(
      "Topology::dimension(): this topology has no intrinsic manifold dimension "
      "(it is parametric in the spacetime signature). Build it with an explicit "
      "Signature(d, …) instead of deriving d from the topology.");
}

void Topology::buildExplicit(
    Spacetime *spacetime, std::size_t numVertices,
    const std::vector<std::vector<std::uint64_t>> &topSimplices) {
  std::vector<VertexPtr> verts;
  verts.reserve(numVertices);
  // Coordinate-free vertices: the triangulation is purely combinatorial
  // here.  Allocate through the id-counter path (not the explicit-id
  // overload) so the counter advances past these ids — on a fresh
  // spacetime this still assigns 0..numVertices-1, but it leaves the
  // counter able to hand out collision-free ids to anything that later
  // grows the complex (e.g. a pre-geometric Pachner add move).
  for (std::size_t i = 0; i < numVertices; ++i) {
    verts.push_back(spacetime->createVertex());
  }
  for (const auto &simplex : topSimplices) {
    VertexPtrs sv;
    sv.reserve(simplex.size());
    for (auto id : simplex) sv.push_back(verts.at(static_cast<std::size_t>(id)));
    spacetime->createSimplex(sv);
  }
}
} // namespace tessera::spacetime
