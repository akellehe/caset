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

#include "Simplex.h"

namespace caset {
std::vector<std::shared_ptr<Simplex> > Simplex::getFacets() noexcept {
  if (!facets.empty()) return facets;
  auto verts = getVertices();
  facets.reserve(verts.size());
  for (int skip = 0; skip < verts.size(); skip++) {
    std::vector<std::shared_ptr<Vertex> > faceVertices;
    faceVertices.reserve(verts.size());
    faceVertices.insert(faceVertices.end(), verts.begin(), verts.begin() + skip);
    faceVertices.insert(faceVertices.end(), verts.begin() + skip + 1, verts.end());
    std::shared_ptr<Simplex> facet = std::make_shared<Simplex>(faceVertices);
    facet->addCoface(shared_from_this());
    facets.push_back(facet);
  }
  return facets;
}
}
