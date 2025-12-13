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

//
// Created by andrew on 10/23/25.
//

#ifndef CASET_VERTEXLIST_H
#define CASET_VERTEXLIST_H

#include <memory>
#include <vector>
#include <unordered_map>

#include "Vertex.h"

namespace caset {
class VertexList {
  public:
    bool contains(const IdType id) const noexcept;
    std::shared_ptr<Vertex> add(const IdType id);
    std::shared_ptr<Vertex> add(const IdType id, const std::vector<double> &coords);
    std::shared_ptr<Vertex> add(const std::shared_ptr<Vertex> &vertex);
    std::shared_ptr<Vertex> get(IdType id);
    std::size_t size() noexcept;
    std::vector<std::shared_ptr<Vertex>> toVector() const noexcept;
    void remove(const std::shared_ptr<Vertex> &vertex);
    void replace(const std::shared_ptr<Vertex> &toRemove, const std::shared_ptr<Vertex> &toAdd);
  private:
    std::unordered_map<std::uint64_t, std::shared_ptr<Vertex>> vertexList{};
};
} // caset

#endif //CASET_VERTEXLIST_H