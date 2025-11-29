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
    std::shared_ptr<Vertex> operator[](const std::uint64_t vertexId) {
      return vertexList[vertexId];
    }

    std::shared_ptr<Vertex> get(std::uint64_t id) {
      return vertexList[id];
    }

    std::shared_ptr<Vertex> add(const std::shared_ptr<Vertex> &vertex) noexcept {
      vertexList.insert_or_assign(vertex->getId(), vertex);
      return vertex;
    }

    bool contains(const std::uint64_t id) const noexcept {
      return vertexList.contains(id);
    }

    std::shared_ptr<Vertex> add(const std::uint64_t id, const std::vector<double> &coords) noexcept {
      if (vertexList.contains(id)) {
        return vertexList.at(id);
      }
      std::shared_ptr<Vertex> vertex = std::make_shared<Vertex>(id, coords);
      vertexList.insert_or_assign(id, vertex);
      return vertex;
    }

    std::shared_ptr<Vertex> add(const std::uint64_t id) noexcept {
      if (vertexList.contains(id)) {
        return vertexList.at(id);
      }
      std::shared_ptr<Vertex> vertex = std::make_shared<Vertex>(id);
      vertexList.insert_or_assign(id, vertex);
      return vertex;
    }

    void replace(const std::shared_ptr<Vertex> &toRemove, const std::shared_ptr<Vertex> &toAdd) noexcept {
      remove(toRemove);
      add(toAdd);
    }

    void remove(const std::shared_ptr<Vertex> &vertex) noexcept {
      vertexList.erase(vertex->getId());
    }

    std::size_t size() noexcept {
      return vertexList.size();
    }

    std::vector<std::shared_ptr<Vertex>> toVector() const noexcept {
      std::vector<std::shared_ptr<Vertex>> result;
      result.reserve(vertexList.size());
      for (const auto &[key, vertex] : vertexList) {
        result.push_back(vertex);
      }
      return result;
    }
  private:
    std::unordered_map<std::uint64_t, std::shared_ptr<Vertex>> vertexList{};
};
} // caset

#endif //CASET_VERTEXLIST_H