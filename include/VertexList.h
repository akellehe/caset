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
template<int D>
class VertexList {
  public:
    std::shared_ptr<Vertex<D>> operator[](const std::uint64_t vertexId) {
      return vertexList[vertexId];
    }

    std::shared_ptr<Vertex<D>> get(std::uint64_t id) {
      return vertexList[id];
    }

    std::shared_ptr<Vertex<D>> add(const std::shared_ptr<Vertex<D>> &vertex) noexcept {
      auto found = vertexList.find(vertex->getId());
      if (found != vertexList.end()) {
        CLOG(INFO_LEVEL, "Vertex ", std::to_string(vertex->getId()), " already exists!");
      }
      const auto &[it, inserted] = vertexList.insert_or_assign(vertex->getId(), vertex);
      if (!inserted) {
        CLOG(INFO_LEVEL, "Vertex ", std::to_string(vertex->getId()), " already exists!");
        return it->second;
      }
      return vertex;
    }

    bool contains(const std::uint64_t id) const noexcept {
      return vertexList.contains(id);
    }

    std::shared_ptr<Vertex<D>> add(const std::uint64_t id, const std::vector<double> &coords) noexcept {
      auto found = vertexList.find(id);
      if (found != vertexList.end()) {
        CLOG(INFO_LEVEL, "Vertex ", std::to_string(id), " already exists!");
        return found->second;
      }
      std::shared_ptr<Vertex<D>> vertex = std::make_shared<Vertex<D>>(id, coords);
      const auto &[it, inserted] = vertexList.insert_or_assign(id, vertex);
      if (!inserted) {
        CLOG(INFO_LEVEL, "Vertex ", std::to_string(id), " already exists!");
        return it->second;
      }
      return vertex;
    }

    std::shared_ptr<Vertex<D>> add(const std::uint64_t id) noexcept {
      return add(id, std::vector<double>{});
    }

    void replace(const std::shared_ptr<Vertex<D>> &toRemove, const std::shared_ptr<Vertex<D>> &toAdd) {
#if CASET_ASSERTIONS
      if (toAdd == nullptr) throw std::invalid_argument("Cannot remove a nullptr vertex");
      if (toRemove == nullptr) throw std::invalid_argument("Cannot remove a nullptr vertex");
#endif

      remove(toRemove);
      add(toAdd);
    }

    void remove(const std::shared_ptr<Vertex<D>> &vertex) noexcept {
      vertexList.erase(vertex->getId());
    }

    std::size_t size() noexcept {
      return vertexList.size();
    }

    std::vector<std::shared_ptr<Vertex<D>>> toVector() const noexcept {
      std::vector<std::shared_ptr<Vertex<D>>> result{};
      result.reserve(vertexList.size());
      for (const auto &[key, vertex] : vertexList) {
        result.push_back(vertex);
      }
      return result;
    }

    void reserve(std::size_t nSimplices) noexcept;
  private:
    std::unordered_map<std::uint64_t, std::shared_ptr<Vertex<D>>> vertexList{};
};

using VertexList4D = VertexList<4>;
} // caset

#endif //CASET_VERTEXLIST_H