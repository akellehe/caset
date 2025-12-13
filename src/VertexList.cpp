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


#include <memory>
#include <vector>
#include <unordered_map>

#include "Vertex.h"
#include "VertexList.h"
#include "Simplex.h"

namespace caset {
    std::shared_ptr<Vertex> VertexList::get(IdType id) {
      auto found = vertexList.find(id);
      if (found == vertexList.end()) {
        CLOG(WARN_LEVEL, "You searched for a vertex that did not exist.");
        return nullptr;
      }
      return found->second;
    }

    std::shared_ptr<Vertex> VertexList::add(const std::shared_ptr<Vertex> &vertex) {
      if (vertexList.contains(vertex->getId())) {
        return vertexList.at(vertex->getId());
      }
      auto [it, inserted] = vertexList.emplace(vertex->getId(), vertex);
#ifdef CASET_DEBUG
      if (!inserted) {
        throw std::runtime_error("You attempted to overwrite a vertex! (multi-threading issue?)");
      }
#endif
      return it->second;
    }

    bool VertexList::contains(const IdType id) const noexcept {
      return vertexList.contains(id);
    }

    std::shared_ptr<Vertex> VertexList::add(const IdType id, const std::vector<double> &coords) {
#ifdef CASET_DEBUG
if (id == 0) throw std::invalid_argument("Cannot add a 0 vertex");
#endif
      if (vertexList.contains(id)) {
        return vertexList.at(id);
      }
      std::shared_ptr<Vertex> vertex = std::make_shared<Vertex>(id, coords);
      auto [it, inserted] = vertexList.emplace(id, vertex);
#ifdef CASET_DEBUG
      if (!inserted) throw std::runtime_error("Failed to emplace a vertex!");
      if (vertex->getId() == 0) throw std::invalid_argument("You passed a non-zero ID but the vertex ended up having a 0-id.");
#endif
      return it->second;
    }

    std::shared_ptr<Vertex> VertexList::add(const IdType id) {
#ifdef CASET_DEBUG
      if (id == 0) throw std::invalid_argument("Cannot add a 0 vertex");
#endif
      if (vertexList.contains(id)) return vertexList.at(id);
      std::shared_ptr<Vertex> vertex = std::make_shared<Vertex>(id);
      auto [it, inserted] = vertexList.emplace(id, vertex);
#ifdef CASET_DEBUG
      if (!inserted) throw std::runtime_error("Failed to add a vertex!");
      if (vertex->getId() == 0) throw std::invalid_argument("You passed a non-zero ID but the vertex ended up having a 0-id.");
#endif
      return it->second;
    }

    void VertexList::replace(const std::shared_ptr<Vertex> &toRemove, const std::shared_ptr<Vertex> &toAdd) {
#if CASET_DEBUG
      if (toAdd == nullptr) throw std::invalid_argument("Cannot remove a nullptr vertex");
      if (toRemove == nullptr) throw std::invalid_argument("Cannot remove a nullptr vertex");
#endif

      remove(toRemove);
      add(toAdd);
    }

    void VertexList::remove(const std::shared_ptr<Vertex> &vertex) {
#ifdef CASET_DEBUG
      CLOG(INFO_LEVEL, "Erasing vertex: ", std::to_string(vertex->getId()));
      for (const auto &simplex : vertex->getSimplices()) {
        for (const auto &e : simplex->getEdges()) {
          if (e->hasVertex(vertex)) {
            throw std::runtime_error("Cannot remove a vertex from the VertexList that still has referents: " + simplex->toString() + " at edge " + e->toString());
          }
        }
        if (simplex->hasVertex(vertex)) {
          throw std::runtime_error("Cannot remove a vertex from the VertexList that still has referents: " + simplex->toString());
        }
      }
      for (const auto &e : vertex->getEdges()) {
        if (e->hasVertex(vertex)) {
          throw std::runtime_error("Cannot remove a vertex from the VertexList that still has referents: " + e->toString());
        }
      }
#endif
      vertexList.erase(vertex->getId());
    }

    std::size_t VertexList::size() noexcept {
      return vertexList.size();
    }

    std::vector<std::shared_ptr<Vertex>> VertexList::toVector() const noexcept {
      std::vector<std::shared_ptr<Vertex>> result{};
      result.reserve(vertexList.size());
      for (const auto &[key, vertex] : vertexList) {
        result.push_back(vertex);
      }
      return result;
    }
} // caset
