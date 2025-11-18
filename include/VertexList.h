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
      vertexList.erase(toRemove->getId());
      add(toAdd);
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