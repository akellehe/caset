//
// Created by Andrew Kelleher on 10/19/25.
//

#ifndef CASET_CASET_SRC_VERTEX_H_
#define CASET_CASET_SRC_VERTEX_H_

#include <vector>
#include <memory>
#include <unordered_set>
#include "Edge.h"

namespace caset {
///
/// Vertexes in modern lattic gauge theory have different coupling parameters. We have to add them in for strong vs
/// weak forces, for example. If we can reproduce the quark spectrum with a homogenous coupling parameter then we've
/// established the Gold Standard. The strong force is not actually observable. Observables are gauge variant. If you
/// change your gauge then it changes what you observe. The EM vector potential is gauge invariant, so it cannot be
/// observed.
///
/// Quantum chromodynamics have different and paradoxical coupling parameters at different energy scales. The leading
/// theories about it are called "running coupling"
///
class Vertex : public std::enable_shared_from_this<Vertex> {
    public:
        Vertex() noexcept { id = 0; }
        Vertex(const std::uint64_t id_, const std::vector<double> &coords) noexcept : id(id_), coordinates(coords) {
        }
        Vertex(const std::uint64_t id_) noexcept : id(id_) {
        }

        const std::uint64_t getId() const noexcept { return id; }

        ///
        /// We still need to implement what time means in the context of higher dimensional spacetimes. It seems like a
        /// good idea to require users to specify dimensionality at compile-time, but maybe that's asking a little too
        /// much.
        ///
        /// Let's just call 'time' the Euclidean magnitude of the elements of the coordinate vector excluding the
        /// spatial elements.
        ///
        /// By convention this will be \f$ \sqrt{\sum_{i=0}^{i=N-3}x_i^2} \f$ for all coordinate vectors of 4 or more
        /// elements or just the absolute value of \f$ x_0 \f$ otherwise.
        /// @return
        [[nodiscard]] double getTime() const {
            if (coordinates.empty()) {
                return 0;
            }
            if (coordinates.size() == 1) {
                return std::abs(coordinates[0]);
            }
            if (coordinates.size() >= 4) {
                double sumOfSquares = 0;
                for (int i = 0; i < coordinates.size(); i++) {
                    sumOfSquares += coordinates[i] * coordinates[i];
                }
                return std::sqrt(sumOfSquares);
            }
            const std::string msg = "Invalid coordinate vector of length " + std::to_string(coordinates.size());
            throw std::out_of_range(msg);
        }

        bool operator==(const Vertex &vertex) const noexcept {
            return vertex.getId() == id;
        }

        std::vector<double> getCoordinates() const {
            if (coordinates.empty()) {
                throw new std::runtime_error("You requested coordinates for a vertex that is coordinate independent.");
            }
            return coordinates;
        }

        void setCoordinates(const std::vector<double> &coords) noexcept { coordinates = coords; }

        [[nodiscard]] std::pair<std::shared_ptr<Edge>, std::shared_ptr<Vertex> > moveTo(
            const std::shared_ptr<Vertex> &vertex) {
            if (outEdges.empty()) {
                throw new std::runtime_error("Cannot execute move; outEdges is empty!");
            }
            std::shared_ptr<Edge> edge = std::make_shared<Edge>(getId(), vertex->getId());
            if (!outEdges.contains(edge)) {
                throw new std::runtime_error("No edge to this vertex exists.");
            }
            return std::pair<std::shared_ptr<Edge>, std::shared_ptr<Vertex> >({
                *outEdges.find(edge), vertex
            });
        }

        void addInEdge(const std::shared_ptr<Edge> &edge) noexcept { inEdges.insert(edge); }
        void addOutEdge(const std::shared_ptr<Edge> &edge) noexcept { outEdges.insert(edge); }
        void removeInEdge(const std::shared_ptr<Edge> &edge) noexcept { inEdges.erase(edge); }
        void removeOutEdge(const std::shared_ptr<Edge> &edge) noexcept { outEdges.erase(edge); }

        std::unordered_set<std::shared_ptr<Edge>, EdgeHash, EdgeEq>
        getInEdges() const noexcept { return inEdges; }

        std::unordered_set<std::shared_ptr<Edge>, EdgeHash, EdgeEq>
        getOutEdges() const noexcept { return outEdges; }

        std::unordered_set<std::shared_ptr<Edge>, EdgeHash, EdgeEq>
        getEdges() const noexcept {
            std::unordered_set<std::shared_ptr<Edge>, EdgeHash, EdgeEq> edges;
            edges.reserve(inEdges.size() + outEdges.size());
            edges.insert(inEdges.begin(), inEdges.end());
            edges.insert(outEdges.begin(), outEdges.end());
            return edges;
        }

        std::string toString() const noexcept {
            return "<V" + std::to_string(id) + ">";
        }

    private:
        std::vector<double> coordinates{};
        std::unordered_set<std::shared_ptr<Edge>, EdgeHash, EdgeEq> outEdges{};
        std::unordered_set<std::shared_ptr<Edge>, EdgeHash, EdgeEq> inEdges{};
        std::uint64_t id;
};
}

namespace std {
template<>
struct hash<caset::Vertex> {
    size_t operator()(const caset::Vertex &vertex) const noexcept {
        return std::hash<std::uint64_t>{}(vertex.getId());
    }
};

template<>
struct hash<std::shared_ptr<caset::Vertex> > {
    size_t operator()(const std::shared_ptr<caset::Vertex> &vertex) const noexcept {
        return std::hash<std::uint64_t>{}(vertex->getId());
    }
};
}
#endif //CASET_CASET_SRC_VERTEX_H_

