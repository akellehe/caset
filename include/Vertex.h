//
// Created by Andrew Kelleher on 10/19/25.
//

#ifndef CASET_CASET_SRC_VERTEX_H_
#define CASET_CASET_SRC_VERTEX_H_

#include <vector>
#include <memory>

namespace caset {
class Edge;

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
class Vertex final {
    public:
        Vertex() noexcept {id = 0;}
        Vertex(const std::uint64_t id_, const std::vector<double> &coords) noexcept : id(id_), coordinates(coords) {}
        Vertex(const std::uint64_t id_) noexcept : id(id_) {}

        std::uint64_t getId() noexcept { return id; }

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

        std::vector<double> getCoordinates() const {
            if (coordinates.size() == 0) {
                throw new std::runtime_error("You requested coordinates for a vertex that is coordinate independent.");
            }
            return coordinates;
        }

        void addOutEdge(std::shared_ptr<Edge> edge) noexcept {
            outEdges.push_back(edge);
            edges.push_back(edge);
        }

        void addInEdge(std::shared_ptr<Edge> edge) noexcept {
            inEdges.push_back(edge);
            edges.push_back(edge);
        }

        std::vector<std::shared_ptr<Edge> > getEdges() const noexcept { return edges; }
        std::vector<std::shared_ptr<Edge> > getInEdges() const noexcept { return inEdges; }
        std::vector<std::shared_ptr<Edge> > getOutEdges() const noexcept { return outEdges; }

    private:
        void addEdge(std::shared_ptr<Edge> edge) noexcept { edges.push_back(edge); }
        std::vector<double> coordinates{};
        std::vector<std::shared_ptr<Edge> > outEdges{};
        std::vector<std::shared_ptr<Edge> > inEdges{};
        std::vector<std::shared_ptr<Edge> > edges{};
        std::uint64_t id;
};
}

#endif //CASET_CASET_SRC_VERTEX_H_

