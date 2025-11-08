//
// Created by Andrew Kelleher on 10/19/25.
//

#ifndef CASET_CASET_SRC_VERTEX_H_
#define CASET_CASET_SRC_VERTEX_H_

#include <array>

#include <vector>


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
        Vertex() noexcept {}
        explicit Vertex(const std::vector<double> &coords) noexcept : coordinates(coords) {}
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

        std::vector<std::shared_ptr<Edge>> getEdges() const noexcept {return edges;}
        std::vector<std::shared_ptr<Edge>> getInEdges() const noexcept {return inEdges;}
        std::vector<std::shared_ptr<Edge>> getOutEdges() const noexcept {return outEdges;}
    private:
        void addEdge(std::shared_ptr<Edge> edge) noexcept { edges.push_back(edge); }

        std::vector<double> coordinates;
        std::vector<std::shared_ptr<Edge> > outEdges;
        std::vector<std::shared_ptr<Edge> > inEdges;
        std::vector<std::shared_ptr<Edge>> edges;
};
}

#endif //CASET_CASET_SRC_VERTEX_H_

