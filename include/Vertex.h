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
// Created by Andrew Kelleher on 10/19/25.
//

#ifndef CASET_CASET_SRC_VERTEX_H_
#define CASET_CASET_SRC_VERTEX_H_

#include <pybind11/pybind11.h>
#include <vector>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include "Logger.h"
#include "EdgeKey.h"
#include "ForwardDeclarations.h"

namespace py = pybind11;

namespace caset {

enum class EdgeDirection {
    In,
    Out
};

///
/// Vertices in modern lattice gauge theory have different coupling parameters. We have to add them in for strong vs
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
        Vertex() noexcept {
            id = 1;
            CLOG(INFO_LEVEL, "Vertex was default-constructed.");
        }
        Vertex(const std::uint64_t id_, const std::vector<double> &coords) : id(id_), coordinates(coords) {
#ifdef CASET_DEBUG
            if (id_ == 0) throw std::runtime_error("Vertex had a 0 ID.");
#endif
        }
        explicit Vertex(const std::uint64_t id_) : id(id_) {
#ifdef CASET_DEBUG
            if (id_ == 0) throw std::runtime_error("Vertex had a 0 ID.");
#endif
        }

        std::uint64_t getId() const {
#if CASET_DEBUG
        if (id == 0) throw std::runtime_error("Vertex had a 0 ID.");
#endif

            return id;
        }

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
        [[nodiscard]] double getTime() const;

        bool operator==(const Vertex &vertex) const noexcept;

        std::vector<double> getCoordinates() const;

        void setCoordinates(const std::vector<double> &coords) noexcept;

        std::pair<EdgeRawPtr, bool> addOutEdge(Edge *edge) noexcept;
        std::pair<EdgeRawPtr, bool> addInEdge(Edge *edge) noexcept;
        void removeInEdge(Edge *edge) noexcept;
        void removeOutEdge(Edge *edge) noexcept;

        std::size_t degree() const noexcept;

        std::vector<std::shared_ptr<Simplex>> getSimplicesForPython() const;

        std::unordered_set<Edge *>
        getInEdges() const noexcept;

        std::unordered_set<Edge *>
        getOutEdges() const noexcept;

        std::unordered_set<Edge *>
        getEdges() const noexcept;

        std::pair<std::shared_ptr<EdgeKeySet>, std::shared_ptr<EdgeKeySet>>
        moveInEdgesTo(const std::shared_ptr<Vertex> &recipient);

        std::pair<std::shared_ptr<EdgeKeySet>, std::shared_ptr<EdgeKeySet>>
        moveOutEdgesTo(const std::shared_ptr<Vertex> &recipient);

        py::object
        moveInEdgesToForPython(const std::shared_ptr<Vertex> &vertex);

        py::object
        moveOutEdgesToForPython(const std::shared_ptr<Vertex> &vertex);

        std::pair<std::shared_ptr<EdgeKeySet>, std::shared_ptr<EdgeKeySet>>
        absorbInto(const std::shared_ptr<Vertex> &vertex);

        py::object
        moveEdgesToForPython(const std::shared_ptr<Vertex> &vertex);

        std::string toString() const noexcept;

        std::unordered_set<Simplex *> getSimplices() const noexcept;

        void addSimplex(Simplex *simplex);
        void removeSimplex(Simplex *simplex);

        void assertUnused() const;
        [[nodiscard]] bool const hasEdge(const Edge *edge) const;

    private:
        std::unordered_set<Edge *> outEdges{};
        std::unordered_set<Edge *> inEdges{};
        std::unordered_set<Simplex *> simplices{};
        std::uint64_t id;
        std::vector<double> coordinates{};

        /// Helper method for moving edges in either direction. Returns toUpdate, toDelete listing which EdgeKey (s)
        /// should be re-keyed or deleted in the EdgeList maintaining ownership for the Edge (s).
        std::pair<std::shared_ptr<EdgeKeySet>, std::shared_ptr<EdgeKeySet>>
        moveEdgesToImpl(const std::shared_ptr<Vertex> &recipient, EdgeDirection direction);
};

// VertexPtr, VertexPtrs, VertexPtrSet are now defined in ForwardDeclarations.h
using VertexIndexMap = std::unordered_map<IdType, std::size_t>;
using VertexIdMap = std::unordered_map<IdType, VertexPtr>;
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

template<>
struct equal_to<caset::Vertex> {
    size_t operator()(const caset::Vertex &a, const caset::Vertex &b) const noexcept {
        return a.getId() == b.getId();
    }
};

template<>
struct equal_to<std::shared_ptr<caset::Vertex> > {
    size_t operator()(const caset::VertexPtr &a, const caset::VertexPtr &b) const noexcept {
        return a->getId() == b->getId();
    }
};
}
#endif //CASET_CASET_SRC_VERTEX_H_

