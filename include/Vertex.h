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

#include <vector>
#include <memory>
#include <unordered_set>
#include "ForwardDeclarations.h"
#include "Fingerprint.h"

namespace caset {
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
        Vertex() noexcept;
        Vertex(const std::uint64_t id_, const std::vector<double> &coords) noexcept;
        explicit Vertex(const std::uint64_t id_) noexcept;

        std::uint64_t getId() const noexcept;

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

        void checkDuplicates(std::string msg) const;
        EdgePtr getEdge(const EdgePtr &edge);
        EdgePtrSet getEdges() const noexcept;
        EdgePtrSet getInEdges() const noexcept;
        EdgePtrSet getOutEdges() const noexcept;
        SimplexPtrSet getSimplices() const noexcept;
        SimplexPtrSet removeInEdge(const EdgePtr &edge) noexcept;
        SimplexPtrSet removeOutEdge(const EdgePtr &edge) noexcept;
        bool addSimplex(const SimplexPtr &simplex);
        bool operator==(const Vertex &vertex) const noexcept;
        bool removeSimplex(const SimplexPtr &simplex);
        std::pair<EdgePtrSet, EdgePtrSet> moveEdgesTo(const VertexPtr &vertex, Spacetime *spacetime);
        std::pair<EdgePtrSet, EdgePtrSet> moveInEdgesTo(const VertexPtr &vertex, Spacetime *spacetime);
        std::pair<EdgePtrSet, EdgePtrSet> moveOutEdgesTo(const VertexPtr &vertex, Spacetime *spacetime);
        std::size_t degree() const noexcept;
        std::string toString() const noexcept;
        std::vector<double> getCoordinates() const;
        void addInEdge(const EdgePtr &edge) noexcept;
        void addOutEdge(const EdgePtr &edge) noexcept;
        void setCoordinates(const std::vector<double> &coords) noexcept;

        Fingerprint fingerprint;

    private:
        enum class EdgeDirection { In, Out };

        std::pair<EdgePtrSet, EdgePtrSet>
        moveEdgesToImpl(const VertexPtr &recipient, Spacetime *spacetime, EdgeDirection direction);

        EdgePtrSet outEdges{};
        EdgePtrSet inEdges{};
        SimplexPtrSet simplices{};
        std::uint64_t id;
        std::vector<double> coordinates{};
};

// using VertexPtr = VertexPtr;
// using VertexIndexMap = std::unordered_map<IdType, std::size_t>;
// using VertexIdMap = std::unordered_map<IdType, VertexPtr>;
// using VertexSet = std::unordered_set<VertexPtr>;
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

