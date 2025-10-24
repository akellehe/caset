//
// Created by Andrew Kelleher on 10/19/25.
//

#ifndef CASET_CASET_SRC_VERTEX_H_
#define CASET_CASET_SRC_VERTEX_H_

#include <array>
#include <cstdint>

#pragma once
#if __cplusplus >= 202002L
  #include <span>
  using std::span;
#else
// Minimal span backport (read-only)
template <class T> class span {
    const T* p_; std::size_t n_;
    public:
        span(const T* p, std::size_t n) : p_(p), n_(n) {}
        const T* data() const noexcept { return p_; }
        std::size_t size() const noexcept { return n_; }
        const T& operator[](std::size_t i) const noexcept { return p_[i]; }
        const T* begin() const noexcept { return p_; }
        const T* end()   const noexcept { return p_ + n_; }
};
#endif

namespace caset {
template<int N>
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
        static_assert(N >= 1 && N <= 11, "Dimension N must be between 1 and 11");

        Vertex() noexcept : coordinates{} {}

        explicit Vertex(const std::array<double, N> &coords) noexcept : coordinates(coords) {}

        span<const double> getCoordinates() const noexcept { return {coordinates.data(), N}; }
        const double *data() const noexcept {return coordinates.data();}
        static constexpr int dimensions() noexcept { return N; }

    private:
        std::array<double, N> coordinates;
};
}

#endif //CASET_CASET_SRC_VERTEX_H_

