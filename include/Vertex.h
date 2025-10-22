//
// Created by Andrew Kelleher on 10/19/25.
//

#ifndef CASET_CASET_SRC_VERTEX_H_
#define CASET_CASET_SRC_VERTEX_H_

#include <array>
#include <cstdint>

#include "Signature.h"

#pragma once
#include <array>
#include <cstdint>
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

