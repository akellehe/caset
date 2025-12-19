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

#ifndef CASET_EDGELIST_H
#define CASET_EDGELIST_H

#include <memory>
#include <vector>
#include <unordered_set>

#include "Edge.h"
#include "Logger.h"

namespace caset {
class EdgeList {
  public:
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] Edges toVector() const noexcept;
    EdgePtr add(const EdgePtr &edge);
    EdgePtr add(const VertexPtr &source, const VertexPtr &target);
    EdgePtr add(const VertexPtr &source, const VertexPtr &target, double squaredLength) noexcept;
    EdgePtr get(const std::uint64_t &fingerprint);
    void remove(const EdgePtr &edge) noexcept;
    void replace(const EdgePtr &toRemove, const EdgePtr &toAdd) noexcept;

  private:
    EdgePtrMap edgeList{};
    EdgePtr getOrInsert(const EdgePtr &edge);
};
} // caset

#endif //CASET_EDGELIST_H
