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
// Created by andrew on 12/12/25.
//

#ifndef CASET_EDGEKEY_H
#define CASET_EDGEKEY_H

#include "mesh/ForwardDeclarations.h"
#include <unordered_set>
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace caset {
class EdgeKey {
  public:
    IdType first{0};
    IdType second{0};

    EdgeKey(IdType sourceId_, IdType targetId_);

    bool operator==(const EdgeKey &other) const;

    [[nodiscard]] std::uint64_t hash() const;
#ifdef CASET_VERBOSE
    std::string toString() const noexcept;
#else
    std::string toString() const noexcept;
#endif


    Fingerprint fingerprint;
};
} // caset

#endif //CASET_EDGEKEY_H
