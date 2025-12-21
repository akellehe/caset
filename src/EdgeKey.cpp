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

#include "EdgeKey.h"
#include "ForwardDeclarations.h"
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace caset {


    EdgeKey::EdgeKey(IdType sourceId_, IdType targetId_) : first(sourceId_), second(targetId_), fingerprint({sourceId_, targetId_}) {
#if CASET_ASSERTIONS
      if (sourceId_ == targetId_) throw std::runtime_error("You can't create a self-reference.");
#endif
      fingerprint = Fingerprint(fingerprint);
    }

    bool EdgeKey::operator==(const EdgeKey &other) const {
      return first == other.first && second == other.second;
    }

    [[nodiscard]] std::uint64_t EdgeKey::hash() const {
      std::size_t h1 = std::hash<IdType>{}(first);
      std::size_t h2 = std::hash<IdType>{}(second);
      return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }

#ifdef CASET_VERBOSE
    std::string EdgeKey::toString() const noexcept {
      return "(" + std::to_string(first) + ", " + std::to_string(second) + ")";
    }
#else
std::string constexpr EdgeKey::toString() const noexcept {
      return "";
    }
#endif


} // caset
