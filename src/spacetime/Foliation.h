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
// Created by andrew on 12/20/25.
//

#ifndef CASET_FOLIATION_H
#define CASET_FOLIATION_H
#include <cstdint>

namespace caset {

/// # Foliation
///
/// When someone says "foliation" in the context of lattice models of gravity; they're generally referring to an
/// arrangment by which spatial slices of the lattice manifold are sandwiched between temporal slices. This is called a
/// "preferred" foliation, and is a means of enforcing causal structure. Without preferred foliation there's no
/// requirement between shapes of simplices that can be glued together.
enum class Foliation : std::uint8_t {
  NONE = 0,
  PREFERRED = 1
};
} // caset

#endif //CASET_FOLIATION_H