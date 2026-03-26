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

#ifndef CASET_CYLINDER_H
#define CASET_CYLINDER_H

#include "Topology.h"

namespace caset {
class Spacetime;

/// # Cylindrical Topology \f$ \Sigma \times [0, T] \f$
///
/// Spatial slices have a closed topology \f$ \Sigma \f$ but time is non-periodic:
/// the manifold has the structure \f$ \Sigma \times [0, T] \f$ with open temporal
/// boundaries at \f$ t = 0 \f$ and \f$ t = T \f$.
///
/// This topology is useful for studying spacetimes with initial and final
/// spatial slices, analogous to the "no-boundary" proposals in quantum cosmology,
/// or for computing transition amplitudes between two spatial geometries.
///
/// The build creates layers by coning only in the forward time direction,
/// producing a monotonically increasing time structure.
///
class Cylinder : public Topology {
  public:
    /// Build a cylindrical triangulation with open time boundaries.
    ///
    /// Each layer grows by coning exterior facets forward in time only.
    /// The first and last time slices are boundary slices.
    ///
    /// @param spacetime The spacetime to populate
    /// @param numSimplices Target number of top-dimensional simplices
    void build(Spacetime *spacetime, int numSimplices) override;
};

} // caset

#endif //CASET_CYLINDER_H
