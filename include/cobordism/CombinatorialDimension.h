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

#ifndef TESSERA_COBORDISM_COMBINATORIALDIMENSION_H
#define TESSERA_COBORDISM_COMBINATORIALDIMENSION_H

#include <memory>

#include "observables/Observable.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::observables;
using namespace ::tessera::spacetime;

/// # CombinatorialDimension
///
/// Observable: the combinatorial dimension of a triangulation — the largest
/// \f$ k \f$ for which a \f$ k \f$-simplex is present (the maximum simplex
/// vertex count minus one), or \f$ -1 \f$ for the empty complex.
///
/// This is a purely combinatorial/topological quantity: an exact integer equal
/// to \f$ n \f$ for a clean PL \f$ n \f$-manifold triangulation, fixed by the
/// simplex set alone. It is **not** the spectral dimension
/// (``Spacetime::getSpectralDimensionOnSkeleton``), which is a real-valued,
/// scale-dependent diffusion quantity that can differ from \f$ n \f$
/// (dimensional reduction); nor the Hausdorff (volume-scaling) dimension. It is
/// also distinct from the metric's *declared* dimension: a hand-assembled
/// lower-dimensional triangulation (e.g. a surface \f$ S^2 \f$) can live inside
/// a 4D-signature ``Spacetime``.
///
/// The cobordism capabilities key their dimension-dependent logic off this
/// integer (the existence table is indexed by \f$ n \in \{0,1,2,3,4\} \f$, the
/// signature is defined only for \f$ n=4 \f$, vertex links must be
/// \f$ (n-1) \f$-spheres, …) — a role a real-valued spectral dimension cannot
/// fill.
///
/// Implemented as an :class:`Observable` (returning the dimension as a double)
/// following tessera's convention for scalar measurements of a triangulation.
/// The characteristic-number capabilities (Euler characteristic, signature, …)
/// are likewise Observables; multi-complex / structural operations (cobordism
/// verification, reconstruction, Pachner search) are static-only classes.
class CombinatorialDimension : public Observable {
  public:
    double compute(const std::shared_ptr<Spacetime> &spacetime) override;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_INTRINSICDIMENSION_H
