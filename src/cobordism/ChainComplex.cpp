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

#include "cobordism/ChainComplex.h"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cobordism/IntegerLinalg.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

using Face = std::vector<std::uint64_t>;  // sorted vertex ids

namespace {
// Sorted vertex ids of a simplex — the homological reference ordering.
Face sortedIds(const SimplexPtr &s) {
  Face ids;
  for (const auto &v : s->getVertices()) ids.push_back(v->getId());
  std::sort(ids.begin(), ids.end());
  return ids;
}
}  // namespace

ChainComplex ChainComplex::fromSpacetime(const Spacetime &K) {
  ChainComplex cc;

  // Collect the face lattice through the mesh's own facet operation
  // (Simplex::getFacets) — a BFS down from the registered simplices,
  // de-duplicated by fingerprint and bucketed by dimension. We do NOT
  // re-derive faces; getFacets is the single source of truth for "the faces of
  // a simplex". The only thing ChainComplex adds is the homological boundary
  // sign below, which mesh facets don't carry.
  std::map<int, std::vector<SimplexPtr>> byDim;
  std::unordered_set<std::uint64_t> seen;
  std::vector<SimplexPtr> stack(K.getSimplices().begin(), K.getSimplices().end());
  while (!stack.empty()) {
    SimplexPtr s = stack.back();
    stack.pop_back();
    if (s == nullptr) continue;
    if (!seen.insert(s->fingerprint.fingerprint()).second) continue;
    const int k = static_cast<int>(s->size()) - 1;
    byDim[k].push_back(s);
    if (k >= 1)
      for (const auto &f : s->getFacets()) stack.push_back(f);
  }

  if (byDim.empty()) return cc;  // empty complex
  const int n = byDim.rbegin()->first;
  cc.dimension_ = n;

  // Order each dimension deterministically (by sorted vertex ids) and index
  // the simplices by fingerprint for boundary lookups.
  std::vector<std::vector<SimplexPtr>> faces(n + 1);
  std::vector<std::unordered_map<std::uint64_t, int>> index(n + 1);
  cc.counts_.assign(n + 1, 0);
  cc.faceVerts_.assign(n + 1, {});
  for (int k = 0; k <= n; ++k) {
    auto &vec = byDim[k];
    std::sort(vec.begin(), vec.end(), [](const SimplexPtr &a, const SimplexPtr &b) {
      return sortedIds(a) < sortedIds(b);
    });
    faces[k] = vec;
    cc.counts_[k] = vec.size();
    cc.faceVerts_[k].reserve(vec.size());
    for (int j = 0; j < static_cast<int>(vec.size()); ++j) {
      index[k][vec[j]->fingerprint.fingerprint()] = j;
      cc.faceVerts_[k].push_back(sortedIds(vec[j]));
    }
  }

  // Boundary ∂_k (rows = |C_{k-1}|, cols = |C_k|): each column is a k-simplex,
  // its nonzero rows are its facets, and the orientation is already carried by
  // getFacets()'s canonical order — facet at index i is the i-th vertex
  // dropped, so its coefficient is (-1)^i (see Simplex::getFacets). We read it
  // off the index rather than recomputing any sign.
  cc.boundary_.assign(n + 1, {});
  for (int k = 1; k <= n; ++k) {
    const int rows = static_cast<int>(cc.counts_[k - 1]);
    const int cols = static_cast<int>(cc.counts_[k]);
    std::vector<long> M(static_cast<std::size_t>(rows) * cols, 0);
    for (int j = 0; j < cols; ++j) {
      const auto &facets = faces[k][j]->getFacets();
      for (int i = 0; i < static_cast<int>(facets.size()); ++i) {
        const int r = index[k - 1].at(facets[i]->fingerprint.fingerprint());
        M[static_cast<std::size_t>(r) * cols + j] = (i % 2 == 0) ? 1 : -1;
      }
    }
    cc.boundary_[k] = std::move(M);
  }
  return cc;
}

std::size_t ChainComplex::numSimplices(int k) const noexcept {
  if (k < 0 || k > dimension_) return 0;
  return counts_[static_cast<std::size_t>(k)];
}

int ChainComplex::eulerCharacteristic() const noexcept {
  int chi = 0;
  for (int k = 0; k <= dimension_; ++k)
    chi += (k % 2 == 0 ? 1 : -1) * static_cast<int>(counts_[static_cast<std::size_t>(k)]);
  return chi;
}

const std::vector<long> &ChainComplex::boundaryMatrix(int k) const {
  static const std::vector<long> kEmpty{};
  if (k < 0 || k > dimension_) return kEmpty;
  return boundary_[static_cast<std::size_t>(k)];
}

bool ChainComplex::boundaryComposesToZero() const {
  // ∂_{k-1} ∘ ∂_k = 0 : (|C_{k-2}| x |C_{k-1}|) · (|C_{k-1}| x |C_k|).
  for (int k = 2; k <= dimension_; ++k) {
    const int a = static_cast<int>(counts_[k - 2]);  // rows of ∂_{k-1}
    const int b = static_cast<int>(counts_[k - 1]);  // shared dim
    const int c = static_cast<int>(counts_[k]);      // cols of ∂_k
    const auto &L = boundary_[static_cast<std::size_t>(k - 1)];
    const auto &R = boundary_[static_cast<std::size_t>(k)];
    for (int i = 0; i < a; ++i)
      for (int j = 0; j < c; ++j) {
        long acc = 0;
        for (int m = 0; m < b; ++m)
          acc += L[static_cast<std::size_t>(i) * b + m] * R[static_cast<std::size_t>(m) * c + j];
        if (acc != 0) return false;
      }
  }
  return true;
}

int ChainComplex::rankOfBoundary(int k) const {
  if (k < 1 || k > dimension_) return 0;
  return integerRank(boundary_[static_cast<std::size_t>(k)],
                     static_cast<int>(counts_[k - 1]), static_cast<int>(counts_[k]));
}

int ChainComplex::gf2RankOfBoundary(int k) const {
  if (k < 1 || k > dimension_) return 0;
  const auto &M = boundary_[static_cast<std::size_t>(k)];
  std::vector<int> bits(M.size());
  for (std::size_t i = 0; i < M.size(); ++i) bits[i] = static_cast<int>(M[i] & 1);
  return gf2Rank(std::move(bits), static_cast<int>(counts_[k - 1]),
                 static_cast<int>(counts_[k]));
}

std::vector<int> ChainComplex::bettiNumbers() const {
  std::vector<int> b;
  if (dimension_ < 0) return b;
  b.assign(dimension_ + 1, 0);
  for (int k = 0; k <= dimension_; ++k)
    b[k] = static_cast<int>(counts_[k]) - rankOfBoundary(k) - rankOfBoundary(k + 1);
  return b;
}

std::vector<int> ChainComplex::bettiNumbersGF2() const {
  std::vector<int> b;
  if (dimension_ < 0) return b;
  b.assign(dimension_ + 1, 0);
  for (int k = 0; k <= dimension_; ++k)
    b[k] = static_cast<int>(counts_[k]) - gf2RankOfBoundary(k) - gf2RankOfBoundary(k + 1);
  return b;
}

// The intersection form records how the two-dimensional surfaces sitting
// inside a four-dimensional manifold cross one another: given two such
// surfaces it returns an integer counting their (signed) crossing points. We
// compute it the standard algebraic-topology way, which needs no geometry:
//
//   1. Find the manifold's independent two-dimensional surfaces. Working with
//      "cochains" (a number assigned to each triangle), these are the *closed*
//      cochains that are not *exact*; one representative per two-dimensional
//      hole gives a basis of the relevant cohomology.
//   2. Pair them with the cup product (the Alexander-Whitney recipe): on a
//      four-simplex with vertices v0<v1<v2<v3<v4, the product of two such
//      cochains evaluates the first on the front triangle (v0,v1,v2) and the
//      second on the back triangle (v2,v3,v4).
//   3. Sum those products over the whole manifold, with each four-simplex
//      weighted by its orientation (+/-1, from the "fundamental class"). The
//      result is the symmetric crossing-number matrix.
std::vector<double> ChainComplex::intersectionForm() const {
  if (dimension_ != 4) return {};
  const int numTwoDimensionalHoles = bettiNumbers()[2];  // rank of H_2
  if (numTwoDimensionalHoles == 0) return {};

  const int numEdges = static_cast<int>(counts_[1]);
  const int numTriangles = static_cast<int>(counts_[2]);
  const int numTetrahedra = static_cast<int>(counts_[3]);
  const int numFourSimplices = static_cast<int>(counts_[4]);

  auto boundaryMatrixAsEigen = [&](int k, int rows, int cols) {
    Eigen::MatrixXd matrix(rows, cols);
    const auto &flat = boundary_[static_cast<std::size_t>(k)];
    for (int row = 0; row < rows; ++row)
      for (int col = 0; col < cols; ++col)
        matrix(row, col) =
            static_cast<double>(flat[static_cast<std::size_t>(row) * cols + col]);
    return matrix;
  };
  // Boundary maps: each sends a cell to the (signed) sum of its faces.
  const Eigen::MatrixXd triangleBoundaries =
      boundaryMatrixAsEigen(2, numEdges, numTriangles);          // triangles -> edges
  const Eigen::MatrixXd tetrahedronBoundaries =
      boundaryMatrixAsEigen(3, numTriangles, numTetrahedra);     // tetrahedra -> triangles
  const Eigen::MatrixXd fourSimplexBoundaries =
      boundaryMatrixAsEigen(4, numTetrahedra, numFourSimplices); // 4-simplices -> tetrahedra

  // Fundamental class: the single way (up to sign) to orient all the
  // four-simplices coherently so their boundaries cancel. Algebraically it is
  // the one-dimensional null space of the four-simplex boundary map; its
  // entries are +/-1, one orientation per four-simplex. A closed orientable
  // 4-manifold has exactly this; anything else has no fundamental class and no
  // well-defined signature.
  const Eigen::MatrixXd topCycles =
      Eigen::FullPivLU<Eigen::MatrixXd>(fourSimplexBoundaries).kernel();
  if (topCycles.cols() != 1)
    throw std::runtime_error(
        "ChainComplex::intersectionForm: a closed orientable 4-manifold is "
        "required (the space of top-dimensional cycles is not 1-dimensional, so "
        "there is no fundamental class)");
  Eigen::VectorXd orientationPerFourSimplex = topCycles.col(0);
  int largestEntry = 0;
  for (int i = 1; i < orientationPerFourSimplex.size(); ++i)
    if (std::abs(orientationPerFourSimplex[i]) >
        std::abs(orientationPerFourSimplex[largestEntry]))
      largestEntry = i;
  orientationPerFourSimplex /= orientationPerFourSimplex[largestEntry];  // -> entries +/-1

  // Two-dimensional cohomology classes as triangle-cochains:
  //  - "closed" cochains are the null space of the transposed tetrahedron
  //    boundary map (the coboundary operator on triangle-cochains);
  //  - "exact" cochains are the columns of the transposed triangle boundary map.
  // A basis of cohomology is a set of closed cochains that stay independent
  // after the exact ones are accounted for.
  const Eigen::MatrixXd closedTriangleCochains =
      Eigen::FullPivLU<Eigen::MatrixXd>(tetrahedronBoundaries.transpose()).kernel();
  const Eigen::MatrixXd exactTriangleCochains = triangleBoundaries.transpose();

  const double zeroTolerance = 1e-9;
  auto numericalRank = [&](const Eigen::MatrixXd &matrix) {
    if (matrix.cols() == 0) return 0;
    Eigen::FullPivLU<Eigen::MatrixXd> decomposition(matrix);
    decomposition.setThreshold(zeroTolerance);
    return static_cast<int>(decomposition.rank());
  };
  Eigen::MatrixXd spannedSoFar = exactTriangleCochains;
  int spannedRank = numericalRank(spannedSoFar);
  std::vector<Eigen::VectorXd> cohomologyBasis;
  for (int j = 0; j < closedTriangleCochains.cols() &&
                  static_cast<int>(cohomologyBasis.size()) < numTwoDimensionalHoles;
       ++j) {
    Eigen::MatrixXd augmented(numTriangles, spannedSoFar.cols() + 1);
    if (spannedSoFar.cols() > 0) augmented.leftCols(spannedSoFar.cols()) = spannedSoFar;
    augmented.col(spannedSoFar.cols()) = closedTriangleCochains.col(j);
    if (numericalRank(augmented) > spannedRank) {  // genuinely new cohomology class
      cohomologyBasis.push_back(closedTriangleCochains.col(j));
      spannedSoFar = augmented;
      ++spannedRank;
    }
  }

  // Look up a triangle's index from its (sorted) three vertices, for the
  // cup-product front/back faces below.
  std::map<std::array<std::uint64_t, 3>, int> triangleIndexByVertices;
  for (int j = 0; j < numTriangles; ++j) {
    const auto &vertices = faceVerts_[2][static_cast<std::size_t>(j)];
    triangleIndexByVertices[{vertices[0], vertices[1], vertices[2]}] = j;
  }

  // Cup product summed over the oriented manifold (step 2 + 3 above).
  const int numClasses = static_cast<int>(cohomologyBasis.size());
  std::vector<double> intersectionMatrix(
      static_cast<std::size_t>(numClasses) * numClasses, 0.0);
  for (int s = 0; s < numFourSimplices; ++s) {
    const auto &vertices = faceVerts_[4][static_cast<std::size_t>(s)];
    const int frontTriangle =
        triangleIndexByVertices.at({vertices[0], vertices[1], vertices[2]});
    const int backTriangle =
        triangleIndexByVertices.at({vertices[2], vertices[3], vertices[4]});
    const double orientation = orientationPerFourSimplex[s];
    for (int a = 0; a < numClasses; ++a)
      for (int b = 0; b < numClasses; ++b)
        intersectionMatrix[static_cast<std::size_t>(a) * numClasses + b] +=
            orientation * cohomologyBasis[a][frontTriangle] *
            cohomologyBasis[b][backTriangle];
  }
  // The crossing pairing is symmetric; average away any numerical asymmetry.
  for (int a = 0; a < numClasses; ++a)
    for (int b = a + 1; b < numClasses; ++b) {
      const double mean =
          0.5 * (intersectionMatrix[static_cast<std::size_t>(a) * numClasses + b] +
                 intersectionMatrix[static_cast<std::size_t>(b) * numClasses + a]);
      intersectionMatrix[static_cast<std::size_t>(a) * numClasses + b] = mean;
      intersectionMatrix[static_cast<std::size_t>(b) * numClasses + a] = mean;
    }
  return intersectionMatrix;
}

int ChainComplex::signature() const {
  const std::vector<double> intersectionMatrix = intersectionForm();
  if (intersectionMatrix.empty()) return 0;
  const int size = static_cast<int>(
      std::lround(std::sqrt(static_cast<double>(intersectionMatrix.size()))));
  Eigen::MatrixXd form(size, size);
  for (int row = 0; row < size; ++row)
    for (int col = 0; col < size; ++col)
      form(row, col) = intersectionMatrix[static_cast<std::size_t>(row) * size + col];
  // Signature = (number of positive eigenvalues) - (number of negative ones).
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(form, Eigen::EigenvaluesOnly);
  double largestMagnitude = 0.0;
  for (int i = 0; i < size; ++i)
    largestMagnitude = std::max(largestMagnitude, std::abs(solver.eigenvalues()[i]));
  // Relative threshold for "nonzero": the form is nondegenerate (unimodular) on
  // a closed 4-manifold, so its eigenvalues sit well away from zero.
  const double zeroTolerance = 1e-7 * (largestMagnitude > 0 ? largestMagnitude : 1.0);
  int numPositive = 0, numNegative = 0;
  for (int i = 0; i < size; ++i) {
    const double eigenvalue = solver.eigenvalues()[i];
    if (eigenvalue > zeroTolerance) ++numPositive;
    else if (eigenvalue < -zeroTolerance) ++numNegative;
  }
  return numPositive - numNegative;
}

std::vector<long> ChainComplex::torsion(int k) const {
  std::vector<long> out;
  if (k < 0 || k + 1 > dimension_) return out;  // torsion of H_k comes from ∂_{k+1}
  const int kk = k + 1;
  auto snf = smithNormalForm(boundary_[static_cast<std::size_t>(kk)],
                             static_cast<int>(counts_[kk - 1]),
                             static_cast<int>(counts_[kk]));
  for (long d : snf.invariantFactors)
    if (d > 1) out.push_back(d);
  return out;
}

}  // namespace tessera::cobordism
