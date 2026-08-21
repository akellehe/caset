// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/ChainComplex.h"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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
  // Seed the downward BFS from the top-dimensional cells only.  Genuine
  // lower-dimensional faces are reached through getFacets below; starting
  // from *every* registered simplex would also pull in orphaned
  // sub-simplices that the mesh creates lazily (Simplex::getFacets) and
  // never garbage-collects once their cofaces are removed — e.g. the
  // shared facet a 2->3 Pachner flip deletes, or any facet materialised
  // by a previous fromSpacetime() call whose top cell a later move
  // removed.  Such orphans are faces of no current top cell and would
  // corrupt the chain groups (spurious cycles, even negative Betti
  // numbers).  For a pure complex — every manifold/fixture here — the top
  // cells' face-closure is exactly the chain complex, so on a freshly
  // built complex this seeds the identical simplex set as before.
  std::size_t topSize = 0;
  for (const auto &s : K.getSimplices())
    if (s != nullptr) topSize = std::max(topSize, static_cast<std::size_t>(s->size()));
  std::vector<SimplexPtr> stack;
  for (const auto &s : K.getSimplices())
    if (s != nullptr && s->size() == topSize) stack.push_back(s);
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

std::vector<std::vector<std::uint64_t>> ChainComplex::kSimplexVertices(int k) const {
  if (k < 0 || k > dimension_) return {};  // out of range: no such simplices
  return faceVerts_[static_cast<std::size_t>(k)];
}

std::vector<std::vector<std::uint64_t>> ChainComplex::orientedTopSimplices() const {
  return kSimplexVertices(dimension_);  // empty complex (d < 0) yields {}
}

std::vector<int> ChainComplex::fundamentalClass() const {
  // [W] ∈ H_d is the ±1 generator of ker ∂_d: the orientation each top simplex
  // must carry (relative to its increasing-vertex reference orientation) so the
  // top chain Σ_t ε_t·t is a cycle. For a closed connected oriented d-manifold
  // this kernel is one-dimensional (b_d = 1), so the generator is unique up to
  // an overall sign.
  const int d = dimension_;
  if (d < 1)
    throw std::runtime_error(
        "ChainComplex::fundamentalClass: a closed oriented manifold of "
        "dimension >= 1 is required");
  const int rows = static_cast<int>(counts_[static_cast<std::size_t>(d - 1)]);
  const int cols = static_cast<int>(counts_[static_cast<std::size_t>(d)]);
  Eigen::MatrixXd topBoundary(rows, cols);
  const auto &flat = boundary_[static_cast<std::size_t>(d)];
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c)
      topBoundary(r, c) =
          static_cast<double>(flat[static_cast<std::size_t>(r) * cols + c]);

  // Ask the decomposition for the genuine nullity rather than reading
  // kernel().cols(): Eigen's FullPivLU::kernel() returns a single all-zero
  // column for a 0-dimensional kernel (it never hands back a zero-*column*
  // matrix), so kernel().cols() is always ≥ 1 and cannot tell b_d = 0 (every
  // ball SolidSimplex(n), ℝP²) apart from b_d = 1. dimensionOfKernel() = cols −
  // rank reports the true dim ker ∂_d, so the documented contract — a
  // fundamental class exists only when dim ker ∂_d = 1 — is actually enforced.
  const Eigen::FullPivLU<Eigen::MatrixXd> decomposition(topBoundary);
  if (decomposition.dimensionOfKernel() != 1)
    throw std::runtime_error(
        "ChainComplex::fundamentalClass: a closed connected oriented " +
        std::to_string(d) + "-manifold is required (dim ker ∂_" +
        std::to_string(d) + " = b_" + std::to_string(d) +
        " must be 1, so the fundamental class is unique up to sign)");
  const Eigen::MatrixXd kernel = decomposition.kernel();

  // Every entry of this generator has the same magnitude (one orientation per
  // top simplex), so scaling by the first nonzero entry makes the entries
  // exactly ±1; fixing that entry to +1 makes the overall sign deterministic.
  // dim ker ∂_d = 1 guarantees a genuine (nonzero) generator, so firstNonzero
  // lands on a real entry; the size() guard keeps sign normalization from ever
  // indexing past the end even if the generator were numerically zero.
  Eigen::VectorXd generator = kernel.col(0);
  const double scale = generator.cwiseAbs().maxCoeff();
  const double threshold = 1e-9 * (scale > 0.0 ? scale : 1.0);
  int firstNonzero = 0;
  while (firstNonzero < generator.size() &&
         std::abs(generator[firstNonzero]) <= threshold)
    ++firstNonzero;
  if (firstNonzero == generator.size())
    throw std::runtime_error(
        "ChainComplex::fundamentalClass: the generator of ker ∂_" +
        std::to_string(d) + " is numerically zero (no fundamental class)");
  generator /= generator[firstNonzero];

  std::vector<int> epsilon(static_cast<std::size_t>(cols), 0);
  for (int i = 0; i < generator.size(); ++i)
    epsilon[static_cast<std::size_t>(i)] =
        static_cast<int>(std::lround(generator[i]));
  return epsilon;
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

  // Fundamental class: the single way (up to sign) to orient all the
  // four-simplices coherently so their boundaries cancel — the ±1 generator of
  // ker ∂_4, one orientation per four-simplex (see fundamentalClass()). A closed
  // orientable 4-manifold has exactly this; anything else has no fundamental
  // class and no well-defined signature, and the call throws.
  const std::vector<int> orientationPerFourSimplex = fundamentalClass();

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
    const double orientation =
        static_cast<double>(orientationPerFourSimplex[static_cast<std::size_t>(s)]);
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

// ===========================================================================
// Stiefel–Whitney numbers (mod-2 characteristic numbers)
// ===========================================================================
//
// These are read off the mod-2 cohomology ring. The pipeline is:
//
//   1. Mod-2 cohomology H^k(K; Z/2): cocycles modulo coboundaries, with the
//      coboundary operator being the transpose of the (mod-2) boundary map.
//   2. Cup product on cochains via the Alexander–Whitney recipe — the same
//      front/back-face rule used by the integral intersection form, now mod 2
//      and for arbitrary degrees.
//   3. Wu classes v_k, defined by <v_k ∪ x, [K]> = <Sq^k x, [K]> for every
//      x in H^{n-k}. Solving this small linear system (its matrix is the
//      nondegenerate Poincaré-duality pairing) gives each v_k.
//   4. The total Stiefel–Whitney class w = Sq(v); then each Stiefel–Whitney
//      number is a degree-n monomial in the w_i evaluated on the fundamental
//      class [K] (the mod-2 sum of all top simplices).
//
// Only the Steenrod squares expressible through the ordinary cup product are
// implemented (Sq^k on a degree-k class is the cup square; Sq^k on a lower
// degree class is zero). The general Sq^i needs higher cup-i products and is
// deferred (#65); a class that genuinely requires it raises an exception.
namespace {

using Gf2Vector = std::vector<std::uint8_t>;  // dense vector over GF(2), entries 0/1
using Gf2Matrix = std::vector<Gf2Vector>;     // a list of rows, each the same length

// Reduce `rows` to reduced row-echelon form in place. Returns the pivot column
// of each surviving (nonzero) row; its size is the rank.
std::vector<int> gf2ReduceRows(Gf2Matrix &rows, int numColumns) {
  std::vector<int> pivotColumns;
  int pivotRow = 0;
  for (int column = 0; column < numColumns && pivotRow < static_cast<int>(rows.size());
       ++column) {
    int found = -1;
    for (int r = pivotRow; r < static_cast<int>(rows.size()); ++r)
      if (rows[r][column]) { found = r; break; }
    if (found < 0) continue;
    std::swap(rows[pivotRow], rows[found]);
    for (int r = 0; r < static_cast<int>(rows.size()); ++r)
      if (r != pivotRow && rows[r][column])
        for (int c = 0; c < numColumns; ++c) rows[r][c] ^= rows[pivotRow][c];
    pivotColumns.push_back(column);
    ++pivotRow;
  }
  return pivotColumns;
}

// Basis of the null space {x : matrix·x = 0} of a GF(2) matrix with `numColumns`
// columns (rows may be empty, in which case every standard basis vector is a
// kernel vector).
Gf2Matrix gf2Kernel(Gf2Matrix matrix, int numColumns) {
  const std::vector<int> pivotColumns = gf2ReduceRows(matrix, numColumns);
  std::vector<char> isPivot(numColumns, 0);
  for (int p : pivotColumns) isPivot[p] = 1;
  Gf2Matrix basis;
  for (int freeColumn = 0; freeColumn < numColumns; ++freeColumn) {
    if (isPivot[freeColumn]) continue;
    Gf2Vector x(numColumns, 0);
    x[freeColumn] = 1;
    for (int t = 0; t < static_cast<int>(pivotColumns.size()); ++t)
      x[pivotColumns[t]] = matrix[t][freeColumn];
    basis.push_back(std::move(x));
  }
  return basis;
}

// Incrementally maintained spanning set (kept in echelon form). add() returns
// true iff `candidate` was linearly independent of everything added so far
// (and then records it). Used to split cocycles into cohomology classes.
struct Gf2Span {
  Gf2Matrix echelonRows;
  std::vector<int> leadingColumn;
  int numColumns;
  explicit Gf2Span(int columns) : numColumns(columns) {}
  bool add(Gf2Vector candidate) {
    for (int t = 0; t < static_cast<int>(echelonRows.size()); ++t)
      if (candidate[leadingColumn[t]])
        for (int c = 0; c < numColumns; ++c) candidate[c] ^= echelonRows[t][c];
    int lead = -1;
    for (int c = 0; c < numColumns; ++c)
      if (candidate[c]) { lead = c; break; }
    if (lead < 0) return false;  // already in the span
    echelonRows.push_back(std::move(candidate));
    leadingColumn.push_back(lead);
    return true;
  }
};

// Solve matrix·x = rhs over GF(2) for a square, invertible matrix (the
// duality pairing). Throws if the system is not uniquely solvable.
Gf2Vector gf2Solve(const Gf2Matrix &matrix, const Gf2Vector &rhs) {
  const int n = static_cast<int>(rhs.size());
  Gf2Matrix augmented(matrix.size(), Gf2Vector(n + 1, 0));
  for (int r = 0; r < static_cast<int>(matrix.size()); ++r) {
    for (int c = 0; c < n; ++c) augmented[r][c] = matrix[r][c];
    augmented[r][n] = rhs[r];
  }
  const std::vector<int> pivotColumns = gf2ReduceRows(augmented, n + 1);
  if (static_cast<int>(pivotColumns.size()) != n || pivotColumns.back() == n)
    throw std::runtime_error(
        "ChainComplex::stiefelWhitneyNumbers: the Poincaré-duality pairing is "
        "not invertible (the complex is not a closed manifold)");
  Gf2Vector x(n, 0);
  for (int t = 0; t < n; ++t) x[pivotColumns[t]] = augmented[t][n];
  return x;
}

bool isZeroVector(const Gf2Vector &v) {
  for (std::uint8_t e : v)
    if (e) return false;
  return true;
}

}  // namespace

std::map<std::string, int> ChainComplex::stiefelWhitneyNumbers() const {
  std::map<std::string, int> numbers;
  const int n = dimension_;
  if (n < 0) return numbers;  // empty complex: no characteristic numbers

  const auto countAt = [&](int k) {
    return (k < 0 || k > n) ? 0 : static_cast<int>(counts_[static_cast<std::size_t>(k)]);
  };

  // Mod-2 boundary entry ∂_k[row][col] (k-simplex col -> its (k-1)-faces).
  const auto boundaryBit = [&](int k, int row, int col) -> std::uint8_t {
    const auto &flat = boundary_[static_cast<std::size_t>(k)];
    const int cols = countAt(k);
    return static_cast<std::uint8_t>(
        std::abs(flat[static_cast<std::size_t>(row) * cols + col]) & 1);
  };

  // Index of a k-simplex from its sorted vertex ids (for cup-product faces).
  std::vector<std::map<Face, int>> indexOfFace(n + 1);
  for (int k = 0; k <= n; ++k)
    for (int i = 0; i < countAt(k); ++i)
      indexOfFace[k][faceVerts_[k][static_cast<std::size_t>(i)]] = i;

  // ---- mod-2 cohomology bases, one representative cocycle per class ----
  // H^k = ker(δ^k) / im(δ^{k-1}); δ^k = (∂_{k+1})^T, coboundaries = rows of ∂_k.
  std::vector<Gf2Matrix> cohomology(n + 1);  // cohomology[k] = basis cochains in C^k
  for (int k = 0; k <= n; ++k) {
    const int dim = countAt(k);
    // Coboundary operator δ^k as a matrix on length-`dim` cochains, with one
    // row per (k+1)-simplex: (δ^k α)(τ) = α(∂τ).
    Gf2Matrix coboundaryOperator;
    if (k + 1 <= n) {
      const int higher = countAt(k + 1);
      coboundaryOperator.assign(higher, Gf2Vector(dim, 0));
      for (int tau = 0; tau < higher; ++tau)
        for (int face = 0; face < dim; ++face)
          coboundaryOperator[tau][face] = boundaryBit(k + 1, face, tau);
    }
    const Gf2Matrix cocycles = gf2Kernel(std::move(coboundaryOperator), dim);

    // Span seeded with the coboundaries (rows of ∂_k); cocycles independent of
    // them are the cohomology generators.
    Gf2Span span(dim);
    for (int row = 0; row < countAt(k - 1); ++row) {
      Gf2Vector coboundary(dim, 0);
      for (int col = 0; col < dim; ++col) coboundary[col] = boundaryBit(k, row, col);
      span.add(std::move(coboundary));
    }
    for (const Gf2Vector &cocycle : cocycles)
      if (span.add(cocycle)) cohomology[k].push_back(cocycle);
  }

  // ---- Alexander–Whitney cup product on cochains (mod 2) ----
  // (α ∪ β) on a (p+q)-simplex [v0..v_{p+q}] = α(v0..vp) · β(vp..v_{p+q}).
  const auto cup = [&](const Gf2Vector &alpha, int p, const Gf2Vector &beta,
                       int q) -> Gf2Vector {
    const int degree = p + q;
    const int dim = countAt(degree);
    Gf2Vector product(dim, 0);
    if (degree > n) return product;
    for (int s = 0; s < dim; ++s) {
      const Face &vertices = faceVerts_[degree][static_cast<std::size_t>(s)];
      const Face front(vertices.begin(), vertices.begin() + (p + 1));
      const Face back(vertices.begin() + p, vertices.end());
      product[s] = static_cast<std::uint8_t>(
          alpha[indexOfFace[p].at(front)] & beta[indexOfFace[q].at(back)]);
    }
    return product;
  };

  // Evaluate a top-degree cochain on the fundamental class [K]: the mod-2 sum
  // over all top simplices.
  const auto evaluateOnFundamentalClass = [&](const Gf2Vector &topCochain) -> int {
    int total = 0;
    for (std::uint8_t e : topCochain) total ^= e;
    return total;
  };

  // ---- Wu classes v_k (1 ≤ k ≤ n/2; the rest vanish for degree reasons) ----
  // v_k is the element of H^k with <v_k ∪ x, [K]> = <Sq^k x, [K]> for all
  // x in H^{n-k}. Sq^k on a degree-(n-k) class is the cup square when k = n-k,
  // zero when k > n-k, and (deferred) a higher cup-i product when k < n-k.
  std::vector<Gf2Vector> wuClass(n + 1);
  for (int k = 0; k <= n; ++k) wuClass[k].assign(countAt(k), 0);
  for (int k = 1; 2 * k <= n; ++k) {
    const int complement = n - k;
    const auto &basisK = cohomology[k];
    const auto &basisComplement = cohomology[complement];
    if (basisK.empty()) continue;  // H^k = 0 ⇒ v_k = 0
    if (basisK.size() != basisComplement.size())
      throw std::runtime_error(
          "ChainComplex::stiefelWhitneyNumbers: mod-2 Poincaré duality fails "
          "(dim H^" + std::to_string(k) + " != dim H^" + std::to_string(complement) +
          "); the complex is not a closed manifold");

    // Pairing matrix P[j][i] = <e_i ∪ x_j, [K]> and right-hand side
    // r[j] = <Sq^k x_j, [K]>.
    Gf2Matrix pairing(basisComplement.size(), Gf2Vector(basisK.size(), 0));
    Gf2Vector rightHandSide(basisComplement.size(), 0);
    for (int j = 0; j < static_cast<int>(basisComplement.size()); ++j) {
      for (int i = 0; i < static_cast<int>(basisK.size()); ++i)
        pairing[j][i] = static_cast<std::uint8_t>(
            evaluateOnFundamentalClass(cup(basisK[i], k, basisComplement[j], complement)));
      if (k == complement)  // Sq^k on a degree-k class is the cup square
        rightHandSide[j] = static_cast<std::uint8_t>(evaluateOnFundamentalClass(
            cup(basisComplement[j], complement, basisComplement[j], complement)));
      // k < complement would need a higher cup-i product; but H^k ≠ 0 with
      // k < n-k cannot occur for the supported manifolds (it requires a
      // nonzero low-degree cohomology paired against a higher one). Guard it.
      else
        throw std::runtime_error(
            "ChainComplex::stiefelWhitneyNumbers: Wu class v_" + std::to_string(k) +
            " requires a higher Steenrod cup-i product (i>0), which is deferred "
            "(see issue #65)");
    }
    const Gf2Vector coefficients = gf2Solve(pairing, rightHandSide);
    Gf2Vector v(countAt(k), 0);
    for (int i = 0; i < static_cast<int>(coefficients.size()); ++i)
      if (coefficients[i])
        for (int c = 0; c < countAt(k); ++c) v[c] ^= basisK[i][c];
    wuClass[k] = std::move(v);
  }

  // ---- Stiefel–Whitney classes w_i = Σ_j Sq^j(v_{i-j}) ----
  // Sq^0(v_i) = v_i; Sq^j(v_{i-j}) for j ≥ 1 is the cup square when 2j = i
  // (degree i-j = j) and zero when the Wu class vanishes; anything else is a
  // deferred higher square. Computed lazily so a deferred term is only hit if
  // it is actually needed (and nonzero).
  std::vector<bool> haveW(n + 1, false);
  std::vector<Gf2Vector> wClass(n + 1);
  const auto stiefelWhitney = [&](int i) -> const Gf2Vector & {
    if (haveW[i]) return wClass[i];
    Gf2Vector w(countAt(i), 0);
    if (i <= n - i)  // Sq^0(v_i): v_i itself (zero unless 1 ≤ i ≤ n/2)
      for (int c = 0; c < countAt(i); ++c) w[c] ^= wuClass[i][c];
    for (int j = 1; j <= i; ++j) {
      const int m = i - j;  // degree of the Wu class being squared
      if (isZeroVector(wuClass[m])) continue;  // Sq^j(0) = 0
      if (j > m) continue;                      // Sq^j on degree m < j is 0
      if (j == m) {                             // Sq^j on degree j is the square
        const Gf2Vector square = cup(wuClass[m], m, wuClass[m], m);
        for (int c = 0; c < countAt(i); ++c) w[c] ^= square[c];
        continue;
      }
      throw std::runtime_error(
          "ChainComplex::stiefelWhitneyNumbers: Stiefel–Whitney class w_" +
          std::to_string(i) + " requires a higher Steenrod cup-i product (i>0), "
          "which is deferred (see issue #65)");
    }
    haveW[i] = true;
    wClass[i] = std::move(w);
    return wClass[i];
  };

  // ---- Stiefel–Whitney numbers: every partition of n into positive parts ----
  // The monomial w_{i_1}···w_{i_r} (parts ascending) cupped together and
  // evaluated on [K]. Parts are processed smallest-first so a zero factor (e.g.
  // w_1 = 0 on an orientable manifold) short-circuits before a deferred,
  // would-be-higher-square factor is ever requested.
  std::function<void(int, int, std::vector<int> &)> forEachPartition =
      [&](int remaining, int minimumPart, std::vector<int> &parts) {
        if (remaining == 0) {
          // Build a readable monomial key, e.g. {1,1,2} -> "w1^2w2".
          std::string key;
          for (int p = 0; p < static_cast<int>(parts.size());) {
            int q = p;
            while (q < static_cast<int>(parts.size()) && parts[q] == parts[p]) ++q;
            const int multiplicity = q - p;
            key += "w" + std::to_string(parts[p]);
            if (multiplicity > 1) key += "^" + std::to_string(multiplicity);
            p = q;
          }
          // Evaluate the cup-product monomial, short-circuiting on a zero factor.
          Gf2Vector accumulator;
          int accumulatorDegree = 0;
          bool zero = false;
          for (int idx = 0; idx < static_cast<int>(parts.size()); ++idx) {
            const Gf2Vector &factor = stiefelWhitney(parts[idx]);
            if (idx == 0) {
              accumulator = factor;
              accumulatorDegree = parts[idx];
            } else {
              accumulator = cup(accumulator, accumulatorDegree, factor, parts[idx]);
              accumulatorDegree += parts[idx];
            }
            if (isZeroVector(accumulator)) { zero = true; break; }
          }
          numbers[key] = zero ? 0 : evaluateOnFundamentalClass(accumulator);
          return;
        }
        for (int part = minimumPart; part <= remaining; ++part) {
          parts.push_back(part);
          forEachPartition(remaining - part, part, parts);
          parts.pop_back();
        }
      };
  std::vector<int> parts;
  forEachPartition(n, 1, parts);
  return numbers;
}

std::pair<bool, std::string> ChainComplex::dualComplexIsValid(
    const std::vector<std::vector<std::uint64_t>> &topCells, int dim,
    const std::vector<std::vector<std::uint64_t>> &facetCells) {
  using Cell = std::vector<std::uint64_t>;
  const auto joinIds = [](const Cell &c) {
    std::string out = "(";
    for (std::size_t i = 0; i < c.size(); ++i) {
      if (i) out += ",";
      out += std::to_string(c[i]);
    }
    return out + ")";
  };
  const auto connectedFrom = [](int start, int count,
                                const std::map<int, std::vector<int>> &adj) {
    std::vector<int> stack{start};
    std::map<int, bool> seen{{start, true}};
    while (!stack.empty()) {
      const int x = stack.back();
      stack.pop_back();
      const auto it = adj.find(x);
      if (it == adj.end()) continue;
      for (const int y : it->second)
        if (!seen.count(y)) {
          seen[y] = true;
          stack.push_back(y);
        }
    }
    return static_cast<int>(seen.size()) == count;
  };

  if (topCells.empty()) return {false, "no top cells"};
  const std::size_t nv = static_cast<std::size_t>(dim) + 1;
  std::vector<Cell> cells;
  cells.reserve(topCells.size());
  for (const auto &raw : topCells) {
    if (raw.size() != nv) return {false, "mixed top-cell dimension"};
    Cell c = raw;
    std::sort(c.begin(), c.end());
    cells.push_back(std::move(c));
  }
  {
    auto dedup = cells;
    std::sort(dedup.begin(), dedup.end());
    if (std::adjacent_find(dedup.begin(), dedup.end()) != dedup.end())
      return {false, "duplicate top cell"};
  }

  // Facet coface counts in {1, 2}; the dangling-facet check against the
  // supplied (n-1)-cell universe.
  std::map<Cell, int> cofaces;
  for (const auto &c : cells)
    for (std::size_t j = 0; j < nv; ++j) {
      Cell f;
      f.reserve(nv - 1);
      for (std::size_t i = 0; i < nv; ++i)
        if (i != j) f.push_back(c[i]);
      ++cofaces[f];
    }
  for (const auto &[f, count] : cofaces)
    if (count > 2)
      return {false,
              "facet " + joinIds(f) + " has " + std::to_string(count) + " cofaces"};
  for (const auto &raw : facetCells) {
    Cell f = raw;
    std::sort(f.begin(), f.end());
    if (!cofaces.count(f))
      return {false, "dangling facet " + joinIds(f) + " (0 cofaces)"};
  }

  // Ridge links: the top cells around each (n-2)-simplex, glued along the
  // facets containing it, must form ONE path or cycle (no pinches).
  std::map<Cell, std::vector<int>> atRidge;
  for (std::size_t ci = 0; ci < cells.size(); ++ci) {
    const Cell &c = cells[ci];
    for (std::size_t a = 0; a < nv; ++a)
      for (std::size_t b = a + 1; b < nv; ++b) {
        Cell r;
        r.reserve(nv - 2);
        for (std::size_t i = 0; i < nv; ++i)
          if (i != a && i != b) r.push_back(c[i]);
        atRidge[r].push_back(static_cast<int>(ci));
      }
  }
  for (const auto &[r, cis] : atRidge) {
    if (cis.size() < 2) continue;
    std::map<Cell, std::vector<int>> byFacet;
    for (const int ci : cis)
      for (const std::uint64_t v : cells[static_cast<std::size_t>(ci)])
        if (!std::binary_search(r.begin(), r.end(), v)) {
          Cell f = r;
          f.insert(std::upper_bound(f.begin(), f.end(), v), v);
          byFacet[f].push_back(ci);
        }
    std::map<int, std::vector<int>> adj;
    for (const auto &[f, fc] : byFacet)
      if (fc.size() == 2) {
        adj[fc[0]].push_back(fc[1]);
        adj[fc[1]].push_back(fc[0]);
      }
    if (!connectedFrom(cis.front(), static_cast<int>(cis.size()), adj))
      return {false, "ridge " + joinIds(r) + ": link is disconnected (pinch)"};
  }
  if (dim == 2) return {true, "ok"};

  // n >= 4: a complex is a PL manifold iff every facet is in <= 2 cofaces (checked
  // above) AND every vertex link is itself a valid (n-1)-manifold. Recurse on the
  // links (link top cells = each cell minus the vertex); the recursion bottoms out
  // at the n==3 S^2-vertex-link rigor below, so a 4-manifold's links are certified
  // as genuine closed 3-manifolds (interior) or balls (boundary). The n==3 inline
  // check (chi-based, slightly stronger: it pins links to S^2 rather than any
  // closed surface) is kept; sphere-vs-other-manifold at the top is the caller's
  // separate Betti check.
  if (dim >= 4) {
    std::map<std::uint64_t, std::vector<Cell>> linkTops;
    for (const auto &c : cells)
      for (const std::uint64_t v : c) {
        Cell lf;
        lf.reserve(nv - 1);
        for (const std::uint64_t u : c)
          if (u != v) lf.push_back(u);
        linkTops[v].push_back(std::move(lf));
      }
    for (const auto &[v, lt] : linkTops) {
      const auto verdict = dualComplexIsValid(lt, dim - 1, {});
      if (!verdict.first)
        return {false,
                "vertex " + std::to_string(v) + " link: " + verdict.second};
    }
    return {true, "ok"};
  }

  // n == 3: vertex links must be 2-spheres (interior) or disks (boundary).
  std::map<std::uint64_t, std::vector<Cell>> atVertex;
  for (const auto &c : cells)
    for (const std::uint64_t v : c) {
      Cell lf;
      lf.reserve(nv - 1);
      for (const std::uint64_t u : c)
        if (u != v) lf.push_back(u);
      atVertex[v].push_back(lf);
    }
  for (const auto &[v, linkFaces] : atVertex) {
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::vector<int>> edgeFaces;
    for (std::size_t i = 0; i < linkFaces.size(); ++i) {
      const Cell &lf = linkFaces[i];
      for (std::size_t a = 0; a < lf.size(); ++a)
        for (std::size_t b = a + 1; b < lf.size(); ++b)
          edgeFaces[{lf[a], lf[b]}].push_back(static_cast<int>(i));
    }
    std::map<int, std::vector<int>> adj;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> boundaryEdges;
    for (const auto &[le, fs] : edgeFaces) {
      if (fs.size() == 2) {
        adj[fs[0]].push_back(fs[1]);
        adj[fs[1]].push_back(fs[0]);
      } else if (fs.size() == 1) {
        boundaryEdges.push_back(le);
      } else {
        return {false, "vertex " + std::to_string(v) + ": link edge in " +
                           std::to_string(fs.size()) + " faces"};
      }
    }
    if (!connectedFrom(0, static_cast<int>(linkFaces.size()), adj))
      return {false,
              "vertex " + std::to_string(v) + ": link is disconnected (pinch)"};
    std::map<std::uint64_t, int> linkVertexDegree;
    for (const auto &lf : linkFaces)
      for (const std::uint64_t u : lf) linkVertexDegree[u] = 0;
    const int chi = static_cast<int>(linkVertexDegree.size()) -
                    static_cast<int>(edgeFaces.size()) +
                    static_cast<int>(linkFaces.size());
    if (boundaryEdges.empty()) {
      if (chi != 2)
        return {false, "vertex " + std::to_string(v) + ": closed link has chi=" +
                           std::to_string(chi) + ", not S^2"};
    } else {
      if (chi != 1)
        return {false, "vertex " + std::to_string(v) + ": bounded link has chi=" +
                           std::to_string(chi) + ", not a disk"};
      std::map<std::uint64_t, std::vector<std::uint64_t>> badj;
      for (const auto &[a, b] : boundaryEdges) {
        badj[a].push_back(b);
        badj[b].push_back(a);
      }
      for (const auto &[u, nbrs] : badj)
        if (nbrs.size() != 2)
          return {false, "vertex " + std::to_string(v) +
                             ": link boundary is not a 1-manifold"};
      std::vector<std::uint64_t> stack{boundaryEdges.front().first};
      std::map<std::uint64_t, bool> seen{{boundaryEdges.front().first, true}};
      while (!stack.empty()) {
        const std::uint64_t x = stack.back();
        stack.pop_back();
        for (const std::uint64_t y : badj[x])
          if (!seen.count(y)) {
            seen[y] = true;
            stack.push_back(y);
          }
      }
      if (seen.size() != badj.size())
        return {false, "vertex " + std::to_string(v) +
                           ": link boundary has several circles"};
    }
  }
  return {true, "ok"};
}

std::vector<int> ChainComplex::endSignCovector(
    const std::vector<std::vector<std::uint64_t>> &surfaceCells,
    const std::vector<std::vector<std::uint64_t>> &holes) {
  using Cell = std::vector<std::uint64_t>;
  const auto joinIds = [](const Cell &c) {
    std::string out = "(";
    for (std::size_t i = 0; i < c.size(); ++i) {
      if (i) out += ",";
      out += std::to_string(c[i]);
    }
    return out + ")";
  };
  if (holes.empty()) return {};

  // The oriented complex is the union surface ∪ holes (the capped end), as
  // sorted-unique sorted tuples — the lexicographic order makes the component
  // roots, and with them the whole covector, deterministic.
  std::set<Cell> uniq;
  const std::size_t nv = holes.front().size();
  const auto addCell = [&](const Cell &raw) {
    if (raw.size() != nv)
      throw std::runtime_error(
          "ChainComplex::endSignCovector: cell " + joinIds(raw) + " has " +
          std::to_string(raw.size()) + " vertices, expected " +
          std::to_string(nv) + " (one dimension throughout)");
    Cell c = raw;
    std::sort(c.begin(), c.end());
    uniq.insert(std::move(c));
  };
  for (const auto &raw : holes) addCell(raw);
  for (const auto &raw : surfaceCells) addCell(raw);
  const std::vector<Cell> cells(uniq.begin(), uniq.end());

  // facet -> its cofaces as (cell index, boundary sign of the facet in that
  // cell): facet j of a sorted cell drops vertex j and carries (-1)^j.
  std::map<Cell, std::vector<std::pair<std::size_t, int>>> cofaces;
  for (std::size_t ci = 0; ci < cells.size(); ++ci)
    for (std::size_t j = 0; j < nv; ++j) {
      Cell f;
      f.reserve(nv - 1);
      for (std::size_t i = 0; i < nv; ++i)
        if (i != j) f.push_back(cells[ci][i]);
      cofaces[f].emplace_back(ci, (j % 2 == 0) ? 1 : -1);
    }
  for (const auto &[f, at] : cofaces)
    if (at.size() > 2)
      throw std::runtime_error(
          "ChainComplex::endSignCovector: facet " + joinIds(f) + " has " +
          std::to_string(at.size()) + " cofaces (not a pseudomanifold)");

  // Orient by propagation: across an interior facet the two induced signs must
  // cancel (eps_b = -eps_a * s_a * s_b); boundary facets (one coface) impose
  // nothing. Component roots are the lex-smallest unvisited cells, eps = +1.
  std::vector<int> eps(cells.size(), 0);
  for (std::size_t root = 0; root < cells.size(); ++root) {
    if (eps[root] != 0) continue;
    eps[root] = 1;
    std::vector<std::size_t> stack{root};
    while (!stack.empty()) {
      const std::size_t a = stack.back();
      stack.pop_back();
      for (std::size_t j = 0; j < nv; ++j) {
        Cell f;
        f.reserve(nv - 1);
        for (std::size_t i = 0; i < nv; ++i)
          if (i != j) f.push_back(cells[a][i]);
        const int sa = (j % 2 == 0) ? 1 : -1;
        for (const auto &[b, sb] : cofaces.at(f)) {
          if (b == a) continue;
          const int want = -eps[a] * sa * sb;
          if (eps[b] == 0) {
            eps[b] = want;
            stack.push_back(b);
          } else if (eps[b] != want) {
            throw std::runtime_error(
                "ChainComplex::endSignCovector: orientation propagation "
                "contradicts itself at facet " + joinIds(f) +
                " (the end surface is non-orientable)");
          }
        }
      }
    }
  }

  std::vector<int> sigma;
  sigma.reserve(holes.size());
  for (const auto &raw : holes) {
    Cell h = raw;
    std::sort(h.begin(), h.end());
    const auto it = std::lower_bound(cells.begin(), cells.end(), h);
    sigma.push_back(eps[static_cast<std::size_t>(it - cells.begin())]);
  }
  return sigma;
}

bool OrientationLocalSystem::orientable() const noexcept {
  return std::all_of(transitions.begin(), transitions.end(),
                     [](const OrientationTransition &transition) {
                       return transition.holonomy == 1;
                     });
}

std::vector<int> OrientationLocalSystem::holonomies() const {
  std::vector<int> result;
  result.reserve(transitions.size());
  for (const auto &transition : transitions)
    result.push_back(transition.holonomy);
  return result;
}

std::vector<double> OrientationLocalSystem::connectionLaplacian() const {
  const std::size_t count = cells.size();
  std::vector<double> result(count * count, 0.0);
  for (const auto &transition : transitions) {
    const std::size_t a = transition.first;
    const std::size_t b = transition.second;
    if (a >= count || b >= count)
      throw std::runtime_error(
          "OrientationLocalSystem::connectionLaplacian: transition index "
          "outside the cell ordering");
    result[a * count + a] += 1.0;
    result[b * count + b] += 1.0;
    result[a * count + b] -= static_cast<double>(transition.transport);
    result[b * count + a] -= static_cast<double>(transition.transport);
  }
  return result;
}

OrientationLocalSystem ChainComplex::orientationLocalSystem(
    const std::vector<std::vector<std::uint64_t>> &topCells) {
  using Cell = std::vector<std::uint64_t>;
  const auto joinIds = [](const Cell &c) {
    std::string out = "(";
    for (std::size_t i = 0; i < c.size(); ++i) {
      if (i) out += ",";
      out += std::to_string(c[i]);
    }
    return out + ")";
  };
  OrientationLocalSystem result;
  if (topCells.empty()) return result;

  // Sorted-unique cells: the canonical C_d column order, so the returned
  // covector aligns with orientedTopSimplices() / kSimplexVertices(dim) and is
  // independent of the order topCells is supplied in.
  std::set<Cell> uniq;
  const std::size_t nv = topCells.front().size();
  for (const auto &raw : topCells) {
    if (raw.size() != nv)
      throw std::runtime_error(
          "ChainComplex::orientationLocalSystem: cell " + joinIds(raw) + " has " +
          std::to_string(raw.size()) + " vertices, expected " +
          std::to_string(nv) + " (one dimension throughout)");
    Cell c = raw;
    std::sort(c.begin(), c.end());
    uniq.insert(std::move(c));
  }
  const std::vector<Cell> cells(uniq.begin(), uniq.end());
  result.cells = cells;

  // facet -> its cofaces as (cell index, boundary sign): facet j of a sorted
  // cell drops vertex j and carries (-1)^j.
  std::map<Cell, std::vector<std::pair<std::size_t, int>>> cofaces;
  for (std::size_t ci = 0; ci < cells.size(); ++ci)
    for (std::size_t j = 0; j < nv; ++j) {
      Cell f;
      f.reserve(nv - 1);
      for (std::size_t i = 0; i < nv; ++i)
        if (i != j) f.push_back(cells[ci][i]);
      cofaces[f].emplace_back(ci, (j % 2 == 0) ? 1 : -1);
    }
  for (const auto &[f, at] : cofaces)
    if (at.size() > 2)
      throw std::runtime_error(
          "ChainComplex::orientationLocalSystem: facet " + joinIds(f) + " has " +
          std::to_string(at.size()) + " cofaces (not a pseudomanifold)");

  // The orientation connection on the dual graph. Across an interior facet the
  // two induced signs must cancel, so eps_b = transport * eps_a with
  // transport = -s_a*s_b. Boundary facets carry no dual edge.
  for (const auto &[facet, at] : cofaces) {
    if (at.size() != 2) continue;
    auto [a, sa] = at[0];
    auto [b, sb] = at[1];
    if (b < a) {
      std::swap(a, b);
      std::swap(sa, sb);
    }
    result.transitions.push_back(
        OrientationTransition{a, b, facet, -sa * sb, 1});
  }

  // Choose a deterministic spanning-forest gauge. Every discovery edge is
  // trivialized to +1. A non-tree edge is then +1 when the assignment closes
  // consistently and -1 when it represents orientation-reversing holonomy.
  std::vector<std::vector<std::size_t>> adjacency(cells.size());
  for (std::size_t index = 0; index < result.transitions.size(); ++index) {
    const auto &transition = result.transitions[index];
    adjacency[transition.first].push_back(index);
    adjacency[transition.second].push_back(index);
  }

  std::vector<int> eps(cells.size(), 0);
  for (std::size_t root = 0; root < cells.size(); ++root) {
    if (eps[root] != 0) continue;
    ++result.components;
    eps[root] = 1;
    std::vector<std::size_t> stack{root};
    while (!stack.empty()) {
      const std::size_t a = stack.back();
      stack.pop_back();
      for (const std::size_t transitionIndex : adjacency[a]) {
        const auto &transition = result.transitions[transitionIndex];
        const std::size_t b = transition.first == a ? transition.second
                                                     : transition.first;
        const int want = transition.transport * eps[a];
        if (eps[b] == 0) {
          eps[b] = want;
          stack.push_back(b);
        }
      }
    }
  }
  result.trivialization = std::move(eps);
  for (auto &transition : result.transitions)
    transition.holonomy =
        result.trivialization[transition.first] * transition.transport *
        result.trivialization[transition.second];
  return result;
}

std::vector<int> ChainComplex::orientationCovector(
    const std::vector<std::vector<std::uint64_t>> &topCells) {
  const OrientationLocalSystem localSystem = orientationLocalSystem(topCells);
  const auto joinIds = [](const std::vector<std::uint64_t> &cell) {
    std::string out = "(";
    for (std::size_t i = 0; i < cell.size(); ++i) {
      if (i) out += ",";
      out += std::to_string(cell[i]);
    }
    return out + ")";
  };
  for (const auto &transition : localSystem.transitions)
    if (transition.holonomy == -1)
      throw std::runtime_error(
          "ChainComplex::orientationCovector: orientation propagation "
          "contradicts itself at facet " + joinIds(transition.facet) +
          " (the complex is non-orientable)");
  return localSystem.trivialization;
}

}  // namespace tessera::cobordism
