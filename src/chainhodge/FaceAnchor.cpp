// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "chainhodge/FaceAnchor.h"

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/SVD>

namespace tessera::chainhodge {

int FaceAnchor::numericalRank(const Eigen::MatrixXcd &A, double kappa) {
  if (A.rows() == 0 || A.cols() == 0) return 0;
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(A);
  const Eigen::VectorXd sv = svd.singularValues();
  const double tol = kappa * static_cast<double>(std::max(A.rows(), A.cols())) *
                     std::numeric_limits<double>::epsilon() * sv(0);
  int r = 0;
  for (int i = 0; i < sv.size(); ++i)
    if (sv(i) > tol) ++r;
  return r;
}

std::vector<int> FaceAnchor::triangleEdgeIndices(const cobordism::ChainComplex &K,
                                                 std::size_t faceIndex) {
  if (K.dimension() < 2)
    throw std::invalid_argument("FaceAnchor: the complex has no triangles");
  const auto tris = K.kSimplexVertices(2);
  if (faceIndex >= tris.size())
    throw std::invalid_argument("FaceAnchor: triangle index " + std::to_string(faceIndex) +
                                " out of range (" + std::to_string(tris.size()) + " triangles)");
  const auto edges = K.kSimplexVertices(1);
  std::map<std::pair<std::uint64_t, std::uint64_t>, int> index;
  for (int j = 0; j < static_cast<int>(edges.size()); ++j)
    index[{edges[static_cast<std::size_t>(j)][0], edges[static_cast<std::size_t>(j)][1]}] = j;
  const auto &t = tris[faceIndex];
  // Local order (v0 v1), (v0 v2), (v1 v2): the lexicographic order of the
  // triangle's 2-subsets, the same order WhitneyMass uses for a top triangle.
  std::vector<int> out;
  for (const auto &[a, b] : {std::make_pair(t[0], t[1]), std::make_pair(t[0], t[2]),
                             std::make_pair(t[1], t[2])}) {
    const auto it = index.find({a, b});
    if (it == index.end())
      throw std::invalid_argument("FaceAnchor: a triangle edge is missing from C_1");
    out.push_back(it->second);
  }
  return out;
}

FaceBlock FaceAnchor::whitneyFaceBlock(const cobordism::ChainComplex &K, const SquaredLengths &s,
                                       std::size_t faceIndex, Branch branch) {
  const std::vector<int> edges = triangleEdgeIndices(K, faceIndex);
  FaceBlock out;
  out.faceIndex = faceIndex;
  out.edgeIndices = edges;
  out.preset = Preset::L2;
  out.block = Eigen::MatrixXcd::Zero(3, 3);
  // Sum over the top simplices that contain the triangle (all of its edges):
  // at d = 2 exactly the triangle's own local block.
  for (const auto &tb : WhitneyMass::topSimplexBlocks(K, s, 1, branch, false)) {
    std::vector<int> local(3, -1);
    bool contains = true;
    for (int p = 0; p < 3 && contains; ++p) {
      const auto it = std::find(tb.cellIndices.begin(), tb.cellIndices.end(), edges[static_cast<std::size_t>(p)]);
      if (it == tb.cellIndices.end())
        contains = false;
      else
        local[static_cast<std::size_t>(p)] = static_cast<int>(it - tb.cellIndices.begin());
    }
    if (!contains) continue;
    for (int p = 0; p < 3; ++p)
      for (int q = 0; q < 3; ++q)
        out.block(p, q) += tb.block(local[static_cast<std::size_t>(p)], local[static_cast<std::size_t>(q)]);
  }
  out.rank = numericalRank(out.block);
  return out;
}

std::vector<FaceBlock> FaceAnchor::whitneyFaceBlocks(const cobordism::ChainComplex &K,
                                                     const SquaredLengths &s, Branch branch) {
  std::vector<FaceBlock> out;
  const std::size_t n = K.dimension() >= 2 ? K.numSimplices(2) : 0;
  out.reserve(n);
  for (std::size_t t = 0; t < n; ++t) out.push_back(whitneyFaceBlock(K, s, t, branch));
  return out;
}

FaceBlock FaceAnchor::grassmannFaceBlock(const cobordism::ChainComplex &K, const SquaredLengths &s,
                                         std::size_t faceIndex) {
  const std::vector<int> edges = triangleEdgeIndices(K, faceIndex);
  if (s.size() != K.numSimplices(1))
    throw std::invalid_argument("FaceAnchor: one squared length per edge is required");
  const auto cells = K.kSimplexVertices(1);
  std::map<std::pair<std::uint64_t, std::uint64_t>, Complex> S;
  for (std::size_t j = 0; j < cells.size(); ++j) S[{cells[j][0], cells[j][1]}] = s[j];
  auto Sof = [&](std::uint64_t a, std::uint64_t b) -> Complex {
    if (a == b) return Complex(0.0, 0.0);
    if (a > b) std::swap(a, b);
    return S.at({a, b});
  };
  // <u_e, u_f> for u_(a,b) = x_b - x_a by polarization: 1/2 (S(b,c) + S(a,d) - S(b,d) - S(a,c)).
  auto dot = [&](const std::vector<std::uint64_t> &e, const std::vector<std::uint64_t> &f) -> Complex {
    return 0.5 * (Sof(e[1], f[0]) + Sof(e[0], f[1]) - Sof(e[1], f[1]) - Sof(e[0], f[0]));
  };
  FaceBlock out;
  out.faceIndex = faceIndex;
  out.edgeIndices = edges;
  out.preset = Preset::GRASSMANN_ALL;
  out.block = Eigen::MatrixXcd::Zero(3, 3);
  for (int p = 0; p < 3; ++p)
    for (int q = 0; q < 3; ++q)
      out.block(p, q) = dot(cells[static_cast<std::size_t>(edges[static_cast<std::size_t>(p)])],
                            cells[static_cast<std::size_t>(edges[static_cast<std::size_t>(q)])]);
  out.rank = numericalRank(out.block);
  return out;
}

FaceBlock FaceAnchor::faceBlock(const ChainHodge &hodge, std::size_t faceIndex) {
  if (hodge.preset() == Preset::GRASSMANN_ALL)
    return grassmannFaceBlock(hodge.complex(), hodge.squaredLengths(), faceIndex);
  return whitneyFaceBlock(hodge.complex(), hodge.squaredLengths(), faceIndex, hodge.branch());
}

Eigen::MatrixXcd FaceAnchor::dressedFaceBlock(const FaceBlock &block, const cobordism::ChainComplex &K,
                                              const Connection &U) {
  const auto edges = K.kSimplexVertices(1);
  Eigen::MatrixXcd out = block.block;
  for (int p = 0; p < 3; ++p)
    for (int q = 0; q < 3; ++q) {
      const auto &e = edges[static_cast<std::size_t>(block.edgeIndices[static_cast<std::size_t>(p)])];
      const auto &f = edges[static_cast<std::size_t>(block.edgeIndices[static_cast<std::size_t>(q)])];
      out(p, q) *= U.link(e[0], f[0]);  // U_{b(e) b(e')}, b = the smaller (first) vertex
    }
  return out;
}

Eigen::MatrixXcd FaceAnchor::applyFaceEndomorphism(const CovariantChainHodge &cov, std::size_t faceIndex,
                                                   const Eigen::MatrixXcd &c) {
  const FaceBlock fb = faceBlock(cov.base(), faceIndex);
  const Eigen::MatrixXcd dressed = dressedFaceBlock(fb, cov.base().complex(), cov.connection());
  const Eigen::MatrixXcd y = cov.applyG(1, c);  // G_1^U c
  Eigen::MatrixXcd local(3, y.cols());
  for (int p = 0; p < 3; ++p) local.row(p) = y.row(fb.edgeIndices[static_cast<std::size_t>(p)]);
  const Eigen::MatrixXcd w = dressed * local;
  Eigen::MatrixXcd lifted = Eigen::MatrixXcd::Zero(y.rows(), y.cols());
  for (int p = 0; p < 3; ++p) lifted.row(fb.edgeIndices[static_cast<std::size_t>(p)]) = w.row(p);
  return cov.applyG(1, lifted);  // G_1^U M^{(tau)U} G_1^U c
}

Complex FaceAnchor::anchorCoordinate(const CovariantChainHodge &cov, std::size_t faceIndex,
                                     const Eigen::MatrixXcd &Zdual, const Eigen::MatrixXcd &Z) {
  if (Zdual.rows() != Z.rows() || Zdual.cols() != Z.cols())
    throw std::invalid_argument("FaceAnchor::anchorCoordinate: Zdual and Z must have the same shape");
  const FaceBlock fb = faceBlock(cov.base(), faceIndex);
  const Eigen::MatrixXcd dressed = dressedFaceBlock(fb, cov.base().complex(), cov.connection());
  Eigen::MatrixXcd localZ(3, Z.cols()), localZd(3, Z.cols());
  for (int p = 0; p < 3; ++p) {
    localZ.row(p) = Z.row(fb.edgeIndices[static_cast<std::size_t>(p)]);
    localZd.row(p) = Zdual.row(fb.edgeIndices[static_cast<std::size_t>(p)]);
  }
  const Eigen::MatrixXcd paired = localZd.transpose() * dressed * localZ;  // r x r, transpose pairing
  return paired.determinant();
}

Complex FaceAnchor::anchorCoordinateFromChains(const CovariantChainHodge &cov, std::size_t faceIndex,
                                               const Eigen::MatrixXcd &PhiDual,
                                               const Eigen::MatrixXcd &Phi) {
  const Eigen::MatrixXcd Z = cov.applyG(1, Phi);            // G_1^U Phi
  const Eigen::MatrixXcd Zd = cov.dual().applyG(1, PhiDual);  // G_1^{U^{-1}} Phi^vee
  return anchorCoordinate(cov, faceIndex, Zd, Z);
}

std::vector<Complex> FaceAnchor::anchorCoordinates(const CovariantChainHodge &cov,
                                                   const Eigen::MatrixXcd &Zdual,
                                                   const Eigen::MatrixXcd &Z) {
  const std::size_t n = cov.base().complex().dimension() >= 2 ? cov.base().complex().numSimplices(2) : 0;
  std::vector<Complex> out;
  out.reserve(n);
  for (std::size_t t = 0; t < n; ++t) out.push_back(anchorCoordinate(cov, t, Zdual, Z));
  return out;
}

}  // namespace tessera::chainhodge
