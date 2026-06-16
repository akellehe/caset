// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/Register.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include <Eigen/SVD>

#include "cobordism/ChainComplex.h"
#include "cobordism/HodgeLaplacian.h"

namespace tessera::cobordism {

using cd = std::complex<double>;

Register::Register(std::shared_ptr<Spacetime> st,
                   const std::vector<std::vector<std::uint64_t>> &classHoles,
                   const std::vector<std::vector<std::uint64_t>> &extraHoles,
                   int growVertices, std::uint64_t growSeed)
    : st_(st), synth_(std::move(st), 1) {
  // --- normalize the class holes to sorted triples and open them by surgery ---
  for (const auto &hole : classHoles) {
    std::vector<std::uint64_t> h(hole);
    std::sort(h.begin(), h.end());
    classHoles_.push_back(h);
    if (!synth_.removeInteriorCell(h))
      throw std::runtime_error(
          "Register: class hole is not a removable interior top cell");
  }

  // --- additive growth: up to growVertices boundary-fixed stellar subdivisions ---
  if (growVertices > 0) {
    std::mt19937_64 rng(growSeed);
    for (int i = 0; i < growVertices; ++i) {
      std::vector<std::vector<std::uint64_t>> sites = synth_.interiorTopCells();
      if (sites.empty()) break;
      std::sort(sites.begin(), sites.end());
      std::uniform_int_distribution<std::size_t> pick(0, sites.size() - 1);
      if (synth_.stellarSubdivideInterior(sites[pick(rng)]).first) ++grown_;
    }
  }

  // --- open any extra holes that are still removable interior top cells ---
  for (const auto &cell : extraHoles) {
    std::vector<std::uint64_t> c(cell);
    std::sort(c.begin(), c.end());
    std::vector<std::vector<std::uint64_t>> avail = synth_.interiorTopCells();
    if (std::find(avail.begin(), avail.end(), c) == avail.end()) continue;
    if (synth_.removeInteriorCell(c)) extraOpened_.push_back(c);
  }

  cells_ = synth_.cellSimplices();
  const std::size_t nCells = cells_.size();
  const std::size_t m = classHoles_.size();

  // --- the harmonic amplitude matrix H (dim x |C_1|), reused from HodgeLaplacian ---
  std::vector<cd> hflat = HodgeLaplacian(st_).harmonicMatrix(1);
  dim_ = (nCells > 0) ? static_cast<int>(hflat.size() / nCells) : 0;
  hFull_.resize(dim_, static_cast<Eigen::Index>(nCells));
  for (int r = 0; r < dim_; ++r)
    for (std::size_t c = 0; c < nCells; ++c)
      hFull_(r, static_cast<Eigen::Index>(c)) = hflat[r * nCells + c];

  // --- the period matrix P (dim x #holes), reused from the #286 read-out ---
  std::vector<cd> pflat = synth_.cyclePeriods(classHoles_);
  periods_.resize(dim_, static_cast<Eigen::Index>(m));
  for (int r = 0; r < dim_; ++r)
    for (std::size_t q = 0; q < m; ++q)
      periods_(r, static_cast<Eigen::Index>(q)) = pflat[r * m + q];

  // rank(P): the number of singular values above the 1e-9 floor (numpy.matrix_rank).
  if (dim_ > 0 && m > 0) {
    Eigen::BDCSVD<Eigen::MatrixXcd> svd(periods_);
    const auto &sv = svd.singularValues();
    for (Eigen::Index i = 0; i < sv.size(); ++i)
      if (sv(i) > 1e-9) ++rank_;
  }

  // --- the induced-orientation constraint n = sign = the end-sign covector ---
  std::vector<int> covec = ChainComplex::endSignCovector(synth_.topCells(), classHoles_);
  n_.resize(static_cast<Eigen::Index>(covec.size()));
  for (std::size_t i = 0; i < covec.size(); ++i)
    n_(static_cast<Eigen::Index>(i)) = static_cast<double>(covec[i]);
  sign_ = n_;
}

}  // namespace tessera::cobordism
