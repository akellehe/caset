// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/EmergentOptimizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include <Eigen/Dense>

#include "cobordism/ChainComplex.h"
#include "cobordism/EigenstateSynthesis.h"
#include "cobordism/SurgicalCone.h"
#include "matter/MatterConfiguration.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "simulations/ReggeSolver.h"
#include "spacetime/Spacetime.h"
#include "spacetime/pachner/AddMove.h"
#include "spacetime/pachner/FlipMove.h"
#include "spacetime/pachner/IFlipMove.h"
#include "spacetime/pachner/RemoveMove.h"

namespace tessera::cobordism {

using ::tessera::MatterConfiguration;
using ::tessera::simulations::ReggeSolver;
using cd = std::complex<double>;

namespace {
constexpr int kDim = 4;  // framework dimension (closed S^4 host)

// Sorted vertex-id tuple of a top simplex (the reference's `_top_tuple`).
std::vector<std::uint64_t> topTuple(const ::tessera::mesh::Simplex &s) {
  std::vector<std::uint64_t> ids;
  for (const auto *v : s.getVertices()) ids.push_back(v->getId());
  std::sort(ids.begin(), ids.end());
  return ids;
}

std::pair<std::uint64_t, std::uint64_t> edgeKey(const ::tessera::mesh::Edge *e) {
  const auto a = e->getSource()->getId();
  const auto b = e->getTarget()->getId();
  return {std::min(a, b), std::max(a, b)};
}
}  // namespace

EmergentOptimizer::EmergentOptimizer(
    std::shared_ptr<Spacetime> host,
    std::vector<std::vector<cd>> inputTargets, std::vector<cd> outputTarget,
    std::vector<int> degrees, double gamma, std::uint64_t seed)
    : st_(std::move(host)),
      inputTargets_(std::move(inputTargets)),
      outputTarget_(std::move(outputTarget)),
      degrees_(std::move(degrees)),
      gateK_(degrees_.empty() ? 0
                              : *std::max_element(degrees_.begin(),
                                                  degrees_.end())),
      gamma_(gamma),
      rng_(seed) {}

std::vector<int> EmergentOptimizer::betti(const Spacetime &st) {
  return ChainComplex::fromSpacetime(st).bettiNumbers();
}

std::vector<std::vector<std::uint64_t>> EmergentOptimizer::emergentHoles(
    const Spacetime &st, int k) {
  // The (k+2)-vertex tuples all of whose drop-one facets are boundary facets.
  std::set<std::vector<std::uint64_t>> bnd;
  for (auto f : st.getBoundary()) {
    std::sort(f.begin(), f.end());
    bnd.insert(std::move(f));
  }
  std::vector<std::vector<std::uint64_t>> out;
  if (bnd.empty() ||
      static_cast<int>(bnd.begin()->size()) != k + 1)  // facets must be k-cells
    return out;
  std::set<std::uint64_t> verts;
  for (const auto &f : bnd)
    for (auto v : f) verts.insert(v);
  std::set<std::vector<std::uint64_t>> holes;
  for (const auto &f : bnd) {
    for (auto v : verts) {
      if (std::find(f.begin(), f.end(), v) != f.end()) continue;
      std::vector<std::uint64_t> tup = f;
      tup.push_back(v);
      std::sort(tup.begin(), tup.end());
      bool allBnd = true;
      for (std::size_t i = 0; i < tup.size(); ++i) {
        std::vector<std::uint64_t> facet;
        for (std::size_t j = 0; j < tup.size(); ++j)
          if (j != i) facet.push_back(tup[j]);
        if (!bnd.count(facet)) {
          allBnd = false;
          break;
        }
      }
      if (allBnd) holes.insert(tup);
    }
  }
  out.assign(holes.begin(), holes.end());
  return out;
}

double EmergentOptimizer::gradNorm2(const std::shared_ptr<Spacetime> &st) {
  ReggeSolver solver(st, MatterConfiguration());
  double n2 = 0.0;
  for (const auto &z : solver.actionGradientExact()) n2 += std::norm(z);
  return n2;
}

double EmergentOptimizer::rState(const std::shared_ptr<Spacetime> &st, int k,
                                 const std::vector<cd> &target) {
  const std::size_t d = target.size();
  Eigen::VectorXcd t(d);
  for (std::size_t i = 0; i < d; ++i) t(i) = target[i];
  const double full = t.squaredNorm();  // zero-filled full leak

  const auto bettis = betti(*st);
  if (k < 0 || k >= static_cast<int>(bettis.size())) return full;
  const int bk = bettis[k];
  if (bk == 0) return full;
  auto holes = emergentHoles(*st, k);
  if (holes.empty()) return full;
  if (holes.size() > d) holes.resize(d);
  const std::size_t m = holes.size();

  EigenstateSynthesis es(st, k);
  const auto flat = es.cyclePeriods(holes);  // bk x m, row-major
  // pd: (bk, d), zero-filled beyond m columns; pdT = pd.transpose() is (d, bk).
  Eigen::MatrixXcd pdT = Eigen::MatrixXcd::Zero(static_cast<int>(d), bk);
  for (int r = 0; r < bk; ++r)
    for (std::size_t q = 0; q < m; ++q)
      pdT(static_cast<int>(q), r) = flat[static_cast<std::size_t>(r) * m + q];

  // min over permutations of the target components of ||pdT c - ts||^2 (lstsq c).
  std::vector<int> perm(d);
  std::iota(perm.begin(), perm.end(), 0);
  double best = std::numeric_limits<double>::infinity();
  Eigen::BDCSVD<Eigen::MatrixXcd> svd(pdT, Eigen::ComputeThinU | Eigen::ComputeThinV);
  do {
    Eigen::VectorXcd ts(d);
    for (std::size_t i = 0; i < d; ++i) ts(i) = t(perm[i]);
    const Eigen::VectorXcd c = svd.solve(ts);
    best = std::min(best, (pdT * c - ts).squaredNorm());
  } while (std::next_permutation(perm.begin(), perm.end()));
  return best;
}

std::shared_ptr<Spacetime> EmergentOptimizer::subOf(
    const std::shared_ptr<Spacetime> &st,
    const std::set<std::uint64_t> &verts) const {
  std::vector<std::vector<std::uint64_t>> cells;
  for (const auto &s : st->getTopSimplices()) {
    auto c = topTuple(*s);
    bool inside = true;
    for (auto v : c)
      if (!verts.count(v)) {
        inside = false;
        break;
      }
    if (inside) cells.push_back(std::move(c));
  }
  if (cells.empty()) return nullptr;
  return Spacetime::fromCells(kDim, cells, 1.0, 0.0);
}

double EmergentOptimizer::rInput(const Input &inp,
                                 const std::shared_ptr<Spacetime> &st) const {
  auto sub = subOf(st, inp.verts);
  double r = 0.0;
  if (!sub) {
    Eigen::VectorXcd t(inp.target.size());
    for (std::size_t i = 0; i < inp.target.size(); ++i) t(i) = inp.target[i];
    return static_cast<double>(degrees_.size()) * t.squaredNorm();
  }
  for (int k : degrees_) r += rState(sub, k, inp.target);
  return r;
}

double EmergentOptimizer::rU(const std::shared_ptr<Spacetime> &st) const {
  double total = 0.0;
  for (int k : degrees_) total += rState(st, k, outputTarget_);  // output
  for (const auto &inp : inputs_) total += rInput(inp, st);      // inputs
  return total;
}

double EmergentOptimizer::objective() const {
  return gradNorm2(st_) + gamma_ * rU(st_);
}

std::set<std::uint64_t> EmergentOptimizer::inputVerts() const {
  std::set<std::uint64_t> out;
  for (const auto &inp : inputs_)
    out.insert(inp.verts.begin(), inp.verts.end());
  return out;
}

EmergentOptimizer::Snapshot EmergentOptimizer::snapshotOf(
    const Spacetime &st) const {
  std::vector<std::vector<std::uint64_t>> cells;
  for (const auto &s : st.getTopSimplices()) cells.push_back(topTuple(*s));
  std::map<std::pair<std::uint64_t, std::uint64_t>, cd> l2;
  for (const auto *e : st.getEdgeList()->toVector())
    l2[edgeKey(e)] = e->getSquaredLength();
  return {std::move(cells), std::move(l2)};
}

EmergentOptimizer::Snapshot EmergentOptimizer::snapshot() const {
  return snapshotOf(*st_);
}

std::shared_ptr<Spacetime> EmergentOptimizer::build(const Snapshot &snap) const {
  auto st = Spacetime::fromCells(kDim, snap.first, 1.0, 0.0);
  for (auto *e : st->getEdgeList()->toVector()) {
    const auto it = snap.second.find(edgeKey(e));
    if (it != snap.second.end()) e->setSquaredLength(it->second);
  }
  return st;
}

EmergentOptimizer::MoveSpec EmergentOptimizer::randomSpec(const Spacetime &st) {
  static const char *kinds[] = {"add",      "remove",  "flip",
                                "iflip",    "cone_out", "cone_in"};
  const std::string kind = kinds[rng_() % 6];
  if (kind == "add" || kind == "remove" || kind == "flip" || kind == "iflip")
    return {kind, {static_cast<std::uint64_t>(rng_() % (1u << 31))}};
  std::vector<std::vector<std::uint64_t>> tops;
  for (const auto &s : st.getTopSimplices()) tops.push_back(topTuple(*s));
  if (tops.empty()) return {"noop", {}};
  const auto &cell = tops[rng_() % tops.size()];
  if (kind == "cone_out") return {"cone_out", cell};
  const std::size_t drop = rng_() % cell.size();
  std::vector<std::uint64_t> face;
  for (std::size_t i = 0; i < cell.size(); ++i)
    if (i != drop) face.push_back(cell[i]);
  return {"cone_in", face};
}

bool EmergentOptimizer::applySpec(const std::shared_ptr<Spacetime> &st,
                                  const MoveSpec &spec) {
  const auto &kind = spec.first;
  if (kind == "noop") return false;
  bool applied = false;
  if (kind == "add" || kind == "remove" || kind == "flip" || kind == "iflip") {
    std::mt19937 r(static_cast<std::uint32_t>(spec.second[0]));
    using ::tessera::spacetime::PachnerMode;
    if (kind == "add") {
      ::tessera::spacetime::AddMove mv(st.get(), &r, false,
                                                PachnerMode::PreGeometric, false);
      applied = mv.propose() && mv.apply();
    } else if (kind == "remove") {
      ::tessera::spacetime::RemoveMove mv(st.get(), &r,
                                                   PachnerMode::PreGeometric, false);
      applied = mv.propose() && mv.apply();
    } else if (kind == "flip") {
      ::tessera::spacetime::FlipMove mv(st.get(), &r,
                                                 PachnerMode::PreGeometric, false);
      applied = mv.propose() && mv.apply();
    } else {
      ::tessera::spacetime::IFlipMove mv(st.get(), &r,
                                                  PachnerMode::PreGeometric, false);
      applied = mv.propose() && mv.apply();
    }
  } else if (kind == "cone_out") {
    applied = SurgicalCone(st.get()).coneOut(spec.second).first;
  } else {
    applied = SurgicalCone(st.get()).coneIn(spec.second).first;
  }
  if (!applied) return false;
  std::set<std::uint64_t> live;
  for (const auto &s : st->getTopSimplices())
    for (auto v : topTuple(*s)) live.insert(v);
  for (auto v : inputVerts())
    if (!live.count(v)) return false;  // an input vertex was removed
  return EigenstateSynthesis(st, gateK_).dualComplexValid().first;
}

double EmergentOptimizer::deltaF(
    const std::shared_ptr<Spacetime> &cand, double baseRu,
    const std::set<std::vector<std::uint64_t>> &baseCells) const {
  std::set<std::vector<std::uint64_t>> candCells;
  for (const auto &s : cand->getTopSimplices()) candCells.insert(topTuple(*s));
  std::vector<std::vector<std::uint64_t>> touched;
  for (const auto &c : baseCells)
    if (!candCells.count(c)) touched.push_back(c);
  for (const auto &c : candCells)
    if (!baseCells.count(c)) touched.push_back(c);
  ReggeSolver baseSolver(st_, MatterConfiguration());
  ReggeSolver candSolver(cand, MatterConfiguration());
  std::set<std::pair<std::uint64_t, std::uint64_t>> edgeSet;
  for (const auto &p : baseSolver.affectedEdgesOfCells(touched)) edgeSet.insert(p);
  for (const auto &p : candSolver.affectedEdgesOfCells(touched)) edgeSet.insert(p);
  std::vector<std::pair<std::uint64_t, std::uint64_t>> edges(edgeSet.begin(),
                                                            edgeSet.end());
  const double dGrad = candSolver.gradientNorm2OverEdges(edges) -
                       baseSolver.gradientNorm2OverEdges(edges);
  const double dRu = rU(cand) - baseRu;
  return dGrad + gamma_ * dRu;
}

double EmergentOptimizer::step(int nCandidates) {
  const auto snap = snapshot();
  const double baseRu = rU(st_);
  std::set<std::vector<std::uint64_t>> baseCells;
  for (const auto &s : st_->getTopSimplices()) baseCells.insert(topTuple(*s));
  double bestDF = -tol_;
  bool haveBest = false;
  Snapshot bestSnap;
  for (int i = 0; i < nCandidates; ++i) {
    const auto spec = randomSpec(*st_);
    auto cand = build(snap);
    if (!applySpec(cand, spec)) continue;
    const double dF = deltaF(cand, baseRu, baseCells);
    if (dF < bestDF) {
      bestDF = dF;
      bestSnap = snapshotOf(*cand);
      haveBest = true;
    }
  }
  if (haveBest) {
    st_ = build(bestSnap);
    return bestDF;
  }
  return 0.0;
}

std::vector<double> EmergentOptimizer::runStage1(int maxSteps, int nCandidates,
                                                 int patience) {
  std::vector<double> trace = {objective()};
  int stalls = 0;
  for (int s = 0; s < maxSteps; ++s) {
    const double dF = step(nCandidates);
    trace.push_back(trace.back() + dF);
    if (dF >= -tol_) {
      ++stalls;
      rng_.seed(rng_());
      if (stalls >= patience) break;
    } else {
      stalls = 0;
    }
  }
  return trace;
}

void EmergentOptimizer::constructInputs(const std::vector<std::uint64_t> &seeds,
                                        int rounds) {
  // Region-restricted surgical solve per input: grow whatever emergent topology
  // in the seed's neighbourhood carries the input target (kept by Δr_input).
  for (std::size_t idx = 0;
       idx < inputTargets_.size() && idx < seeds.size(); ++idx) {
    const std::uint64_t seedV = seeds[idx];
    std::set<std::uint64_t> region;
    for (const auto &s : st_->getTopSimplices()) {
      auto c = topTuple(*s);
      if (std::find(c.begin(), c.end(), seedV) != c.end())
        region.insert(c.begin(), c.end());
    }
    Input inp{region, inputTargets_[idx]};
    double r = rInput(inp, st_);
    for (int round = 0; round < rounds; ++round) {
      const auto snap = snapshot();
      // a region-restricted move: cone on a cell inside the region.
      std::vector<std::vector<std::uint64_t>> regionCells;
      for (const auto &s : st_->getTopSimplices()) {
        auto c = topTuple(*s);
        bool inside = true;
        for (auto v : c)
          if (!region.count(v)) {
            inside = false;
            break;
          }
        if (inside) regionCells.push_back(std::move(c));
      }
      if (regionCells.empty()) break;
      const auto &cell = regionCells[rng_() % regionCells.size()];
      auto cand = build(snap);
      const bool applied = (rng_() % 2)
                               ? SurgicalCone(cand.get()).coneOut(cell).first
                               : SurgicalCone(cand.get()).coneIn(cell).first;
      if (!applied ||
          !EigenstateSynthesis(cand, gateK_).dualComplexValid().first)
        continue;
      const double rNew = rInput(inp, cand);
      if (rNew < r - tol_) {
        r = rNew;
        st_ = build(snapshotOf(*cand));
        // refresh the region with any new vertices the move added near the seed.
        for (const auto &s : st_->getTopSimplices()) {
          auto c = topTuple(*s);
          if (std::find(c.begin(), c.end(), seedV) != c.end())
            region.insert(c.begin(), c.end());
        }
        inp.verts = region;
      }
    }
    inputs_.push_back(inp);
  }
}

std::vector<double> EmergentOptimizer::relaxStage2(double beta, int maxIters,
                                                   double alpha0) {
  auto edges = st_->getEdgeList()->toVector();
  const std::size_t n = edges.size();
  auto fullF = [&]() { return beta * gradNorm2(st_) + gamma_ * rU(st_); };
  std::vector<double> trace = {fullF()};
  double alpha = alpha0;
  for (int it = 0; it < maxIters; ++it) {
    ReggeSolver rs(st_, MatterConfiguration());
    const auto gv = rs.actionGradientExact();
    const auto hmat = rs.actionHessianExact();  // vector<vector<complex>> (rows)
    Eigen::VectorXcd g(n);
    for (std::size_t i = 0; i < n; ++i) g(i) = gv[i];
    Eigen::MatrixXcd H(n, n);
    for (std::size_t r = 0; r < n; ++r)
      for (std::size_t c = 0; c < n; ++c) H(r, c) = hmat[r][c];
    const Eigen::VectorXcd grad = beta * 2.0 * (H.conjugate() * g);
    Eigen::VectorXcd l2(n);
    for (std::size_t i = 0; i < n; ++i) l2(i) = edges[i]->getSquaredLength();
    const double f0 = trace.back();
    double stepSize = alpha;
    bool improved = false;
    for (int ls = 0; ls < 24; ++ls) {
      for (std::size_t i = 0; i < n; ++i) {
        cd v = l2(i) - stepSize * grad(i);
        double re = std::min(std::max(v.real(), 0.05), 20.0);  // bound real part
        edges[i]->setSquaredLength(cd(re, v.imag()));
      }
      double f1;
      try {
        f1 = fullF();
      } catch (...) {
        f1 = std::numeric_limits<double>::infinity();
      }
      if (f1 < f0 - tol_) {
        trace.push_back(f1);
        alpha = std::min(alpha * 1.3, 1.0);
        improved = true;
        break;
      }
      stepSize *= 0.5;
    }
    if (!improved) {
      for (std::size_t i = 0; i < n; ++i) edges[i]->setSquaredLength(l2(i));
      break;
    }
  }
  return trace;
}

}  // namespace tessera::cobordism
