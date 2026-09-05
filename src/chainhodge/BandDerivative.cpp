// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "chainhodge/BandDerivative.h"

#include <stdexcept>

namespace tessera::chainhodge {

BandDerivative::ResolventFrames BandDerivative::resolventFrames(const CovariantChainHodge &cov, int k,
                                                                const Contour &contour,
                                                                const Eigen::MatrixXcd &frame) {
  if (contour.nodes.size() != contour.weights.size() || contour.nodes.empty())
    throw std::invalid_argument("BandDerivative: a contour needs matching nodes and weights");
  ResolventFrames out;
  out.degree = k;
  out.contour = contour;
  out.frame = frame;
  out.applied.reserve(contour.nodes.size());
  for (const auto &zeta : contour.nodes) out.applied.push_back(cov.resolvent(k, zeta, frame));
  return out;
}

Eigen::MatrixXcd BandDerivative::projectorDerivativeApplied(const CovariantChainHodge &cov,
                                                            const ResolventFrames &frames,
                                                            const Eigen::MatrixXcd &dh) {
  Eigen::MatrixXcd out = Eigen::MatrixXcd::Zero(frames.frame.rows(), frames.frame.cols());
  for (std::size_t j = 0; j < frames.contour.nodes.size(); ++j)
    out += frames.contour.weights[j] * cov.resolvent(frames.degree, frames.contour.nodes[j], dh * frames.applied[j]);
  return out;
}

Eigen::MatrixXcd BandDerivative::imagesDerivative(const CovariantChainHodge &cov, const ResolventFrames &frames,
                                                  const Eigen::MatrixXcd &images, const SparseMatrix &dM,
                                                  const Eigen::MatrixXcd &dh) {
  const Eigen::MatrixXcd inner = -(dM * images) + projectorDerivativeApplied(cov, frames, dh);
  return cov.applyG(frames.degree, inner);
}

Eigen::MatrixXcd BandDerivative::imagesLengthDerivative(const CovariantChainHodge &cov,
                                                        const ResolventFrames &frames,
                                                        const Eigen::MatrixXcd &images,
                                                        std::size_t edgeIndex) {
  return imagesDerivative(cov, frames, images, cov.dressedDerivative(frames.degree, edgeIndex),
                          cov.covariantOperatorDerivative(frames.degree, edgeIndex));
}

Eigen::MatrixXcd BandDerivative::imagesPhaseDerivative(const CovariantChainHodge &cov,
                                                       const ResolventFrames &frames,
                                                       const Eigen::MatrixXcd &images,
                                                       std::size_t edgeIndex) {
  return imagesDerivative(cov, frames, images, cov.dressedPhaseDerivative(frames.degree, edgeIndex),
                          cov.covariantOperatorPhaseDerivative(frames.degree, edgeIndex));
}

// The covariant operator is h = A~^U (M^U)^{-1} (the resolvent identity
// (zeta I - h)^{-1} = M^U (zeta M^U - A~^U)^{-1}), so A~^U = h M^U and
// d A~^U = dh M^U + h dM^U.

Eigen::MatrixXcd BandDerivative::pencilOperatorLengthDerivative(const CovariantChainHodge &cov, int k,
                                                                std::size_t edgeIndex) {
  const Eigen::MatrixXcd h = cov.covariantOperator(k);
  const Eigen::MatrixXcd M(cov.Minv(k));
  const Eigen::MatrixXcd dM(cov.dressedDerivative(k, edgeIndex));
  return cov.covariantOperatorDerivative(k, edgeIndex) * M + h * dM;
}

Eigen::MatrixXcd BandDerivative::pencilOperatorPhaseDerivative(const CovariantChainHodge &cov, int k,
                                                               std::size_t edgeIndex) {
  const Eigen::MatrixXcd h = cov.covariantOperator(k);
  const Eigen::MatrixXcd M(cov.Minv(k));
  const Eigen::MatrixXcd dM(cov.dressedPhaseDerivative(k, edgeIndex));
  return cov.covariantOperatorPhaseDerivative(k, edgeIndex) * M + h * dM;
}

}  // namespace tessera::chainhodge
