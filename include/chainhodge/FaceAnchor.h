// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_CHAINHODGE_FACEANCHOR_H
#define TESSERA_CHAINHODGE_FACEANCHOR_H

#include <complex>
#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "chainhodge/CovariantChainHodge.h"
#include "chainhodge/WhitneyMass.h"
#include "cobordism/ChainComplex.h"

namespace tessera::chainhodge {

/// One triangle's face block: the \f$ 3\times3 \f$ matrix over the triangle's
/// three edges (canonical \f$ C_1 \f$ indices in `edgeIndices`, in the local
/// order \f$ (v_0v_1), (v_0v_2), (v_1v_2) \f$), its numerical rank, and the
/// preset it was assembled under.
struct FaceBlock {
  /// Index of the triangle in `ChainComplex::kSimplexVertices(2)`.
  std::size_t faceIndex{0};
  std::vector<int> edgeIndices{};
  Eigen::MatrixXcd block{};
  /// Numerical rank at tolerance \f$ \kappa\,3\,\epsilon_m\,\sigma_{\max} \f$
  /// with \f$ \kappa = 10 \f$.
  int rank{0};
  Preset preset{Preset::L2};
};

/// # FaceAnchor
///
/// The face anchor of specification §8, realized with the Whitney metric.
///
/// For a nondegenerate triangle \f$ t \f$ the Whitney block
/// \f$ M_1^{(t)}\in\mathbb{C}^{3\times3} \f$ — the contribution of \f$ t \f$ to
/// \f$ M_1 \f$, restricted to its three edges — is nonsingular (Prop. 8.1):
/// the three Whitney 1-forms of \f$ t \f$ are linearly independent functions on
/// \f$ t \f$ and \f$ M_1^{(t)} \f$ is their Gram matrix in a nondegenerate
/// metric. At \f$ d = 2 \f$ the triangle is a top simplex and the block is its
/// local block. At \f$ d\ge3 \f$ the specification writes the block for a
/// triangle only; here it is read as the sum of the local \f$ M_1 \f$ blocks of
/// every top simplex containing \f$ t \f$, restricted to \f$ t \f$'s three
/// edges, i.e. the part of \f$ M_1 \f$'s assembly that \f$ t \f$'s edges receive
/// from the top simplices through \f$ t \f$. That reading is this class's, not
/// the specification's text.
///
/// With the connection, the connection-dressed face endomorphism on chains is
/// \f[
///   \Pi_\tau(U) = G_1^U\,M_1^{(\tau)U}\,G_1^U,\qquad
///   (M_1^{(\tau)U})_{ee'} = (M_1^{(\tau)})_{ee'}\,U_{b(e)b(e')},
/// \f]
/// covariant by the argument of Prop. 5.1(iii), applied by solves and never
/// formed densely; and the invariant anchor coordinate of a fiber
/// \f$ Q \f$ with right frame \f$ \Phi_Q \f$ (chains) and dual frame
/// \f$ \Phi_Q^\vee \f$ (the same band for \f$ U^{-1} \f$) is
/// \f[
///   \alpha_\tau = \det\bigl((\Phi_Q^\vee)^T\,\Pi_\tau(U)\,\Phi_Q\bigr)
///               = \det\bigl((Z_Q^\vee)^T\,M_1^{(\tau)U}\,Z_Q\bigr),\qquad
///   Z_Q = G_1^U\Phi_Q,\ \ Z_Q^\vee = G_1^{U^{-1}}\Phi_Q^\vee ,
/// \f]
/// the face block paired through the geometric images (the two forms agree
/// exactly because \f$ (G_1^U)^T = G_1^{U^{-1}} \f$). It is a well-defined
/// complex number, invariant under the paired frame gauge
/// \f$ \Phi\mapsto\Phi g \f$, \f$ \Phi^\vee\mapsto\Phi^\vee g^{-T} \f$ (at
/// \f$ U = 1 \f$ the \f$ O(r,\mathbb{C}) \f$ gauge of one frame) and under a
/// vertex gauge \f$ g \f$ through \f$ \rho_1(g) \f$, and generically nonzero
/// for a rank-three fiber. Under the Grassmann preset the per-face blade
/// block has rank two, so the same determinant vanishes identically for any
/// rank-three fiber. No conjugation appears anywhere; the pairing is the
/// transpose.
class FaceAnchor {
 public:
  /// \f$ M_1^{(t)} \f$ of the triangle at canonical index \p faceIndex under the
  /// Whitney metric (sum over the top simplices containing \f$ t \f$ of their
  /// local blocks, restricted to \f$ t \f$'s edges).
  /// @throws std::invalid_argument on a bad index, a complex without
  ///   triangles, or a squared-length vector of the wrong size.
  [[nodiscard]] static FaceBlock whitneyFaceBlock(const cobordism::ChainComplex &K,
                                                  const SquaredLengths &s, std::size_t faceIndex,
                                                  Branch branch = Branch::Continuation);
  /// Every triangle's Whitney block, in canonical triangle order.
  [[nodiscard]] static std::vector<FaceBlock> whitneyFaceBlocks(const cobordism::ChainComplex &K,
                                                                const SquaredLengths &s,
                                                                Branch branch = Branch::Continuation);
  /// The Grassmann per-face blade block: \f$ \langle u_e, u_{e'}\rangle \f$ of the
  /// triangle's three edge vectors by the polarization identity (CH §6),
  /// rank two for a nondegenerate triangle.
  [[nodiscard]] static FaceBlock grassmannFaceBlock(const cobordism::ChainComplex &K,
                                                    const SquaredLengths &s, std::size_t faceIndex);
  /// The face block of the instance's own preset: Whitney for `Preset::L2`,
  /// Grassmann for `Preset::GRASSMANN_ALL`.
  [[nodiscard]] static FaceBlock faceBlock(const ChainHodge &hodge, std::size_t faceIndex);

  /// \f$ M_1^{(\tau)U} \f$: the block dressed entrywise by \f$ U_{b(e)b(e')} \f$,
  /// \f$ b(e) = \min e \f$.
  [[nodiscard]] static Eigen::MatrixXcd dressedFaceBlock(const FaceBlock &block,
                                                         const cobordism::ChainComplex &K,
                                                         const Connection &U);

  /// \f$ \Pi_\tau(U)\,c = G_1^U\,M_1^{(\tau)U}\,G_1^U\,c \f$ for chains \p c
  /// (\f$ n_1\times m \f$), by two solves and one \f$ 3\times3 \f$ product; the
  /// block acts on the triangle's three edge coordinates and annihilates the
  /// rest.
  [[nodiscard]] static Eigen::MatrixXcd applyFaceEndomorphism(const CovariantChainHodge &cov,
                                                              std::size_t faceIndex,
                                                              const Eigen::MatrixXcd &c);

  /// \f$ \alpha_\tau = \det\bigl((Z_Q^\vee)^T M_1^{(\tau)U} Z_Q\bigr) \f$ from the
  /// fiber's geometric images \p Zdual (\f$ Z_Q^\vee = G_1^{U^{-1}}\Phi_Q^\vee \f$)
  /// and \p Z (\f$ Z_Q = G_1^U\Phi_Q \f$), each \f$ n_1\times r \f$.
  [[nodiscard]] static Complex anchorCoordinate(const CovariantChainHodge &cov, std::size_t faceIndex,
                                                const Eigen::MatrixXcd &Zdual,
                                                const Eigen::MatrixXcd &Z);
  /// The same from the chain frames \p PhiDual and \p Phi:
  /// \f$ \alpha_\tau = \det((\Phi_Q^\vee)^T\Pi_\tau(U)\Phi_Q) \f$, the images
  /// formed by solves with \f$ M_1^{U^{-1}} \f$ and \f$ M_1^U \f$.
  [[nodiscard]] static Complex anchorCoordinateFromChains(const CovariantChainHodge &cov,
                                                          std::size_t faceIndex,
                                                          const Eigen::MatrixXcd &PhiDual,
                                                          const Eigen::MatrixXcd &Phi);
  /// \f$ \alpha_\tau \f$ for every triangle, canonical order.
  [[nodiscard]] static std::vector<Complex> anchorCoordinates(const CovariantChainHodge &cov,
                                                              const Eigen::MatrixXcd &Zdual,
                                                              const Eigen::MatrixXcd &Z);

  /// Numerical rank of a small dense block at tolerance
  /// \f$ \kappa\,\max(m,n)\,\epsilon_m\,\sigma_{\max} \f$.
  [[nodiscard]] static int numericalRank(const Eigen::MatrixXcd &A, double kappa = 10.0);

 private:
  [[nodiscard]] static std::vector<int> triangleEdgeIndices(const cobordism::ChainComplex &K,
                                                            std::size_t faceIndex);
};

}  // namespace tessera::chainhodge

#endif  // TESSERA_CHAINHODGE_FACEANCHOR_H
