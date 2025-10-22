//
// Created by andrew on 10/21/25.
//

#include "Simplex.h"

namespace caset {

const double Simplex::getDeficitAngle() const {
  return 0;
}

const std::vector<std::shared_ptr<Simplex>> Simplex::getHinges() const {
  return {};
}

const double Simplex::computeDihedralAngles() const {
  return 0.;
}

torch::Tensor Simplex::getGramMatrix() const {
  // Compute the Gram matrix G from the edge lengths.
  torch::Tensor gramMatrix;
  for (int i=0; i<edges.size(); i++) {
    for (int j=0; j<edges.size(); j++) {
      // G_ij = 1/2 (l_0i^2 + l_0j^2 - l_ij^2)
      double l_0i = edges[i]->getLength();
      double l_0j = edges[j]->getLength();
      double l_ij = edges[std::abs(i - j)]->getLength(); // Placeholder; replace with actual edge lookup
      gramMatrix[i][j] = 0.5 * (l_0i * l_0i + l_0j * l_0j - l_ij * l_ij);
    }
  }
  return gramMatrix;
}

torch::Tensor Simplex::getGramCofactor(const torch::Tensor &gramMatrix) const {
  TORCH_CHECK(gramMatrix.dim() >= 2, "G must be at least 2D");
  auto n = gramMatrix.size(-1);
  TORCH_CHECK(gramMatrix.size(-2) == n, "G must be square in the last two dims");
  TORCH_CHECK(gramMatrix.dtype() == torch::kFloat || gramMatrix.dtype() == torch::kDouble,
              "G must be float/double");

  // Cholesky factorization (lower-triangular L with G = L L^T)
  // If your data might include near-singular simplices, catch exceptions and handle them.
  auto L = torch::linalg_cholesky(gramMatrix);

  // det(G) = (prod(diag(L)))^2 ; compute in log-space to avoid overflow
  auto diagL = L.diagonal(0, -2, -1);
  auto logdetG = 2.0 * diagL.abs().log().sum(/*dim=*/-1, /*keepdim=*/true);
  auto detG = logdetG.exp();                                // shape (..., 1)

  // Inverse via Cholesky (much stabler than generic inverse for PD matrices)
  auto Ginv = torch::cholesky_inverse(L);            // (..., n, n)

  // Cofactor = det(G) * (G^{-1})^T
  auto cof = detG.unsqueeze(-1) * Ginv.transpose(-2, -1);   // broadcast det
  return cof;
}



}