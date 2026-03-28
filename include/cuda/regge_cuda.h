// MIT License -- Copyright (c) 2025 Andrew Kelleher
#pragma once

#include <vector>
#include <cstdint>

namespace caset {
namespace cuda {

/// Maximum Cayley-Menger matrix dimension (d+2 for d-simplices).
/// For 4D CDT: 5 vertices → 6×6 bordered matrix. Extra margin for safety.
constexpr int MAX_DIM = 8;

/// Flattened mesh topology for GPU computation.
///
/// All connectivity is encoded as CSR (Compressed Sparse Row) arrays
/// so the GPU can traverse the mesh without pointer chasing.
struct GpuMeshData {
    int n_hinges = 0;
    int n_simplices = 0;
    int n_edges = 0;

    // --- Per-simplex squared distance matrices (flattened) ---
    std::vector<int> simplex_sq_dist_offsets; // [n_simplices+1] offset into sq_dist_flat
    std::vector<double> simplex_sq_dist_flat; // flattened nv×nv distance matrices
    std::vector<int> simplex_n_verts;         // [n_simplices]

    // --- Hinge → simplex incidence (CSR) ---
    std::vector<int> hinge_simplex_offsets;   // [n_hinges+1]
    std::vector<int> hinge_simplex_ids;       // simplex index per entry
    std::vector<int> hinge_opposite_a;        // local vertex idx a opposite hinge
    std::vector<int> hinge_opposite_b;        // local vertex idx b opposite hinge

    // --- Edge → affected hinges (CSR) ---
    std::vector<int> edge_hinge_offsets;      // [n_edges+1]
    std::vector<int> edge_hinge_ids;          // hinge indices

    // --- Edge → positions in sq_dist_flat to perturb (CSR) ---
    // Each edge appears as D[i][j] and D[j][i] in each simplex that contains it.
    std::vector<int> edge_dist_offsets;       // [n_edges+1]
    std::vector<int> edge_dist_positions;     // absolute positions in sq_dist_flat

    // --- Edge → neighbor edges (CSR) ---
    // Two edges are neighbors if they share at least one hinge.  When one
    // edge is perturbed, only its neighbors' action gradients change.
    // Each edge is its own neighbor (self-loop included).
    std::vector<int> edge_nbr_offsets;        // [n_edges+1]
    std::vector<int> edge_nbr_ids;            // neighbor edge indices

    // --- Target deficit angles ---
    std::vector<double> target_deficits;      // [n_hinges]

    // --- Precomputed base hinge contributions: A_h * ε_h per hinge ---
    std::vector<double> base_hinge_contribs;  // [n_hinges]

    // --- Worldline (matter) ---
    std::vector<int> worldline_edge_mask;     // [n_edges] 1 if on worldline
    double worldline_mass = 0.0;
};

/// Compute deficit angles for all hinges on the GPU.
void compute_deficits_gpu(const GpuMeshData &mesh, double *h_deficits);

/// Compute numerical gradients of the deficit-residual Σ(ε_h - ε_target)²
/// for all edges in parallel on the GPU.
void compute_gradients_gpu(const GpuMeshData &mesh,
                           double *h_gradients);

/// Compute the action gradient ∂S/∂ℓ²_e for all edges in parallel,
/// where S = Σ A_h·ε_h + S_matter.  This is the correct gradient for
/// the Regge equation solver's step() function.
void compute_action_gradient_gpu(const GpuMeshData &mesh,
                                 double *h_gradients);

/// Compute one full solver step on the GPU:
///   1. Base action gradient ∂S/∂W_e  (1 kernel launch)
///   2. Fused ∂F/∂W_e where F = ||∇S||²  (1 kernel launch)
/// All GPU memory is allocated once, uploaded once, downloaded once.
/// h_base_grad[n_edges] receives ∂S/∂W_e.
/// h_dF[n_edges] receives ∂F/∂W_e.
void compute_step_gpu(const GpuMeshData &mesh,
                      double *h_base_grad,
                      double *h_dF);

} // namespace cuda
} // namespace caset
