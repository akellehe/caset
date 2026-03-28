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

    // --- Target deficit angles ---
    std::vector<double> target_deficits;      // [n_hinges]
};

/// Compute deficit angles for all hinges on the GPU.
void compute_deficits_gpu(const GpuMeshData &mesh, double *h_deficits);

/// Compute numerical gradients for all edges in parallel on the GPU.
/// Each thread perturbs one edge's sq_dist_flat entries, recomputes the
/// partial residual (only affected hinges), and outputs the gradient.
void compute_gradients_gpu(const GpuMeshData &mesh,
                           double *h_gradients);

} // namespace cuda
} // namespace caset
