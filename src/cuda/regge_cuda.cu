// MIT License -- Copyright (c) 2025 Andrew Kelleher
//
// CUDA-accelerated Regge solver: deficit angles and numerical gradients.
//
// The mesh topology is flattened into contiguous CSR arrays on the host,
// uploaded once, and all computation runs in GPU kernels.

#include "cuda/regge_cuda.h"
#include <cmath>
#include <cstdio>

namespace caset {
namespace cuda {

// =====================================================================
// Device helpers: small dense linear algebra
// =====================================================================

__device__ double det_small(const double *M, int n) {
    double A[MAX_DIM * MAX_DIM];
    for (int i = 0; i < n * n; ++i) A[i] = M[i];
    double d = 1.0;
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double maxVal = fabs(A[col * n + col]);
        for (int row = col + 1; row < n; ++row) {
            double v = fabs(A[row * n + col]);
            if (v > maxVal) { maxVal = v; pivot = row; }
        }
        if (maxVal < 1e-15) return 0.0;
        if (pivot != col) {
            for (int j = 0; j < n; ++j) {
                double tmp = A[col * n + j];
                A[col * n + j] = A[pivot * n + j];
                A[pivot * n + j] = tmp;
            }
            d = -d;
        }
        d *= A[col * n + col];
        for (int row = col + 1; row < n; ++row) {
            double f = A[row * n + col] / A[col * n + col];
            for (int j = col + 1; j < n; ++j)
                A[row * n + j] -= f * A[col * n + j];
        }
    }
    return d;
}

__device__ double cofactor_ij(const double *M, int n, int i, int j) {
    double sub[(MAX_DIM - 1) * (MAX_DIM - 1)];
    int si = 0;
    for (int r = 0; r < n; ++r) {
        if (r == i) continue;
        int sj = 0;
        for (int c = 0; c < n; ++c) {
            if (c == j) continue;
            sub[si * (n - 1) + sj] = M[r * n + c];
            sj++;
        }
        si++;
    }
    double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
    return sign * det_small(sub, n - 1);
}

// Compute dihedral angle from a Cayley-Menger bordered matrix B of size n×n,
// at the hinge opposite local vertices va, vb (0-indexed simplex vertices).
__device__ double dihedral_from_cm(const double *B, int n, int va, int vb) {
    int bi = va + 1; // shift for border row/col
    int bj = vb + 1;
    double Cij = cofactor_ij(B, n, bi, bj);
    double Cii = cofactor_ij(B, n, bi, bi);
    double Cjj = cofactor_ij(B, n, bj, bj);
    double denom = sqrt(fabs(Cii * Cjj));
    if (denom < 1e-15) return 0.0;
    double cosTheta = -Cij / denom;
    cosTheta = fmax(-1.0, fmin(1.0, cosTheta));
    return acos(cosTheta);
}

// Build a Cayley-Menger bordered matrix from a flat nv×nv squared-distance
// matrix.  Output B is (nv+1)×(nv+1).
__device__ void build_cm(const double *sq_dist, int nv, double *B) {
    int n = nv + 1;
    for (int i = 0; i < n * n; ++i) B[i] = 0.0;
    for (int k = 1; k < n; ++k) {
        B[0 * n + k] = 1.0;
        B[k * n + 0] = 1.0;
    }
    for (int i = 0; i < nv; ++i)
        for (int j = 0; j < nv; ++j)
            B[(i + 1) * n + (j + 1)] = sq_dist[i * nv + j];
}

// Compute deficit angle for one hinge, reading from (possibly perturbed)
// sq_dist_flat.
__device__ double compute_deficit(
    int hid,
    const int *hinge_simplex_offsets,
    const int *hinge_simplex_ids,
    const int *hinge_opposite_a,
    const int *hinge_opposite_b,
    const int *simplex_sq_dist_offsets,
    const double *sq_dist_flat,
    const int *simplex_n_verts
) {
    int s_start = hinge_simplex_offsets[hid];
    int s_end = hinge_simplex_offsets[hid + 1];
    double angle_sum = 0.0;

    for (int si = s_start; si < s_end; ++si) {
        int simplex_id = hinge_simplex_ids[si];
        int nv = simplex_n_verts[simplex_id];
        int sd_off = simplex_sq_dist_offsets[simplex_id];

        double B[MAX_DIM * MAX_DIM];
        build_cm(sq_dist_flat + sd_off, nv, B);

        angle_sum += dihedral_from_cm(B, nv + 1,
                                       hinge_opposite_a[si],
                                       hinge_opposite_b[si]);
    }
    return 2.0 * M_PI - angle_sum;
}

// =====================================================================
// Deficit kernel: one thread per hinge
// =====================================================================

__global__ void deficit_kernel(
    const int *hinge_simplex_offsets,
    const int *hinge_simplex_ids,
    const int *hinge_opposite_a,
    const int *hinge_opposite_b,
    const int *simplex_sq_dist_offsets,
    const double *simplex_sq_dist_flat,
    const int *simplex_n_verts,
    double *deficit_angles,
    int n_hinges
) {
    int hid = blockIdx.x * blockDim.x + threadIdx.x;
    if (hid >= n_hinges) return;
    deficit_angles[hid] = compute_deficit(
        hid, hinge_simplex_offsets, hinge_simplex_ids,
        hinge_opposite_a, hinge_opposite_b,
        simplex_sq_dist_offsets, simplex_sq_dist_flat, simplex_n_verts);
}

// =====================================================================
// Gradient kernel: one thread per edge
// =====================================================================

__global__ void gradient_kernel(
    const double *sq_dist_flat,         // [total_sq] base squared distances
    const int *hinge_simplex_offsets,
    const int *hinge_simplex_ids,
    const int *hinge_opposite_a,
    const int *hinge_opposite_b,
    const int *simplex_sq_dist_offsets,
    const int *simplex_n_verts,
    const int *edge_hinge_offsets,       // [n_edges+1] CSR
    const int *edge_hinge_ids,           // affected hinge indices
    const int *edge_dist_offsets,        // [n_edges+1] CSR
    const int *edge_dist_positions,      // positions in sq_dist_flat to perturb
    const double *target_deficits,       // [n_hinges]
    const double *base_deficits,         // [n_hinges] pre-computed unperturbed
    double *gradients,                   // [n_edges] output
    int n_edges,
    int total_sq
) {
    int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= n_edges) return;

    // Which hinges does this edge affect?
    int h_start = edge_hinge_offsets[eid];
    int h_end = edge_hinge_offsets[eid + 1];
    if (h_start == h_end) { gradients[eid] = 0.0; return; }

    // Which positions in sq_dist_flat to perturb?
    int d_start = edge_dist_offsets[eid];
    int d_end = edge_dist_offsets[eid + 1];

    // Determine perturbation size from the first entry
    double origSq = (d_start < d_end) ? sq_dist_flat[edge_dist_positions[d_start]] : 1.0;
    double h = fmax(fabs(origSq) * 1e-4, 1e-8);

    // Compute partial residual (unperturbed) for affected hinges
    double L0 = 0.0;
    for (int hi = h_start; hi < h_end; ++hi) {
        int hinge_id = edge_hinge_ids[hi];
        double diff = base_deficits[hinge_id] - target_deficits[hinge_id];
        L0 += diff * diff;
    }

    // Make a local copy of the affected sq_dist entries, perturbed
    // We can't modify the global array (other threads read it), so we
    // recompute each affected hinge with the perturbation applied inline.
    double Lp = 0.0;
    for (int hi = h_start; hi < h_end; ++hi) {
        int hinge_id = edge_hinge_ids[hi];

        // Recompute deficit for this hinge with perturbed distances
        int s_start = hinge_simplex_offsets[hinge_id];
        int s_end = hinge_simplex_offsets[hinge_id + 1];
        double angle_sum = 0.0;

        for (int si = s_start; si < s_end; ++si) {
            int simplex_id = hinge_simplex_ids[si];
            int nv = simplex_n_verts[simplex_id];
            int sd_off = simplex_sq_dist_offsets[simplex_id];

            // Copy this simplex's distance matrix to local memory
            double local_sq[MAX_DIM * MAX_DIM];
            for (int i = 0; i < nv * nv; ++i)
                local_sq[i] = sq_dist_flat[sd_off + i];

            // Apply perturbation: add h to entries matching this edge
            for (int di = d_start; di < d_end; ++di) {
                int pos = edge_dist_positions[di];
                if (pos >= sd_off && pos < sd_off + nv * nv)
                    local_sq[pos - sd_off] += h;
            }

            double B[MAX_DIM * MAX_DIM];
            build_cm(local_sq, nv, B);
            angle_sum += dihedral_from_cm(B, nv + 1,
                                           hinge_opposite_a[si],
                                           hinge_opposite_b[si]);
        }

        double deficit_p = 2.0 * M_PI - angle_sum;
        double diff = deficit_p - target_deficits[hinge_id];
        Lp += diff * diff;
    }

    gradients[eid] = (Lp - L0) / h;
}

// =====================================================================
// Host-side: allocate, upload, launch, download, free
// =====================================================================

struct GpuArrays {
    int *hinge_simplex_offsets, *hinge_simplex_ids;
    int *hinge_opposite_a, *hinge_opposite_b;
    int *simplex_sq_dist_offsets, *simplex_n_verts;
    double *simplex_sq_dist_flat;
    int *edge_hinge_offsets, *edge_hinge_ids;
    int *edge_dist_offsets, *edge_dist_positions;
    double *target_deficits, *base_deficits;
    double *deficit_angles, *gradients;

    void alloc(const GpuMeshData &m) {
        int nh = m.n_hinges, ns = m.n_simplices, ne = m.n_edges;
        int nnz_hs = (int)m.hinge_simplex_ids.size();
        int nnz_eh = (int)m.edge_hinge_ids.size();
        int nnz_ed = (int)m.edge_dist_positions.size();
        int sq_size = (int)m.simplex_sq_dist_flat.size();

        cudaMalloc(&hinge_simplex_offsets, (nh+1)*sizeof(int));
        cudaMalloc(&hinge_simplex_ids, nnz_hs*sizeof(int));
        cudaMalloc(&hinge_opposite_a, nnz_hs*sizeof(int));
        cudaMalloc(&hinge_opposite_b, nnz_hs*sizeof(int));
        cudaMalloc(&simplex_sq_dist_offsets, (ns+1)*sizeof(int));
        cudaMalloc(&simplex_sq_dist_flat, sq_size*sizeof(double));
        cudaMalloc(&simplex_n_verts, ns*sizeof(int));
        cudaMalloc(&edge_hinge_offsets, (ne+1)*sizeof(int));
        cudaMalloc(&edge_hinge_ids, nnz_eh*sizeof(int));
        cudaMalloc(&edge_dist_offsets, (ne+1)*sizeof(int));
        cudaMalloc(&edge_dist_positions, nnz_ed*sizeof(int));
        cudaMalloc(&target_deficits, nh*sizeof(double));
        cudaMalloc(&base_deficits, nh*sizeof(double));
        cudaMalloc(&deficit_angles, nh*sizeof(double));
        cudaMalloc(&gradients, ne*sizeof(double));
    }

    void upload(const GpuMeshData &m) {
        int nh = m.n_hinges, ns = m.n_simplices, ne = m.n_edges;
        #define UP(dst, src, n) cudaMemcpy(dst, src, (n), cudaMemcpyHostToDevice)
        UP(hinge_simplex_offsets, m.hinge_simplex_offsets.data(), (nh+1)*sizeof(int));
        UP(hinge_simplex_ids, m.hinge_simplex_ids.data(), m.hinge_simplex_ids.size()*sizeof(int));
        UP(hinge_opposite_a, m.hinge_opposite_a.data(), m.hinge_opposite_a.size()*sizeof(int));
        UP(hinge_opposite_b, m.hinge_opposite_b.data(), m.hinge_opposite_b.size()*sizeof(int));
        UP(simplex_sq_dist_offsets, m.simplex_sq_dist_offsets.data(), (ns+1)*sizeof(int));
        UP(simplex_sq_dist_flat, m.simplex_sq_dist_flat.data(), m.simplex_sq_dist_flat.size()*sizeof(double));
        UP(simplex_n_verts, m.simplex_n_verts.data(), ns*sizeof(int));
        UP(edge_hinge_offsets, m.edge_hinge_offsets.data(), (ne+1)*sizeof(int));
        UP(edge_hinge_ids, m.edge_hinge_ids.data(), m.edge_hinge_ids.size()*sizeof(int));
        UP(edge_dist_offsets, m.edge_dist_offsets.data(), (ne+1)*sizeof(int));
        UP(edge_dist_positions, m.edge_dist_positions.data(), m.edge_dist_positions.size()*sizeof(int));
        UP(target_deficits, m.target_deficits.data(), nh*sizeof(double));
        #undef UP
    }

    void free() {
        cudaFree(hinge_simplex_offsets); cudaFree(hinge_simplex_ids);
        cudaFree(hinge_opposite_a); cudaFree(hinge_opposite_b);
        cudaFree(simplex_sq_dist_offsets); cudaFree(simplex_sq_dist_flat);
        cudaFree(simplex_n_verts);
        cudaFree(edge_hinge_offsets); cudaFree(edge_hinge_ids);
        cudaFree(edge_dist_offsets); cudaFree(edge_dist_positions);
        cudaFree(target_deficits); cudaFree(base_deficits);
        cudaFree(deficit_angles); cudaFree(gradients);
    }
};

void compute_deficits_gpu(const GpuMeshData &mesh, double *h_deficits) {
    GpuArrays g;
    g.alloc(mesh);
    g.upload(mesh);

    int nh = mesh.n_hinges;
    int bs = 256, gs = (nh + bs - 1) / bs;
    deficit_kernel<<<gs, bs>>>(
        g.hinge_simplex_offsets, g.hinge_simplex_ids,
        g.hinge_opposite_a, g.hinge_opposite_b,
        g.simplex_sq_dist_offsets, g.simplex_sq_dist_flat,
        g.simplex_n_verts, g.deficit_angles, nh);
    cudaDeviceSynchronize();
    cudaMemcpy(h_deficits, g.deficit_angles, nh*sizeof(double), cudaMemcpyDeviceToHost);
    g.free();
}

void compute_gradients_gpu(const GpuMeshData &mesh, double *h_gradients) {
    GpuArrays g;
    g.alloc(mesh);
    g.upload(mesh);

    int nh = mesh.n_hinges, ne = mesh.n_edges;
    int total_sq = (int)mesh.simplex_sq_dist_flat.size();
    int bs = 256;

    // Step 1: compute base deficit angles
    int gs_h = (nh + bs - 1) / bs;
    deficit_kernel<<<gs_h, bs>>>(
        g.hinge_simplex_offsets, g.hinge_simplex_ids,
        g.hinge_opposite_a, g.hinge_opposite_b,
        g.simplex_sq_dist_offsets, g.simplex_sq_dist_flat,
        g.simplex_n_verts, g.base_deficits, nh);

    // Step 2: compute all gradients in parallel
    int gs_e = (ne + bs - 1) / bs;
    gradient_kernel<<<gs_e, bs>>>(
        g.simplex_sq_dist_flat,
        g.hinge_simplex_offsets, g.hinge_simplex_ids,
        g.hinge_opposite_a, g.hinge_opposite_b,
        g.simplex_sq_dist_offsets, g.simplex_n_verts,
        g.edge_hinge_offsets, g.edge_hinge_ids,
        g.edge_dist_offsets, g.edge_dist_positions,
        g.target_deficits, g.base_deficits,
        g.gradients, ne, total_sq);

    cudaDeviceSynchronize();
    cudaMemcpy(h_gradients, g.gradients, ne*sizeof(double), cudaMemcpyDeviceToHost);
    g.free();
}

} // namespace cuda
} // namespace caset
