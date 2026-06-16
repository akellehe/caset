// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
//
// CUDA-accelerated Regge solver: deficit angles and numerical gradients.
//
// The mesh topology is flattened into contiguous CSR arrays on the host,
// uploaded once, and all computation runs in GPU kernels.

#include "cuda/regge_cuda.h"
#include <cmath>
#include <cstdio>
#include <stdexcept>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        char msg[256]; \
        snprintf(msg, sizeof(msg), "CUDA error at %s:%d: %s", \
                 __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error(msg); \
    } \
} while (0)

#define CUDA_CHECK_KERNEL() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        char msg[256]; \
        snprintf(msg, sizeof(msg), "CUDA kernel error at %s:%d: %s", \
                 __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error(msg); \
    } \
} while (0)

namespace tessera {
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
    int *hinge_simplex_offsets = nullptr, *hinge_simplex_ids = nullptr;
    int *hinge_opposite_a = nullptr, *hinge_opposite_b = nullptr;
    int *simplex_sq_dist_offsets = nullptr, *simplex_n_verts = nullptr;
    double *simplex_sq_dist_flat = nullptr;
    int *edge_hinge_offsets = nullptr, *edge_hinge_ids = nullptr;
    int *edge_dist_offsets = nullptr, *edge_dist_positions = nullptr;
    double *target_deficits = nullptr, *base_deficits = nullptr;
    double *deficit_angles = nullptr, *gradients = nullptr;

    void alloc(const GpuMeshData &m) {
        int nh = m.n_hinges, ns = m.n_simplices, ne = m.n_edges;
        int nnz_hs = (int)m.hinge_simplex_ids.size();
        int nnz_eh = (int)m.edge_hinge_ids.size();
        int nnz_ed = (int)m.edge_dist_positions.size();
        int sq_size = (int)m.simplex_sq_dist_flat.size();

        CUDA_CHECK(cudaMalloc(&hinge_simplex_offsets, (nh+1)*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&hinge_simplex_ids, nnz_hs*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&hinge_opposite_a, nnz_hs*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&hinge_opposite_b, nnz_hs*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&simplex_sq_dist_offsets, (ns+1)*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&simplex_sq_dist_flat, sq_size*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&simplex_n_verts, ns*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&edge_hinge_offsets, (ne+1)*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&edge_hinge_ids, nnz_eh*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&edge_dist_offsets, (ne+1)*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&edge_dist_positions, nnz_ed*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&target_deficits, nh*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&base_deficits, nh*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&deficit_angles, nh*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&gradients, ne*sizeof(double)));
    }

    void upload(const GpuMeshData &m) {
        int nh = m.n_hinges, ns = m.n_simplices, ne = m.n_edges;
        #define UP(dst, src, n) CUDA_CHECK(cudaMemcpy(dst, src, (n), cudaMemcpyHostToDevice))
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
    if (nh == 0) { g.free(); return; }
    int bs = 256, gs = (nh + bs - 1) / bs;
    deficit_kernel<<<gs, bs>>>(
        g.hinge_simplex_offsets, g.hinge_simplex_ids,
        g.hinge_opposite_a, g.hinge_opposite_b,
        g.simplex_sq_dist_offsets, g.simplex_sq_dist_flat,
        g.simplex_n_verts, g.deficit_angles, nh);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_deficits, g.deficit_angles, nh*sizeof(double), cudaMemcpyDeviceToHost));
    g.free();
}

void compute_gradients_gpu(const GpuMeshData &mesh, double *h_gradients) {
    GpuArrays g;
    g.alloc(mesh);
    g.upload(mesh);

    int nh = mesh.n_hinges, ne = mesh.n_edges;
    if (nh == 0 || ne == 0) { g.free(); return; }
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

    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_gradients, g.gradients, ne*sizeof(double), cudaMemcpyDeviceToHost));
    g.free();
}


// =====================================================================
// Action gradient kernel: ∂S/∂ℓ²_e where S = Σ A_h·ε_h + S_matter
// =====================================================================

// Hinge area from a top-simplex's sq_dist matrix via Heron's formula.
// The hinge is the (d-2)-face opposite vertices a and b.
__device__ double hinge_area_dev(const double *sq_dist, int nv, int a, int b) {
    double sq[3];
    int idx = 0;
    for (int i = 0; i < nv && idx < 3; ++i) {
        if (i == a || i == b) continue;
        for (int j = i + 1; j < nv && idx < 3; ++j) {
            if (j == a || j == b) continue;
            sq[idx++] = fabs(sq_dist[i * nv + j]);
        }
    }
    if (idx < 3) return 0.0;
    double a2 = sq[0], b2 = sq[1], c2 = sq[2];
    double val = 2.0*(a2*b2 + b2*c2 + c2*a2) - (a2*a2 + b2*b2 + c2*c2);
    if (val <= 0.0) return 0.0;
    return sqrt(val) / 4.0;
}

// Compute A_h * ε_h for one hinge from (possibly perturbed) sq_dist_flat.
__device__ double compute_hinge_action(
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
    int s_end   = hinge_simplex_offsets[hid + 1];
    double angle_sum = 0.0;
    double area = 0.0;

    for (int si = s_start; si < s_end; ++si) {
        int sid = hinge_simplex_ids[si];
        int nv  = simplex_n_verts[sid];
        int off = simplex_sq_dist_offsets[sid];
        const double *sd = sq_dist_flat + off;

        double B[MAX_DIM * MAX_DIM];
        build_cm(sd, nv, B);
        angle_sum += dihedral_from_cm(B, nv + 1,
                                       hinge_opposite_a[si],
                                       hinge_opposite_b[si]);
        if (si == s_start)
            area = hinge_area_dev(sd, nv,
                                   hinge_opposite_a[si],
                                   hinge_opposite_b[si]);
    }
    return area * (2.0 * M_PI - angle_sum);
}

__global__ void action_gradient_kernel(
    const double *sq_dist_flat,
    const int *hinge_simplex_offsets,
    const int *hinge_simplex_ids,
    const int *hinge_opposite_a,
    const int *hinge_opposite_b,
    const int *simplex_sq_dist_offsets,
    const int *simplex_n_verts,
    const int *edge_hinge_offsets,
    const int *edge_hinge_ids,
    const int *edge_dist_offsets,
    const int *edge_dist_positions,
    const double *base_hinge_contribs,   // precomputed A_h·ε_h per hinge
    const int *worldline_mask,
    const double *worldline_edge_mass,  // per-edge mass (0 if not on worldline)
    double *gradients,
    int n_edges
) {
    int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= n_edges) return;

    int h_start = edge_hinge_offsets[eid];
    int h_end   = edge_hinge_offsets[eid + 1];
    int d_start = edge_dist_offsets[eid];
    int d_end   = edge_dist_offsets[eid + 1];

    double origSq = (d_start < d_end)
                    ? sq_dist_flat[edge_dist_positions[d_start]] : 1.0;
    double h = fmax(fabs(origSq) * 1e-4, 1e-8);

    // Base action contribution from affected hinges
    double S0_partial = 0.0;
    for (int hi = h_start; hi < h_end; ++hi)
        S0_partial += base_hinge_contribs[edge_hinge_ids[hi]];

    // Perturbed action contribution
    double Sp_partial = 0.0;
    for (int hi = h_start; hi < h_end; ++hi) {
        int hid = edge_hinge_ids[hi];
        int s_start_ = hinge_simplex_offsets[hid];
        int s_end_   = hinge_simplex_offsets[hid + 1];

        double angle_sum = 0.0;
        double area = 0.0;

        for (int si = s_start_; si < s_end_; ++si) {
            int sid = hinge_simplex_ids[si];
            int nv  = simplex_n_verts[sid];
            int sd_off = simplex_sq_dist_offsets[sid];

            double local_sq[MAX_DIM * MAX_DIM];
            for (int i = 0; i < nv * nv; ++i)
                local_sq[i] = sq_dist_flat[sd_off + i];
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
            if (si == s_start_)
                area = hinge_area_dev(local_sq, nv,
                                       hinge_opposite_a[si],
                                       hinge_opposite_b[si]);
        }
        Sp_partial += area * (2.0 * M_PI - angle_sum);
    }

    double delta_grav = Sp_partial - S0_partial;

    // Matter contribution: S_matter = -M √(W) for worldline edges
    // (W = |ℓ²| = Wick-rotated squared length, stored in sq_dist_flat)
    double delta_matter = 0.0;
    if (worldline_mask[eid] && origSq > 0.0) {
        double M = worldline_edge_mass[eid];
        delta_matter = -M * (sqrt(origSq + h) - sqrt(origSq));
    }

    gradients[eid] = (delta_grav + delta_matter) / h;
}

// =====================================================================
// Host-side: action gradient
// =====================================================================

void compute_action_gradient_gpu(const GpuMeshData &mesh,
                                 double *h_gradients) {
    GpuArrays g;
    g.alloc(mesh);
    g.upload(mesh);

    int nh = mesh.n_hinges, ne = mesh.n_edges;
    int bs = 256;

    // Upload extra arrays for the action gradient kernel
    double *d_base_contribs, *d_wl_mass;
    int *d_wl_mask;
    CUDA_CHECK(cudaMalloc(&d_base_contribs, nh * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_wl_mask, ne * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_wl_mass, ne * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_base_contribs, mesh.base_hinge_contribs.data(),
               nh * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wl_mask, mesh.worldline_edge_mask.data(),
               ne * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wl_mass, mesh.worldline_edge_mass.data(),
               ne * sizeof(double), cudaMemcpyHostToDevice));

    int gs_e = (ne + bs - 1) / bs;
    action_gradient_kernel<<<gs_e, bs>>>(
        g.simplex_sq_dist_flat,
        g.hinge_simplex_offsets, g.hinge_simplex_ids,
        g.hinge_opposite_a, g.hinge_opposite_b,
        g.simplex_sq_dist_offsets, g.simplex_n_verts,
        g.edge_hinge_offsets, g.edge_hinge_ids,
        g.edge_dist_offsets, g.edge_dist_positions,
        d_base_contribs, d_wl_mask, d_wl_mass,
        g.gradients, ne);

    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_gradients, g.gradients, ne * sizeof(double),
               cudaMemcpyDeviceToHost));

    cudaFree(d_base_contribs);
    cudaFree(d_wl_mask);
    cudaFree(d_wl_mass);
    g.free();
}

// =====================================================================
// Fused ∂F/∂W kernel: one thread per edge, all in one launch
// =====================================================================

// Compute A_h · ε_h for one hinge from sq_dist_flat with two additive
// perturbations applied.  Pass p_end <= p_start to skip a perturbation.
__device__ double hinge_action_2pert(
    int hid,
    const int *hinge_simplex_offsets,
    const int *hinge_simplex_ids,
    const int *hinge_opposite_a,
    const int *hinge_opposite_b,
    const int *simplex_sq_dist_offsets,
    const double *sq_dist_flat,
    const int *simplex_n_verts,
    const int *edge_dist_positions,
    int p1_start, int p1_end, double h1,
    int p2_start, int p2_end, double h2
) {
    int s_start = hinge_simplex_offsets[hid];
    int s_end   = hinge_simplex_offsets[hid + 1];
    double angle_sum = 0.0;
    double area = 0.0;

    for (int si = s_start; si < s_end; ++si) {
        int sid = hinge_simplex_ids[si];
        int nv  = simplex_n_verts[sid];
        int sd_off = simplex_sq_dist_offsets[sid];
        int va = hinge_opposite_a[si];
        int vb = hinge_opposite_b[si];

        double local_sq[MAX_DIM * MAX_DIM];
        for (int i = 0; i < nv * nv; ++i)
            local_sq[i] = sq_dist_flat[sd_off + i];

        for (int di = p1_start; di < p1_end; ++di) {
            int pos = edge_dist_positions[di];
            if (pos >= sd_off && pos < sd_off + nv * nv)
                local_sq[pos - sd_off] += h1;
        }
        for (int di = p2_start; di < p2_end; ++di) {
            int pos = edge_dist_positions[di];
            if (pos >= sd_off && pos < sd_off + nv * nv)
                local_sq[pos - sd_off] += h2;
        }

        double B[MAX_DIM * MAX_DIM];
        build_cm(local_sq, nv, B);
        angle_sum += dihedral_from_cm(B, nv + 1, va, vb);
        if (si == s_start)
            area = hinge_area_dev(local_sq, nv, va, vb);
    }
    return area * (2.0 * M_PI - angle_sum);
}

// For each edge j (one thread), compute ∂F/∂W_j where F = ||∇S||².
//
// When W_j is perturbed by h_j, only the action gradients of edges
// in nbr(j) change.  For each neighbor e, the change Δg(e) comes from
// hinges shared between e and j:
//
//   Δg(e) = Σ_{h shared} [(S_je - S_j0) - (S_0e - S_00)] / h_e
//
// where S_xy denotes A_h·ε_h with perturbation x on j and y on e.
//
//   dF[j] = {Σ_{e ∈ nbr(j)} [2·g0(e)·Δg(e) + Δg(e)²]} / h_j
//
__global__ void fused_F_gradient_kernel(
    const double *sq_dist_flat,
    const int *hinge_simplex_offsets,
    const int *hinge_simplex_ids,
    const int *hinge_opposite_a,
    const int *hinge_opposite_b,
    const int *simplex_sq_dist_offsets,
    const int *simplex_n_verts,
    const int *edge_hinge_offsets,
    const int *edge_hinge_ids,
    const int *edge_dist_offsets,
    const int *edge_dist_positions,
    const int *edge_nbr_offsets,
    const int *edge_nbr_ids,
    const double *base_action_grad,   // g0[n_edges] from first kernel
    const int *worldline_mask,
    const double *worldline_edge_mass,
    double *dF,
    int n_edges
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_edges) return;

    int dj_start = edge_dist_offsets[j];
    int dj_end   = edge_dist_offsets[j + 1];
    if (dj_start == dj_end) { dF[j] = 0.0; return; }

    double Wj = sq_dist_flat[edge_dist_positions[dj_start]];
    double hj = fmax(fabs(Wj) * 1e-4, 1e-8);

    int hj_start = edge_hinge_offsets[j];
    int hj_end   = edge_hinge_offsets[j + 1];

    double F_delta = 0.0;

    int nbr_start = edge_nbr_offsets[j];
    int nbr_end   = edge_nbr_offsets[j + 1];

    for (int ni = nbr_start; ni < nbr_end; ++ni) {
        int e = edge_nbr_ids[ni];

        int de_start = edge_dist_offsets[e];
        int de_end   = edge_dist_offsets[e + 1];
        if (de_start == de_end) continue;

        double We = sq_dist_flat[edge_dist_positions[de_start]];
        double he = fmax(fabs(We) * 1e-4, 1e-8);

        int he_start = edge_hinge_offsets[e];
        int he_end   = edge_hinge_offsets[e + 1];

        double delta_g = 0.0;

        // Sum over hinges of e that are shared with j
        for (int hi_e = he_start; hi_e < he_end; ++hi_e) {
            int hid = edge_hinge_ids[hi_e];

            // Linear search: is hid also a hinge of j?
            bool shared = false;
            for (int hi_j = hj_start; hi_j < hj_end; ++hi_j) {
                if (edge_hinge_ids[hi_j] == hid) { shared = true; break; }
            }
            if (!shared) continue;

            // Four evaluations of A_h · ε_h:
            //   S00 = base,  S0e = +e,  Sj0 = +j,  Sje = +j+e
            double S00 = hinge_action_2pert(
                hid, hinge_simplex_offsets, hinge_simplex_ids,
                hinge_opposite_a, hinge_opposite_b,
                simplex_sq_dist_offsets, sq_dist_flat, simplex_n_verts,
                edge_dist_positions,
                0, 0, 0.0,  0, 0, 0.0);

            double S0e = hinge_action_2pert(
                hid, hinge_simplex_offsets, hinge_simplex_ids,
                hinge_opposite_a, hinge_opposite_b,
                simplex_sq_dist_offsets, sq_dist_flat, simplex_n_verts,
                edge_dist_positions,
                de_start, de_end, he,  0, 0, 0.0);

            double Sj0 = hinge_action_2pert(
                hid, hinge_simplex_offsets, hinge_simplex_ids,
                hinge_opposite_a, hinge_opposite_b,
                simplex_sq_dist_offsets, sq_dist_flat, simplex_n_verts,
                edge_dist_positions,
                dj_start, dj_end, hj,  0, 0, 0.0);

            double Sje = hinge_action_2pert(
                hid, hinge_simplex_offsets, hinge_simplex_ids,
                hinge_opposite_a, hinge_opposite_b,
                simplex_sq_dist_offsets, sq_dist_flat, simplex_n_verts,
                edge_dist_positions,
                dj_start, dj_end, hj,  de_start, de_end, he);

            delta_g += ((Sje - Sj0) - (S0e - S00)) / he;
        }

        // Matter contribution to Δg: only when e = j and j is worldline
        if (e == j && worldline_mask[j] && Wj > 0.0) {
            double M = worldline_edge_mass[j];
            double g_base = -M * (sqrt(Wj + he) - sqrt(Wj)) / he;
            double g_pert = -M * (sqrt(Wj + hj + he) - sqrt(Wj + hj)) / he;
            delta_g += g_pert - g_base;
        }

        double g0e = base_action_grad[e];
        F_delta += 2.0 * g0e * delta_g + delta_g * delta_g;
    }

    dF[j] = F_delta / hj;
}

// =====================================================================
// Host-side: full step (base gradient + fused F gradient)
// =====================================================================

void compute_step_gpu(const GpuMeshData &mesh,
                      double *h_base_grad,
                      double *h_dF) {
    // Increase stack size for deeply nested device calls
    // (build_cm → dihedral_from_cm → cofactor_ij → det_small)
    cudaDeviceSetLimit(cudaLimitStackSize, 8192);

    GpuArrays g;
    g.alloc(mesh);
    g.upload(mesh);

    int nh = mesh.n_hinges, ne = mesh.n_edges;
    int bs = 256;
    int gs_e = (ne + bs - 1) / bs;

    // --- Extra arrays for action gradient kernel ---
    double *d_base_contribs, *d_wl_mass;
    int *d_wl_mask;
    CUDA_CHECK(cudaMalloc(&d_base_contribs, nh * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_wl_mask, ne * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_wl_mass, ne * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_base_contribs, mesh.base_hinge_contribs.data(),
               nh * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wl_mask, mesh.worldline_edge_mask.data(),
               ne * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wl_mass, mesh.worldline_edge_mass.data(),
               ne * sizeof(double), cudaMemcpyHostToDevice));

    // --- Step 1: base action gradient (1 kernel launch) ---
    action_gradient_kernel<<<gs_e, bs>>>(
        g.simplex_sq_dist_flat,
        g.hinge_simplex_offsets, g.hinge_simplex_ids,
        g.hinge_opposite_a, g.hinge_opposite_b,
        g.simplex_sq_dist_offsets, g.simplex_n_verts,
        g.edge_hinge_offsets, g.edge_hinge_ids,
        g.edge_dist_offsets, g.edge_dist_positions,
        d_base_contribs, d_wl_mask, d_wl_mass,
        g.gradients, ne);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_base_grad, g.gradients, ne * sizeof(double),
               cudaMemcpyDeviceToHost));

    // --- Step 2: fused F gradient (1 kernel launch) ---
    double *d_base_grad, *d_dF;
    int *d_nbr_offsets, *d_nbr_ids;
    int nnz_nbr = static_cast<int>(mesh.edge_nbr_ids.size());

    CUDA_CHECK(cudaMalloc(&d_base_grad, ne * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dF, ne * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_nbr_offsets, (ne + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nbr_ids, nnz_nbr * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_base_grad, h_base_grad,
               ne * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nbr_offsets, mesh.edge_nbr_offsets.data(),
               (ne + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nbr_ids, mesh.edge_nbr_ids.data(),
               nnz_nbr * sizeof(int), cudaMemcpyHostToDevice));

    fused_F_gradient_kernel<<<gs_e, bs>>>(
        g.simplex_sq_dist_flat,
        g.hinge_simplex_offsets, g.hinge_simplex_ids,
        g.hinge_opposite_a, g.hinge_opposite_b,
        g.simplex_sq_dist_offsets, g.simplex_n_verts,
        g.edge_hinge_offsets, g.edge_hinge_ids,
        g.edge_dist_offsets, g.edge_dist_positions,
        d_nbr_offsets, d_nbr_ids,
        d_base_grad, d_wl_mask, d_wl_mass,
        d_dF, ne);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_dF, d_dF, ne * sizeof(double), cudaMemcpyDeviceToHost));

    // --- Cleanup ---
    cudaFree(d_base_contribs);
    cudaFree(d_wl_mask);
    cudaFree(d_wl_mass);
    cudaFree(d_base_grad);
    cudaFree(d_dF);
    cudaFree(d_nbr_offsets);
    cudaFree(d_nbr_ids);
    g.free();
}

} // namespace cuda
} // namespace tessera
