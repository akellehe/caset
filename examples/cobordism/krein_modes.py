# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Krein-mode analysis of the metric Hodge Laplacian — the shared core of the
annihilation model.

Under the SquaredContent (V²) weight convention with real signed edge intervals
the degree-k Hodge Laplacian `L_k` is a REAL matrix that is self-adjoint under
the INDEFINITE pairing `⟨a, b⟩_W = aᵀ W_k b` (`W_k L_k` is symmetric), so its
spectrum is closed under complex conjugation and every eigenvalue falls into
exactly one of two classes:

  * a REAL λ whose eigenvector has nonzero W-norm `vᵀWv` — a definite Krein
    signature ±1. The sign is the particle/antiparticle split: the same
    indefinite-norm structure as the Klein–Gordon inner product, where
    positive-norm modes are particles and negative-norm modes antiparticles.
  * a CONJUGATE PAIR (λ, λ̄) whose eigenvectors are W-NULL and conjugates of
    each other — a "broken" pair straddling the light cone in the W metric.

Pair FORMATION is an eigenvalue collision at an exceptional point and obeys
the Krein selection rule: only two real modes of OPPOSITE signature can merge
— the annihilation vertex (measured on a timelike-preconed host: the colliding
modes' W-norms +1.98e-8 / −1.98e-8 both → 0 approaching t* = 0.33722811, the
Re-sum exactly conserved through the event, and the same-signature control
pair passes through untouched; the event is LOCAL — the parents' |ψ|² supports
overlap at cos 0.988 and the newborn W-null pair inherits them). The reverse
crossing is pair creation.

This module packages the classification for the consumers that differ on top
of it: the animation's annihilation-heat panel and the q-q̄ Krein-signature
experiment.
"""
import numpy as np

import tessera

cob = tessera.cobordism


class KreinModes:
    """Eigen-decomposition of the degree-k metric Hodge Laplacian with the
    W-signature classification: real modes with their Krein signatures, and
    broken conjugate pairs with their per-cell localization.

    `cells` is the k-cell vertex-tuple list in `kSimplexVertices(k)` order —
    the index space of every eigenvector component and per-cell weight."""

    def __init__(self, st, degree, im_tol=1e-8):
        chain_complex = cob.ChainComplex.fromSpacetime(st)
        self.degree = degree
        self.cells = [tuple(c) for c in chain_complex.kSimplexVertices(degree)]
        self.n_cells = len(self.cells)
        if self.n_cells == 0:
            self.eigenvalues = np.array([], dtype=complex)
            self.eigenvectors = np.zeros((0, 0), dtype=complex)
            self.w_norms = np.array([])
            self.real_indices = []
            self.pair_indices = []
            return
        hodge = cob.HodgeLaplacian(st)
        laplacian = np.array(hodge.laplacian(degree),
                             dtype=complex).reshape(self.n_cells, self.n_cells)
        # The dichotomy needs the REAL W-pseudo-symmetric operator. Complex
        # entries mean either the Content (V) convention — timelike weights on
        # iℝ, where conjugation pairs the spectrum ACROSS the two ±i branch
        # operators rather than within this one — or genuinely complex edge
        # intervals. Refuse loudly instead of classifying garbage.
        imag_leak = float(np.abs(laplacian.imag).max())
        scale = float(np.abs(laplacian).max()) or 1.0
        if imag_leak > 1e-12 * scale:
            raise ValueError(
                f"L_{degree} is not real (max|Im| = {imag_leak:.3e}): the Krein "
                "classification requires the SquaredContent (V²) convention "
                "with real signed edge intervals")
        self.laplacian = laplacian.real
        self.weights = np.array(hodge.weights(degree),
                                dtype=complex).reshape(-1).real
        eigenvalues, eigenvectors = np.linalg.eig(self.laplacian)
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        # W-norm of each UNIT eigenvector: real modes carry vᵀWv ≠ 0 (the
        # signature), pair modes are W-null (≈ 0 to machine precision).
        squared_norms = (np.abs(eigenvectors) ** 2).sum(axis=0)
        self.w_norms = np.real(
            np.einsum("in,i,in->n", eigenvectors.conj(), self.weights,
                      eigenvectors)) / squared_norms
        imag_magnitudes = np.abs(eigenvalues.imag)
        self.real_indices = [i for i in range(self.n_cells)
                             if imag_magnitudes[i] <= im_tol]
        # One representative per broken pair: the Im > 0 member. The spectrum
        # is conjugation-closed, so this counts each pair exactly once, and
        # conjugate partners share |ψ|², so one member carries the pair's
        # localization.
        self.pair_indices = [i for i in range(self.n_cells)
                             if eigenvalues[i].imag > im_tol]

    @property
    def pair_count(self):
        """Number of broken conjugate pairs — the annihilation content."""
        return len(self.pair_indices)

    def pair_eigenvalues(self):
        """One eigenvalue per broken pair (the Im > 0 representative),
        ascending in real part."""
        return sorted((complex(self.eigenvalues[i]) for i in self.pair_indices),
                      key=lambda z: z.real)

    def signatures(self):
        """(eigenvalue, W-norm) for every real mode, ascending in eigenvalue.
        The W-norm's sign is the mode's Krein signature."""
        order = sorted(self.real_indices,
                       key=lambda i: self.eigenvalues[i].real)
        return [(float(self.eigenvalues[i].real), float(self.w_norms[i]))
                for i in order]

    def cell_weight(self, indices):
        """Per-cell |ψ|² summed over the given mode indices — each mode first
        normalized to unit total, then the sum renormalized to total 1 — the
        same weight the mode-localization panels paint. All-zero when the
        selection (or the complex) is empty."""
        weight = np.zeros(self.n_cells)
        for i in indices:
            amplitude = np.abs(self.eigenvectors[:, i]) ** 2
            total = float(amplitude.sum())
            if total > 0:
                weight += amplitude / total
        total = float(weight.sum())
        return weight / total if total > 0 else weight

    def pair_heat(self):
        """The annihilation heat: per-cell |ψ|² over one representative of
        every broken pair, normalized to total 1."""
        return self.cell_weight(self.pair_indices)
