# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Krein-mode analysis of the metric Hodge Laplacian — the shared core of the
annihilation model.

Under the SquaredContent (V²) weight convention with REAL signed edge intervals
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

THE LOCUS CAVEAT (measured): the classification exists only on the REAL-ℓ²
locus. The live stage-2 relaxation explores genuinely complex edge intervals
(max|Im ℓ²| ~ 0.3 within a dozen combined-run iterations on a precone-8 host),
and off the locus the structure dissolves entirely — the |Im λ| distribution
is gapless (0.15 → 108 on that host), the spectrum is nowhere near
conjugation-closed, and no W-null cluster survives. `KreinModes` therefore
reports `on_locus` / `reason` / the leak instead of classifying garbage:
consumers show WHERE the build stands relative to the locus, and the exact
pair structure whenever the state touches it (`W_k L_k` stays complex
SYMMETRIC for any complex weights, but symmetry alone constrains nothing —
the conjugation closure comes from realness).

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
    broken conjugate pairs with their per-cell localization — when the state
    is ON the real-ℓ² locus (`on_locus`); otherwise the leak diagnostics say
    how far off it stands and `reason` says why.

    `cells` is the k-cell vertex-tuple list in `kSimplexVertices(k)` order —
    the index space of every eigenvector component and per-cell weight."""

    def __init__(self, st, degree, im_tol=1e-8, locus_tol=1e-9, null_tol=1e-6):
        chain_complex = cob.ChainComplex.fromSpacetime(st)
        self.degree = degree
        self.cells = [tuple(c) for c in chain_complex.kSimplexVertices(degree)]
        self.n_cells = len(self.cells)
        self.eigenvalues = np.array([], dtype=complex)
        self.eigenvectors = np.zeros((0, 0), dtype=complex)
        self.w_norms = np.array([])
        self.real_indices = []
        self.pair_indices = []
        # De-rotated structure (#703), filled only OFF the locus: the dominant
        # spectral ray phase (mod pi), exact conjugate pairs about that ray,
        # and the forming quasi-null modes with the adaptive cut used.
        self.ray_phase = 0.0
        self.derotated_pair_partners = []
        self.derotated_pair_indices = []
        self.forming_indices = []
        self.forming_cut = None
        # How far the EDGE INTERVALS stand off the real locus (an absolute
        # max |Im ℓ²| — the state-space distance), and how far the OPERATOR
        # does (relative max |Im L| — what actually breaks the dichotomy).
        self.imag_interval_leak = 0.0
        edge_list = st.getEdgeList()
        for edge in (edge_list.toVector() if edge_list else []):
            self.imag_interval_leak = max(
                self.imag_interval_leak, abs((edge.getLength() ** 2).imag))
        if self.n_cells == 0:
            self.on_locus = True
            self.operator_imag_leak = 0.0
            self.reason = ""
            return
        hodge = cob.HodgeLaplacian(st)
        laplacian = np.array(hodge.laplacian(degree),
                             dtype=complex).reshape(self.n_cells, self.n_cells)
        scale = float(np.abs(laplacian).max()) or 1.0
        self.operator_imag_leak = float(np.abs(laplacian.imag).max()) / scale
        if self.operator_imag_leak > locus_tol:
            # Two distinct causes, distinguished by the interval leak: complex
            # edge intervals (the live stage-2 exploration), or the Content (V)
            # convention putting timelike weights on iℝ — there conjugation
            # pairs the spectrum ACROSS the two ±i branch operators, not
            # within this one.
            self.on_locus = False
            if self.imag_interval_leak <= locus_tol:
                self.reason = ("content-convention weights (L_k not real; "
                               "use SquaredContent)")
            else:
                self.reason = (f"complex edge intervals (max|Im ℓ²| = "
                               f"{self.imag_interval_leak:.3e})")
            # Off the real-l^2 locus the conjugate pairing dissolves, but
            # W_k L_k stays COMPLEX-SYMMETRIC for any complex weights, so the
            # BILINEAR self-norm eta_i = v_i^T W v_i (NO conjugation) is
            # defined everywhere: |eta| -> 0 is the coordinate-free
            # annihilation-proximity signal (an exceptional-point mode is
            # bilinearly self-orthogonal identically), and on the locus eta
            # reduces to the real +/- Krein norm. Pairing-dependent notions
            # (pair_count, signatures) deliberately stay locus-only (#694).
            self.laplacian = laplacian
            self.weights = np.array(hodge.weights(degree),
                                    dtype=complex).reshape(-1)
            eigenvalues, eigenvectors = np.linalg.eig(self.laplacian)
            self.eigenvalues = eigenvalues
            self.eigenvectors = eigenvectors
            self._classify_bilinear(null_tol)
            self.real_indices = []
            self.pair_indices = []
            squared_norms = (np.abs(eigenvectors) ** 2).sum(axis=0)
            self.w_norms = np.real(self.bilinear_norms) / np.maximum(
                squared_norms, 1e-300)
            self._derotated_classification(im_tol)
            return
        self.on_locus = True
        self.reason = ""
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
        # is conjugation-closed on the locus, so this counts each pair exactly
        # once, and conjugate partners share |ψ|², so one member carries the
        # pair's localization.
        self.pair_indices = [i for i in range(self.n_cells)
                             if eigenvalues[i].imag > im_tol]
        self._classify_bilinear(null_tol)

    def _classify_bilinear(self, null_tol):
        """The locus-independent classification (#694): per-mode BILINEAR
        self-norm eta_i = v_i^T W v_i and its normalized magnitude, the
        quasi-null index q_i = |eta_i| / (|v_i|^T |W| |v_i|) in [0, ~1].
        q -> 0 marks annihilation content: on the locus both members of every
        broken conjugate pair, off it the exceptional-point-adjacent modes."""
        v, w = self.eigenvectors, np.asarray(self.weights, dtype=complex)
        eta = np.einsum("in,i,in->n", v, w, v)
        scale = np.einsum("in,i,in->n", np.abs(v), np.abs(w), np.abs(v))
        self.bilinear_norms = eta
        with np.errstate(invalid="ignore", divide="ignore"):
            self.quasi_null_index = np.where(scale > 0,
                                             np.abs(eta) / scale, 0.0)
        self.null_indices = [i for i in range(self.n_cells)
                             if self.quasi_null_index[i] <= null_tol]

    def _derotated_classification(self, im_tol):
        """Dominant-ray de-rotation (#703). Balanced-edge geometry makes every
        edge interval purely imaginary, so the degree-k weights carry one
        phase each and `L_k` is a phase-rotated NEAR-REAL operator: the
        conjugate structure the on-locus branch reads about the real axis
        lives about the ray `e^{i phi} R` instead (measured phi = -pi/2 under
        V^2, -pi/4 under V). Measure phi as the doubled-angle mean (phase is
        only defined mod pi), rotate the spectrum back, and pair modes exactly
        as the on-locus branch would: genuinely off-ray, with the conjugate
        partner present (1e-6 relative — the rotated realness is approximate,
        so closure is a match, not an identity). At phi = 0 on a real
        spectrum this reduces to the on-locus pairing. FORMING modes are the
        quasi-null tail: bilinear self-norm below a quarter of the median —
        the coordinate-free "meaningfully departing toward W-nullness" band
        (exact nulls, q <= null_tol, are a subset by construction)."""
        lam = self.eigenvalues
        mags = np.abs(lam)
        scale = float(mags.max()) if mags.size else 0.0
        if scale <= 0:
            return
        nonzero = mags > 1e-12 * scale
        if nonzero.any():
            doubled_mean = np.mean(np.exp(2j * np.angle(lam[nonzero])))
            if np.abs(doubled_mean) > 0:
                self.ray_phase = float(0.5 * np.angle(doubled_mean))
        rotated = lam * np.exp(-1j * self.ray_phase)
        off_ray = rotated.imag
        unmatched = set(range(self.n_cells))
        pairs = []
        for i in np.argsort(-np.abs(off_ray)):
            i = int(i)
            if i not in unmatched or off_ray[i] <= im_tol * scale:
                continue
            candidates = [j for j in unmatched
                          if j != i and off_ray[j] < -im_tol * scale]
            if not candidates:
                continue
            distances = np.abs(rotated[candidates] - np.conj(rotated[i]))
            best = int(np.argmin(distances))
            if distances[best] <= 1e-6 * max(abs(rotated[i]), 1e-30):
                partner = candidates[best]
                pairs.append((i, partner))
                unmatched.discard(i)
                unmatched.discard(partner)
        self.derotated_pair_partners = pairs
        self.derotated_pair_indices = [i for i, _ in pairs]
        q = self.quasi_null_index
        if q is not None and q.size:
            median = float(np.median(q))
            if median > 0:
                self.forming_cut = median / 4.0
                self.forming_indices = [
                    i for i in range(self.n_cells) if q[i] <= self.forming_cut]

    @property
    def null_mode_count(self):
        """Number of quasi-null (bilinearly self-orthogonal) modes — defined
        EVERYWHERE. On the locus this is exactly 2 x pair_count (both members
        of every broken pair are W-null), so consumers displaying
        null_mode_count / 2 get a trace continuous across locus crossings."""
        return len(self.null_indices)

    def null_heat(self):
        """Per-cell |psi|^2 over every quasi-null mode, normalized to total 1.
        Defined everywhere; on the locus it matches pair_heat (conjugate
        partners share |psi|^2)."""
        return self.cell_weight(self.null_indices)

    @property
    def pair_count(self):
        """Number of broken conjugate pairs — the annihilation content.
        None off the locus, where the structure does not exist."""
        return len(self.pair_indices) if self.on_locus else None

    def pair_eigenvalues(self):
        """One eigenvalue per broken pair (the Im > 0 representative),
        ascending in real part; empty off the locus."""
        return sorted((complex(self.eigenvalues[i]) for i in self.pair_indices),
                      key=lambda z: z.real)

    def signatures(self):
        """(eigenvalue, W-norm) for every real mode, ascending in eigenvalue;
        empty off the locus. The W-norm's sign is the mode's Krein
        signature."""
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
        every broken pair, normalized to total 1. EMPTY (length 0) off the
        locus — deliberately distinct from the all-zero "on the locus with no
        broken pairs", so a consumer can tell "none" from "undefined"."""
        if not self.on_locus:
            return np.array([])
        return self.cell_weight(self.pair_indices)
