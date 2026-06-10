---
myst:
  html_meta:
    description: |
      Whether the causal order defined by majorization on Schmidt
      spectra of a real-time-evolved lattice Schwinger-model state
      agrees with the Lieb-Robinson light cone and with a prescribed
      causet structure: hypothesis, falsification criteria,
      methodology, and recorded runs.
---

# Emergent Causal Order from Majorization

```{contents}
:local:
:depth: 2
```

This document is the complete statement of the emergent-causal-order
question: the hypothesis and its falsification criteria, the lattice
system the question is posed on, the numerical methodology that
answers it, and where the recorded runs live. The machinery that
executes it — MPO + DMRG ground states, the q-q̄ quench, TDVP
evolution, and the three-order comparison harness — is documented in
the [subsystem reference](spectral-dimension-schwinger-mps-dmrg.md).

## Abstract

We investigate whether the partial order induced by Nielsen majorization on
reduced-state Schmidt spectra of a real-time-evolved lattice gauge theory
agrees with two reference causal structures: the Lieb–Robinson cone of the
underlying spin Hamiltonian, and a prescribed causet (causal set) geometry on
which the lattice is embedded. The lattice Schwinger model serves as the test
system. The framework treats spacetime causal order as a derived quantity of
the entanglement structure of a many-body state on a discrete substrate, in
the spirit of the entanglement-builds-spacetime program {cite}`VanRaamsdonk2010Building, VanRaamsdonk2018BCbit`,
and operationalizes the comparison through tensor-network simulation
{cite}`Schollwock2011MPS, Banuls2013Schwinger`.

The work is exploratory. The hypothesis is falsifiable, but the
1+1-dimensional test bed is the simplest setting in which the question can be
posed numerically; agreement between the three orders in this regime would not
by itself establish a deeper principle.

## 1. Hypothesis

Let $\ket{\psi(t)}$ be the state of a quantum system on a discrete lattice
$\Lambda$ at time $t$. Fix a family $\mathcal{F} = \{A\}$ of subsystems
("cuts") closed under containment. For each $A \in \mathcal{F}$ let
$\lambda_A(t)$ denote the Schmidt spectrum of $\ket{\psi(t)}$ across the
bipartition $A \mid \bar A$, sorted nonincreasingly and zero-padded to a
common length.

Define the *majorization order* $\preceq_{\mathrm{maj}}$ on labels
$\{(A, t)\}$ by

```{math}
:label: maj-def
(A, s) \preceq_{\mathrm{maj}} (B, t)
\;\;\iff\;\;
\sum_{i=1}^{k} \lambda_{A,i}^\downarrow(s)
\;\le\;
\sum_{i=1}^{k} \lambda_{B,i}^\downarrow(t)
\quad \forall\, k \ge 1,
```

with the standard normalization $\sum_i \lambda_{A,i}(t) = 1$
{cite}`Nielsen1999LOCC, MarshallOlkinArnold2011`.

Let $\preceq_{\mathrm{LR}}$ denote the partial order induced by the
Lieb–Robinson light cone of the lattice Hamiltonian
{cite}`LiebRobinson1972, HastingsKoma2006`, and let $\preceq_{\mathrm{cs}}$
denote the partial order inherited from a causet $\mathcal{C}$ on which
$\Lambda$ is embedded {cite}`Ambjorn2004CDT, Regge1961`.

**Hypothesis (H).** For physical states arising from local-Hamiltonian
real-time evolution of a gauge-invariant ground state with a localized
charge-anticharge excitation, the three partial orders coincide on the
restricted label set obtained by taking $\mathcal{F}$ to consist of
contiguous spatial intervals at sampled times:

```{math}
:label: hypothesis
\preceq_{\mathrm{maj}}\big|_{\mathcal{F}\times T}
\;=\;
\preceq_{\mathrm{LR}}\big|_{\mathcal{F}\times T}
\;=\;
\preceq_{\mathrm{cs}}\big|_{\mathcal{F}\times T}
\quad
\text{up to an entanglement velocity } v_E \le v_{\mathrm{LR}}.
```

**Falsification criteria.**

1. *Strong falsification.* If $\preceq_{\mathrm{maj}}$ contains pairs whose
   spatial separation exceeds the Lieb–Robinson cone at the corresponding
   time difference, the hypothesis is wrong: majorization sees a
   superluminal-in-the-lattice-sense order.
2. *Weak falsification.* If $\preceq_{\mathrm{maj}}$ disagrees with
   $\preceq_{\mathrm{cs}}$ on a non-vanishing fraction of comparable pairs
   for non-trivial causet topologies, the prescribed causet geometry
   does not control the emergent order.
3. *Trivial agreement.* If $\preceq_{\mathrm{cs}}$ is forced to coincide with
   $\preceq_{\mathrm{LR}}$ by the regularity of the lattice (1+1D total
   order on time slices), agreement is uninformative; the test must be
   repeated on causets whose induced order is strictly coarser than the
   trivial chain.

## 2. Limitations and scope conditions

The following limitations bound the interpretive weight of the result.

- *Pure-state, contiguous-cut regime.* Nielsen's theorem is stated for pure
  bipartite LOCC convertibility {cite}`Nielsen1999LOCC`. Extensions to
  mixed states or non-contiguous cut families weaken or break the
  well-definedness of the $\preceq_{\mathrm{maj}}$ relation between
  *regions*. We restrict to pure states of the full chain and to
  contiguous intervals throughout.
- *Abelian gauge group, 1+1D.* The Schwinger model is U(1) with the gauge
  field eliminable in 1D {cite}`KogutSusskind1975, BanksKogutSusskind1976`.
  Generalization to SU(N) or to higher dimensions is not addressed and is
  likely to require qualitatively different methods.
- *Truncated electric basis.* The link Hilbert space is truncated at
  electric-field cutoff $\Lambda$; convergence in $\Lambda$ is a numerical
  question requiring separate convergence studies
  {cite}`Banuls2013Schwinger, ZoharCiracReznik2016`.
- *Trotter and bond-dimension errors.* TDVP integration introduces a finite
  Trotter error and bond-dimension truncation
  {cite}`Haegeman2016TDVP`. Both are controllable but must be
  characterized on each run before attaching meaning to a poset edge.
- *No claim of fundamentality.* The construction does not derive spacetime
  from entanglement in any constructive sense; it tests an internal
  consistency between three independently defined partial orders on a
  finite system.

## 3. System and notation

### 3.1 Lattice Schwinger model

Open boundary conditions, $N$ staggered sites with lattice spacing $a$, bare
fermion mass $m$, coupling $g$, and background field $L_0$. After a
Jordan–Wigner transformation and Gauss-law elimination of the gauge links
{cite}`KogutSusskind1975, BanksKogutSusskind1976, Banuls2013Schwinger`,
the Hamiltonian becomes a long-ranged spin Hamiltonian on $N$ qubits:

```{math}
:label: ham
H \;=\; H_{\mathrm{hop}} + H_m + H_E,
```

with

```{math}
:label: ham-parts
\begin{aligned}
H_{\mathrm{hop}} &= \tfrac{1}{4a}\sum_{n=1}^{N-1}\bigl(X_n X_{n+1} + Y_n Y_{n+1}\bigr), \\
H_m              &= \tfrac{m}{2}\sum_{n=1}^{N}(-1)^n Z_n, \\
H_E              &= \tfrac{g^2 a}{2}\sum_{n=1}^{N-1} L_n^2,
\quad
L_n = L_0 + \sum_{k=1}^{n}\!\left[\tfrac{1-Z_k}{2} - \tfrac{1-(-1)^k}{2}\right].
\end{aligned}
```

Expanding $L_n^2$ produces a Hamiltonian with $\mathcal{O}(N^2)$ long-range
$Z_m Z_{m'}$ terms; the resulting MPO bond dimension grows accordingly and is
handled with the AutoMPO machinery described below.

This is the same Hamiltonian implemented by `tessera.quantum`; see the
[subsystem reference](spectral-dimension-schwinger-mps-dmrg.md) for the
API surface and convention details, and
`include/quantum/SchwingerModel.hpp` for the algebraic expansion of
$L_n^2$ into operator and c-number pieces.

### 3.2 Cut family and Schmidt spectra

The cut family is fixed throughout:

```{math}
\mathcal{F} \;=\; \bigl\{\, [i,j] : 1 \le i \le j \le N \,\bigr\},
```

i.e. all contiguous spatial intervals. For each $A = [i,j]$ at time
$t$, $\lambda_A(t)$ is read from the bond singular values of the MPS in
left-canonical form at the appropriate cut, with no additional truncation
beyond the variational bond dimension.

### 3.3 Reference causal orders

$\preceq_{\mathrm{LR}}$ is constructed from the Lieb–Robinson velocity
$v_{\mathrm{LR}}$ extracted from out-of-time-ordered commutator
front-propagation in the same simulation. $\preceq_{\mathrm{cs}}$, when
present, is read directly from the causet adjacency provided by the host
package; on a regular 1+1D chain the causet order is trivial, so
$\preceq_{\mathrm{cs}}$ becomes informative only when the chain is
replaced with a non-trivial causet (the causet-embedded variant of
this pipeline).

## 4. Methodology

### 4.1 Numerical pipeline

The simulation is structured as a single C++ pipeline using ITensor v3 for
all tensor-network operations; Python serves only as a result viewer.

```{list-table}
:header-rows: 1
:widths: 18 22 35 25

* - Step
  - Object
  - Operation
  - Reference
* - 1
  - Schwinger MPO
  - AutoMPO assembly with $U(1)$ charge conservation
  - {cite}`Banuls2013Schwinger`
* - 2
  - Ground state $\ket{\Omega}$
  - Two-site DMRG sweeps to convergence
  - {cite}`White1992DMRG, Schollwock2011MPS`
* - 3
  - Initial quenched state
  - Local creation of $q\bar q$ pair at separation $d$ with Gauss-compatible electric string
  - {cite}`Buyens2014StringBreaking, Pichler2016RealTime`
* - 4
  - Time evolution
  - Two-site TDVP with adaptive bond dimension; one-site TDVP after saturation
  - {cite}`Haegeman2016TDVP`
* - 5
  - Per-step observables
  - $\langle L_n \rangle_t$, $\langle Z_n \rangle_t$, contiguous-cut Schmidt spectra
  - {cite}`Buyens2014StringBreaking`
* - 6
  - Majorization poset
  - Pairwise test on padded sorted spectra; transitive reduction to Hasse form
  - {cite}`Nielsen1999LOCC, MarshallOlkinArnold2011, Bhatia1997MatrixAnalysis`
* - 7
  - Lieb–Robinson order
  - $v_{\mathrm{LR}}$ from OTOC front; cone-induced partial order on labels
  - {cite}`LiebRobinson1972, HastingsKoma2006`
* - 8
  - Causet order (optional)
  - Inherited from tessera adjacency on a non-regular causet embedding
  - {cite}`Ambjorn2004CDT, Regge1961`
* - 9
  - Comparison
  - Kendall $\tau$, discordant-pair fraction, Hasse graph edit distance
  -
```

### 4.2 Initial state construction

The $q\bar q$ creation operator is

```{math}
:label: qqbar
\ket{q\bar q;\, i_0,d}
\;=\;
\mathcal{N}^{-1}\,
\sigma^+_{i_0}\,
\Bigl[\textstyle\prod_{n=i_0}^{i_0 + d - 1} U_n\Bigr]\,
\sigma^-_{i_0 + d}\,
\ket{\Omega},
```

where $U_n$ is the gauge link in the original lattice formulation and
$\mathcal{N}$ is the norm; in the Gauss-eliminated representation this
reduces to the appropriate shift in the cumulative electric-field variables
{cite}`Buyens2014StringBreaking`. Skipping the string yields a
Gauss-violating state and is a known failure mode.

### 4.3 Acceptance and validation

Two external benchmarks gate the analysis:

1. *Spectral benchmark.* Ground-state energy per site for $N=20$,
   $m/g \in \{0,\, 0.125,\, 0.25\}$ must agree with
   {cite}`Banuls2013Schwinger` to within $10^{-3}$.
2. *String benchmark.* In the heavy-quark regime $m/g \gg 1$ at
   $d=4$, $\langle L_n \rangle_t$ should exhibit a flat plateau of value
   $\pm 1$ between $i_0$ and $i_0 + d$ at intermediate times, consistent
   with {cite}`Buyens2014StringBreaking`. Energy conservation must hold
   to relative tolerance $10^{-3}$ over the full evolution.

Failure of either benchmark invalidates the upstream construction and
must be addressed before any conclusion is drawn from the poset
comparison.

### 4.4 Comparison statistics

For each pair of labels $((A,s),(B,t))$, the three orders return a value in
$\{\prec,\, \succ,\, =,\, \mathrm{incomparable}\}$. We report:

- *Kendall $\tau$* between the linear extensions of each pair of orders
  (per time slice and pooled);
- *Discordant fraction* — pairs comparable in one order and not the other,
  or oppositely ordered;
- *Hasse graph edit distance* between the transitive reductions, normalized
  by edge count.

Statistical uncertainty is estimated by bootstrap over Trotter step sequences
and over initial $i_0$ within a translation-invariant subregion.

### 4.5 Threats to validity

- *Spurious agreement from regularity.* On a translationally invariant
  lattice the three orders may coincide for reasons unrelated to the
  hypothesis. Re-running on non-trivial causet topologies is required
  to distinguish.
- *Truncation-induced order violations.* Aggressive bond-dimension
  truncation can perturb Schmidt spectra enough to flip a borderline
  majorization comparison. Comparisons within $\epsilon = 10^{-6}$ of the
  decision boundary are flagged and excluded.
- *Quantum-number sectoring.* All three orders are reported globally; a
  refinement that majorizes within fixed $U(1)$ charge sectors may be
  more physically meaningful and is deferred.

## 5. Deliverables and reproducibility

Each run produces (i) the converged ground-state energy and bond-dimension
profile, (ii) per-step electric-field and charge profiles, (iii) the full
time series of contiguous-cut Schmidt spectra, (iv) the three Hasse
diagrams, and (v) the comparison statistics with bootstrap intervals. Random
seeds, Trotter schedule, bond-dimension caps, and cutoff $\Lambda$ are
recorded with each output for reproducibility.

## 6. Recorded runs

The first hypothesis-test scan is written up in
[Lightcone vs. majorization](lightcone_vs_majorization.md).
Its headline: on the regular 1+1D chain the strong-falsification
criterion (§1, criterion 1) fires — at the physical
$v_{\mathrm{LR}} = 1$ roughly half of all majorization-related label
pairs lie outside the Lieb–Robinson cone, across every $(N, m/g, T)$
regime scanned — so the hypothesis as stated is rejected on the
regular chain. Because $\preceq_{\mathrm{cs}}$ is trivial there
(§1, criterion 3), the discriminating follow-up is the
causet-embedded re-run, where the within-slice structure of a
non-trivial causet either recovers the agreement or refutes the
hypothesis outright.

The scan scripts are `examples/quantum/lightcone_vs_majorization.py`
and its N-scaling and cone-overflow companions in the same directory.

```{bibliography}
:style: unsrt
```
