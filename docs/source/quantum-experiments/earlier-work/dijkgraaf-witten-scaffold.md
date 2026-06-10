# The Dijkgraaf–Witten scaffold: the quantized shadow of the spectral Z

The $\mathbb{Z}_2$ Dijkgraaf–Witten state sum was the topological layer the
State–Operation–Cobordism programme was built on: it certified that the bulk
$Z$ is a genuine cobordism functor, carried the sign invariant, calibrated the
spectral reading against an independent metric-free oracle, and supplied the
first (pinned) realizable-gate image. With H3 now validated on the spectral
data alone — the value equation
$Z_{\text{spec}}=\langle\psi_A|U|\psi_B\rangle$ holds at machine precision on
every realized gate, with triangulation invariance and the obstruction
certificate read off the same spectrum (see the
[results page](../state-operation-cobordism/cobordism-results.md)) — the DW
layer carries no live evidence. This page is its record: the functoriality
checks, the sign invariant, the bridge, the pinned image, and the
$b_1$-hole retest that together motivated the continuous spectral method.

The through-line, established by the bridge below and confirmed by the staged
synthesis: **the DW invariant is the quantized shadow of the continuous
spectral $Z$** — integer-valued on the flat-connection lattice, a strict
subset of what the spectrum carries.

## The state sum is a triangulation-invariant TQFT (T1, T2, T5)

On the topological layer ($n=3$, simplicial), the $\mathbb{Z}_2$
Dijkgraaf–Witten state sum $Z(W)$ is a well-defined cobordism functor
(`topological_correspondence.py`):

- **T1 — cylinder $=$ identity.** For $W=\Sigma\times[0,T]$, $\texttt{map}(T^2\times I)$
  is exactly $\mathrm{id}_4$ and $\langle\psi_A|Z(W)|\psi_B\rangle=\langle\psi_A|\psi_B\rangle$
  for harmonic-1-form boundary states $\psi\in\ker L_1(T^2)$; residual
  $6.28\times10^{-16}$.
- **T2 — triangulation independence (make-or-break).** $Z(S^2\times S^1)$ drifts by
  $0.00$ over 18 interior Pachner moves (including $1\!\to\!4$ moves that change
  $\lvert V\rvert$), for both cocycle classes — $Z$ is invariant to machine
  precision with $\partial W$ fixed.
- **T5 — composition / functoriality.** $\texttt{map}(\texttt{glue}(W_1,W_2))=
  \texttt{map}(W_2)\,\texttt{map}(W_1)$ (the matrix product), with
  $\operatorname{Tr}(\texttt{map})=Z_{\text{closed}}$ on the self-glue.

## The sign carries an invariant (T3), and the holonomy is consistent (T4)

The nontrivial cocycle is a genuine topological invariant, not a phase artefact:
$Z_{\text{sign}}(\mathbb{RP}^3)=0\neq1=Z_{\text{triv}}(\mathbb{RP}^3)$ (T3,
`topological_correspondence.py`). The twist is the mod-2 cup-cube
$(-1)^{\langle g^3,[W]\rangle}$, so the positive control needs a 1-class with
$g^3\neq0$: $\mathbb{RP}^3$ ($H^*=\mathbb{Z}_2[t]/t^4$, $t^3\neq0$), whose
2-torsion $H_1=\mathbb{Z}_2$ is exactly what separates it from the **negative
controls** $S^2\times S^1$ and $T^3$ (free $H_1$, $g^3=0$). The untwisted
normalization is $Z_{\text{triv}}(W)=2^{\,b_1(\mathbb{Z}_2)-1}$. Cross-layer
consistency holds too (T4, residual $0.00$): the bulk $\mathbb{Z}_2$ holonomy
equals the cycle flux $\Phi_\gamma$ restricted to $\{0,\pi\}$. (The brute-force
flat-connection enumeration is $2^{\dim Z^1}$, capped at $\dim Z^1\le24$, so the
state sum runs on small manifolds — $S^3$, $S^2\times S^1$, $\mathbb{RP}^3$ — and
$T^3$, $\dim Z^1=29$, is used only for the combinatorial T2 checks.)

## The DW–spectral bridge: three independent readings agree (and only on the lattice)

On a shared boundary surface $\Sigma=T^2$ (with $b_1=2$, $Z(\Sigma)=\mathbb{C}^4$,
spectral qubit $\ker L_1(T^2)=\mathbb{C}^2$) the value of the cobordism at a pair
of prepared boundary states is computed three *independent* ways — the
topological Dijkgraaf–Witten state sum $Z_{DW}$ (metric-free: topology and cocycle
only), the operation's transition amplitude $\langle\psi_A|U|\psi_B\rangle$, and
the spectral Hodge harmonic $Z_{\text{spec}}$ — testing
$Z_{DW}(W_{AB})=\langle\psi_A|U|\psi_B\rangle=Z_{\text{spec}}$
(`dw_spectral_bridge.py`). On longitude-aligned states all three agree to
$\sim10^{-15}$:

| $(a,b)$ | $Z_{DW}$ (topo) | $\langle\psi_A|\,\mathrm{id}\,|\psi_B\rangle$ (op) | $Z_{\text{spec}}$ (Hodge) | max $\Delta$ |
|---|---|---|---|---|
| $1,\,1$ | $+1.000000+0.000000i$ | $+1.000000+0.000000i$ | $+1.000000+0.000000i$ | $1.1\times10^{-15}$ |
| $e^{i\pi/3},\,0.6{+}0.8i$ | $+0.992820-0.119615i$ | $+0.992820-0.119615i$ | $+0.992820-0.119615i$ | $7.8\times10^{-16}$ |
| $0.5{-}0.5i,\,i$ | $-0.500000+0.500000i$ | $-0.500000+0.500000i$ | $-0.500000+0.500000i$ | $7.9\times10^{-16}$ |
| $2,\,0.25{+}0.97i$ | $+0.500000+1.940000i$ | $+0.500000+1.940000i$ | $+0.500000+1.940000i$ | $1.4\times10^{-15}$ |

The spectral side is a certified harmonic: the longitude is realized on the solid
torus at $r=9.65\times10^{-29}$, $\lambda=-4.25\times10^{-17}$ with witness
boundary-block overlap $1.000000$, while the meridian — which bounds a disk and
dies in $H_1(W)$ — floors at $15.2567$ (matching an independent numpy Hodge oracle
to $\Delta=5.3\times10^{-15}$). Crucially, the bridge holds **on the
DW-representable subset and only there**. With the cobordism (hence $Z_{DW}$) held
fixed, a generic $U$ departs from $Z_{DW}$:

| operation $U$ | representable? | $\langle\psi_A|U|\psi_B\rangle$ | $\Delta=\lvert\text{amp}-Z_{DW}\rvert$ |
|---|---|---|---|
| $\mathrm{id}_4\;(=Z(T^2\times[0,T]))$ | **yes** | $+0.992820-0.119615i$ | $0.0$ |
| Hadamard on the qubit block | no | $+0.521099-0.062782i$ | $4.75\times10^{-1}$ |
| Haar-random $U(4)$ (seed 7) | no | $-0.291344+0.172837i$ | $1.32\times10^{0}$ |
| Haar-random $U(4)$ (seed 23) | no | $-0.553146+0.040816i$ | $1.55\times10^{0}$ |

The $\mathbb{Z}_2$ DW maps are integer-quantized in the flat-connection basis
($\mathrm{id}_4$ for both cocycles, solid-torus cap $[1,0,1,0]$); a generic complex
amplitude cannot be hit. Over 200 Haar $U(4)$ the gap has minimum $0.218$ and
median $1.03$ — a generic $U$ is uniformly bounded away from the DW image, not a
measure-zero accident. **The DW invariant is the quantized shadow of the
continuous spectral $Z$:** the spectral oracle realizes the longitude *and a
neighborhood of it*, so $\{Z(W)\}_{\mathbb{Z}_2\text{-DW}}\subsetneq\{\text{spectrally
realizable}\}$, and the two coincide exactly where the operation lands on the
lattice.

## The pinned operation image: $S_3$

When is an operation $U$ on $Z(T^2)=\mathbb{C}[H^1(T^2;\mathbb{Z}_2)]=\mathbb{C}^4$
DW-realizable on a **pinned** cobordism — a twisted cylinder of fixed topology?
Exactly when it is one of the **six holonomy permutations** — the permutation
representation of a $GL(2,\mathbb{Z}_2)=SL(2,\mathbb{Z}_2)$ automorphism of
$H^1(T^2;\mathbb{Z}_2)$, fixing the trivial class (index 0) and permuting the three
non-trivial classes $\{[a],[b],[a{+}b]\}=\{1,2,3\}$. Two twisted cylinders generate
the group (`realizable_image_sweep.py`): the coordinate **swap** $(1\,2)$ on the
9-vertex product torus (its mod-2 reduction is the modular $S$) and the order-3
**multiplier** $(1\,2\,3)$ on the 7-vertex Möbius torus. Every permutation of the
three non-zero vectors of $(\mathbb{Z}_2)^2$ is automatically $GF(2)$-linear, so the
pinned realizable image is the *full* $S_3$.

The content of $S_3$ is **holonomy-class permutation without superposition** — and
that is *not* computational separability. Several computational **entanglers** are
holonomy permutations and so realize: `CNOT`, reversed-`CNOT`, `SWAP`, and the
double-`CNOT` each permute the holonomy classes (each fixes $[0]$), so each is
DW-realizable despite entangling in the computational basis (operator-Schmidt
rank $2$–$4$). What floors is **holonomy superposition** — a gate that sends a
holonomy class to a *superposition* of classes (`H`-type), or that lands off the
$0/1$ permutation lattice with a phase or sign (`CZ`, `T`, `S`, `iSWAP`, `CPHASE`,
$\sqrt{\mathrm{SWAP}}$). Realizability under the pinned construction is a property
of the holonomy action, indifferent to a gate's name:

| gate | pinned-DW realizable? | reason (action on the holonomy classes) |
|---|---|---|
| Identity | **yes** | fixes every class (the trivial permutation) |
| `SWAP` $=(1\,2)$ | **yes** | exchanges $[a]\leftrightarrow[b]$, fixes $[0]$ |
| `CNOT` $=(2\,3)$ | **yes** | transposes $[b]\leftrightarrow[a{+}b]$, fixes $[0]$ |
| reversed-`CNOT` $=(1\,3)$ | **yes** | transposes $[a]\leftrightarrow[a{+}b]$, fixes $[0]$ |
| 3-cycles $(1\,2\,3),(1\,3\,2)$ | **yes** | cycle the three non-trivial classes, fix $[0]$ |
| Hadamard ($H{\otimes}H$) | no | mixing — off the permutation lattice ($\texttt{gap}=2.00$) |
| `T`, `S` | no | diagonal phase, not a $0/1$ permutation |
| `CZ` $=\mathrm{diag}(1,1,1,-1)$ | no | diagonal sign, not a permutation |
| `iSWAP` | no | permutes with $i$ phases (complex permutation), not $0/1$ |
| `CPHASE` | no | diagonal phase, not a permutation |
| Pauli-`X` | no | moves the trivial class ($0\to3$) |

The separator is clean. Of $S_4$'s 24 permutation matrices, exactly the **6**
fixing index 0 are realizable; the **18** that move the trivial class are not.
Across the full labeled sweep, the predicate `is_holonomy_perm` (fixes $[0]$ and
permutes $\{1,2,3\}$) is the *only* property that separates the pinned image with no
errors — **19 TP / 0 FP / 0 FN / 210 TN over 229 operations**. Every weaker
predicate leaks (*is a permutation* admits the 18 that move index 0; *fixes the
trivial class* admits diagonal sign matrices; *near-$I_4$* catches only the seven
identity copies). The realizable rate by family makes the same point:

| family | realizable / total |
|---|---|
| `realizable_s3` | 6 / 6 |
| permutation (the 24 of $S_4$) | 6 / 24 |
| gate | 3 / 23 |
| interp | 3 / 33 |
| diagonal | 1 / 48 |
| signed_perm | 0 / 20 |
| mixing (Hadamard / DFT / Haar / GL) | 0 / 75 |

The staged spectral synthesis later showed this image to be an artefact of the
**bit-exact boundary pinning**, not of the physics: synthesizing the boundary
lifts the integrality constraint, and the realizable set becomes the
charge-conservation criterion — a continuous group containing $S_3$ — with the
value equation holding spectrally on all of it (see the
[results page](../state-operation-cobordism/cobordism-results.md)).

## What the $b_1$ hole adds: superposed states, not new gates

The pinned image above fixes the bulk topology in advance; it need not. The
emergent-bulk work loosened it — handing the realizability search a
**topology-changing surgery move** (a boundary-fixed interior-cell *remove*) so the
first Betti number $b_1$ becomes an **output** rather than an input — on the
conjecture that the resulting $b_1=1$ hole, which carries a superposed boundary
state, would bring the floored superposition/entangling gates into the realizable
set. `loosened_gate_retest.py` tests that conjecture directly, two ways, and reports
the residuals — not a hoped-for answer.

**As operations, the hole adds nothing — every gate floors.** Bending each gate to
its Choi state $\operatorname{vec}(U)$ (length $d_Ad_B=16$) and handing it to the
surgery oracle with $b_1$ free — `decide(vec(U),4,4,growth_mode=SURGERY,
harmonic=True)` on a filled-disk bulk ($b_1=0$, 19 vertices, the rim the pinned
output support) whose interior core triangle the search may remove — **every gate
floors** at $r\approx0.38$–$0.41$: the six $S_3$ controls (the identity included,
$r=0.40$) and every superposition/entangling gate alike, none below the
$\texttt{REALIZE}=10^{-3}$ line.

| gate | family | pinned DW ($\texttt{gap\_to\_S3}$) | superposition? | entangling? | loosened operation ($r$, $b_1$) |
|---|---|---|---|---|---|
| Identity | $S_3$ control | **realizable** ($0.00$) | no | no | floor ($0.40$, $b_1=0$) |
| `SWAP` | $S_3$ control | **realizable** ($0.00$) | no | yes | floor ($0.39$, $b_1:0\to1$) |
| `CNOT` | $S_3$ control | **realizable** ($0.00$) | no | yes | floor ($0.39$, $b_1=0$) |
| reversed-`CNOT` | $S_3$ control | **realizable** ($0.00$) | no | yes | floor ($0.39$, $b_1:0\to1$) |
| 3-cycles | $S_3$ control | **realizable** ($0.00$) | no | yes | floor ($0.38$, $b_1:0\to1$) |
| `DCNOT` | $S_3$ control | **realizable** ($0.00$) | no | yes | floor ($0.39$, $b_1:0\to1$) |
| $H{\otimes}I$ | superposition | floor ($2.27$) | yes | no | floor ($0.39$, $b_1:0\to1$) |
| $I{\otimes}H$ | superposition | floor ($2.27$) | yes | no | floor ($0.40$, $b_1=0$) |
| $H{\otimes}H$ | superposition | floor ($2.00$) | yes | no | floor ($0.38$, $b_1=0$) |
| $\sqrt{\mathrm{SWAP}}$ | superposition | floor ($1.41$) | yes | yes | floor ($0.38$, $b_1:0\to1$) |
| $\sqrt{\mathrm{iSWAP}}$ | superposition | floor ($1.08$) | yes | yes | floor ($0.39$, $b_1:0\to1$) |
| `CZ` | phase/entangler | floor ($2.00$) | no | yes | floor ($0.41$, $b_1:0\to1$) |
| `CPHASE`$(\pi/4)$ | phase/entangler | floor ($0.77$) | no | yes | floor ($0.40$, $b_1=0$) |
| $T{\otimes}I$ | phase | floor ($1.08$) | no | no | floor ($0.40$, $b_1=0$) |
| $S{\otimes}I$ | phase | floor ($2.00$) | no | no | floor ($0.40$, $b_1:0\to1$) |
| `iSWAP` | phase/entangler | floor ($2.00$) | no | yes | floor ($0.39$, $b_1:0\to1$) |
| $X{\otimes}X$ | Pauli perm | floor ($2.00$) | no | no | floor ($0.39$, $b_1:0\to1$) |
| $Z{\otimes}Z$ | diagonal sign | floor ($2.00$) | no | no | floor ($0.40$, $b_1:0\to1$) |

The surgery search opens the handle ($b_1:0\to1$) for many gates, yet the residual
floors regardless: **$b_1$ development is decoupled from realizability.** The reason
is structural — at the Choi-vec degree $k=0$ the harmonic kernel $\ker L_0$ is the
locally-constant functions (dimension $b_0=1$ on a connected bulk), so opening a
$b_1$ handle cannot enlarge it; the hole is spectrally inert for an operation
target. (The eigenvalue-agnostic mode is under-constrained and is *not* a
topological realization: `harmonic=False` drives $H{\otimes}H$ to $r=9.9\times10^{-4}$
but at $\lambda=2.93$ — a non-harmonic eigenvector, accepted only because that
criterion asks for *any* eigenvector, not a $\ker L$ one.)

**As states, the hole adds the superposition — that is exactly what it carries.** At
the Hodge-qubit degree $k=1$ the same surgery move makes $b_1$ an output for a
boundary *1-cycle*. The superposed meridian carried on both boundary circles — the
state an `H`-type gate would create — **floors on the disk** ($b_1=0$,
$r=4.5\times10^{-1}$) and **realizes on the annulus** ($b_1=1$, $r=6.9\times10^{-8}$,
$\lambda\to0$); from the disk seed the surgery search opens the handle on its own
($b_1:0\to1$, scored purely by the harmonic residual, $\partial W$ held bit-exact)
and the meridian then realizes ($r\sim10^{-5}$), while the sign-flipped (non-closed)
conjugation floors on every filling ($r\approx0.22$) — the period-matching
obstruction is genuinely separate from the topological one
(`emergent_bulk_realizability.py`). Removal is the load-bearing move — boundary-fixed
*additive* growth is topology-preserving at $k\ge1$ — so the realizable *state* set
is exactly $\mathrm{image}\big(H_1(\partial W)\to H_1(W)\big)$ for the $W$ the search
grows.

**The verdict.** $S_3$ is and remains the realizable *operation* image of the pinned
DW construction. Loosening the topology so $b_1$ emerges does **not** enlarge it: no
superposition or entangling gate realizes as an operation, hole open or closed. What
the hole enlarges is the realizable *state* space — a superposed/entangled boundary
cycle, unrealizable on the $b_1=0$ filling, realizes once surgery opens $b_1=1$. The
hole represents superposition, as the conjecture held — but it does so for **states**
(homology classes on $\partial W$), not for gate operations. The quantized-shadow
reading of the bridge says the same from the value side: the $\mathbb{Z}_2$-DW
operation image is a discrete lattice that a generic superposition/entangling $U$ is
uniformly bounded away from (gap median $1.03$ over 200 Haar $U(4)$), and the
continuous freedom the hole supplies lives in the *state* — the spectral $Z$ — not
in new realizable maps. The construction that finally carried the superposition
*gates* themselves is the staged spectral synthesis with the boundary synthesized,
whose realizable set is the charge-conservation criterion and whose value equation
holds spectrally — the
[results page](../state-operation-cobordism/cobordism-results.md) reports it.

## Figures

A per-output force-directed render of the simplicial complexes these experiments
build (`force_layout_3d` / `layout_from_spacetime`, vertices colored by
amplitude magnitude where a state is carried).

![A pinned-realizable gate's cobordism W: the SWAP twisted cylinder](https://github.com/akellehe/tessera/releases/download/issue-attachments/cobordism_results_gate_W.png)
*Figure 1 — A pinned-realizable gate as a cobordism.* The `SWAP` gate's bulk
$W=\texttt{twistedCylinder}(T^2,\,\text{swap})$ — the $(1\,2)$ holonomy
permutation. `SWAP` realizes under the pinned construction because it fixes the
trivial class and exchanges $[a]\leftrightarrow[b]$.

![The disk filling, b1=0](https://github.com/akellehe/tessera/releases/download/issue-attachments/cobordism_results_disk.png)
*Figure 2 — The disk filling ($b_1=0$).* The octahedron minus one face: the
boundary meridian bounds the still-filled antipodal cell, so it floors.

![The annulus filling, b1=1](https://github.com/akellehe/tessera/releases/download/issue-attachments/cobordism_results_annulus.png)
*Figure 3 — The annulus filling ($b_1=1$).* The octahedron minus two antipodal
faces: the meridian survives in $H_1$, so the cobordism carries it as a bulk
harmonic.

![The surgery-grown bulk](https://github.com/akellehe/tessera/releases/download/issue-attachments/cobordism_results_surgery.png)
*Figure 4 — The surgery-grown bulk.* From the disk seed, the boundary-fixed remove
opens the handle ($b_1:0\to1$) on its own under residual minimization, and the
floored meridian realizes.

## Method, conventions, and reproduction

These experiments are pure orchestration of separately-tested classes —
`DijkgraafWitten` (the $\mathbb{Z}_2$ state sum + `map` / `amplitude`),
`Cobordism.twistedCylinder` (the pinned maps), `BoundaryStateSpace` (the
$\ker L_1(\Sigma)\to Z(\Sigma)$ preparation), `ChoiJamiolkowski` (the bend), and
`EigenstateSynthesis` / `RealizabilityOracle` (the surgery oracle of the loosened
retest). DW-specific conventions:

- The DW reading is **metric-free** (topology + cocycle only); the Frobenius norm
  $\lVert\cdot\rVert$ scores $\texttt{gap\_to\_S3}=\min_{g\in S_3}\lVert U-g\rVert$,
  with $\texttt{realizable}:=\texttt{gap\_to\_S3}<10^{-7}$.
- The brute-force flat-connection enumeration is $2^{\dim Z^1}$, capped at
  $\dim Z^1\le24$.

To reproduce:

```
pip install -e ".[dev]" -C cmake.define.TESSERA_CUDA=OFF
python examples/cobordism/topological_correspondence.py   # T1–T5 (the state-sum functor) + the §5.6 Lorentzian variant
python examples/cobordism/dw_spectral_bridge.py           # the DW–spectral bridge
python examples/cobordism/realizable_image_sweep.py       # the pinned operation image (S₃)
python examples/cobordism/emergent_bulk_realizability.py  # b₁ as an output (states)
python examples/cobordism/loosened_gate_retest.py         # the loosened gate re-test (k=0)
python -m pytest tests/cobordism
```

Raw tables and figures are written to `/tmp/cobordism/` and are **not
committed**; the figures above are uploaded to the
[`issue-attachments`](https://github.com/akellehe/tessera/releases/tag/issue-attachments)
release and embedded by URL. The 10-CPU cap is honored (thread env set at launch).
