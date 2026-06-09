# The State–Operation–Cobordism Correspondence: results

> The consolidated results companion to the specification
> [`cobordism.md`](cobordism.md). The spec (§1) states the correspondence as
> three hypotheses **H1/H2/H3**; this report organizes the numerical evidence
> under each, ending with the realizable gate set and a per-output gallery.
> Every number below names the runnable example that produced it:
> `topological_correspondence.py` (the Stage-2 TQFT checks T1–T5),
> `realizability_report.py` (boundary synthesis + bulk realizability),
> `dw_spectral_bridge.py` (the DW–spectral bridge), and
> `realizable_image_sweep.py` / `loosened_gate_retest.py` /
> `spectral_gate_realizability.py` (the gate set: pinned ($S_3$), loosened at
> $k=0$, and the staged spectral synthesis with the boundary synthesized
> ($S_3+H{\otimes}H+\sqrt{\mathrm{SWAP}}$)) — all under `examples/cobordism/`.

## The correspondence

A quantum operation $U:\mathcal H_B\to\mathcal H_A$ between two states is read as a
**cobordism** whose boundary is the two states and whose TQFT value is their
transition amplitude. The manifold is the operation; the amplitude is the number
it computes. Concretely (spec §1):

- **H1.** $W_{AB}=\mathrm{geo}(U)$ is an $n$-cobordism.
- **H2.** $\partial W_{AB}=\overline{\mathrm{geo}(\psi_B)}\sqcup\mathrm{geo}(\psi_A)$.
- **H3.** $Z(W_{AB})=\langle\psi_A|\,U\,|\psi_B\rangle$, with
  $\operatorname{rank}(U)=\text{Schmidt rank of }\operatorname{vec}(U)=\text{connectivity of }W_{AB}$.

**The falsifiable core.** The correspondence is *supported* iff (i) the cylinder
cobordism reproduces the inner product, (ii) $Z(W)$ is invariant under interior
re-triangulation, and (iii) a nontrivial sign class produces a $Z(W)$ distinct
from the trivial one — and *refuted* if any fails. All three hold: (i) the
torus cylinder is exactly the identity (T1, residual $6.28\times10^{-16}$); (ii)
$Z$ drifts by $0$ over 18 interior Pachner moves (T2); (iii)
$Z_{\text{sign}}(\mathbb{RP}^3)=0\neq1=Z_{\text{triv}}(\mathbb{RP}^3)$ (T3). The
sections below place the rest of the evidence under the hypothesis it bears on.

## H1 — the operation is a cobordism

### Synthesis realizes the bulk, and an obstruction certifies the rest

Bending $U$ to its Choi state $\operatorname{vec}(U)$ and **synthesizing the bulk
spectrally** — pin the boundary, fill the interior, drive the residual
$r(W)=\lVert(I-\psi\psi^\dagger)L\psi\rVert^2$ to zero — cleanly separates the two
outcomes the hypothesis predicts. A **realizable** $U$ is realized with its bulk
$W_{AB}$ ($r\to0$); an **obstructed** $U$ is certified non-realizable by a
**residual floor** bounded away from zero — a spectral obstruction under the
fixed-boundary constraint, not a failure to search hard enough
(`realizability_report.py`):

| operation $U$ | bulk | verdict | $r$ / floor | cones | interior $\lvert V\rvert$ | $\lambda$ |
|---|---|---|---|---|---|---|
| $\left(\begin{smallmatrix}1&1\\1&1\end{smallmatrix}\right)$ (zero mode) | bipyramid | **realizable** | $r=9.49\times10^{-11}$ | 0 | 0 | $0.000000$ |
| $\left(\begin{smallmatrix}1&0.3{+}0.5i&-0.8{+}0.2i\end{smallmatrix}\right)$ ($1\times3$) | triangle | **realizable** | $r=1.19\times10^{-13}$ | 1 | 1 | $3.923719$ |
| $\left(\begin{smallmatrix}1&2\\3&4\end{smallmatrix}\right)$ (generic) | bipyramid | **obstructed** | floor $=2.35\times10^{-2}$ | 0 | 0 | $1.176471$ |

The realizable residuals ($\sim10^{-11}$, $\sim10^{-13}$) sit **nine to twelve
orders of magnitude** below the obstruction floor ($2.35\times10^{-2}$): the
separation is unambiguous. The floor is a genuine non-existence certificate, not
a stuck optimizer — it is seed-independent, equals $2/85$ to machine precision,
and matches an independent numpy global-min over the lone interior edge
($2.352941\times10^{-2}$ certified vs $2.382794\times10^{-2}$ on a grid,
$\Delta=3.0\times10^{-4}$). Obstruction is also *generic at fixed complexity*:
interpolating from the zero mode toward $\left(\begin{smallmatrix}1&2\\3&4\end{smallmatrix}\right)$
on the one-interior-edge bipyramid, the residual lifts off zero immediately
($t{=}0$: $9.5\times10^{-11}$, realizable; $t{=}0.1$: $5.3\times10^{-3}$, floored)
and saturates near $2.4\times10^{-2}$. Growth converts an obstruction into a
realization where the topology allows it — the $1\times3$ operation floors at
$9.6\times10^{-1}$ with no budget and drops to $1.2\times10^{-13}$ after a single
boundary-fixed cone.

### The bulk is a triangulation-invariant TQFT (T1, T2, T5)

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

(The §5.6 Lorentzian variant corroborates: on the 3-cycle with one timelike edge
the spectrum is exactly $\{0,3,1-2/\alpha\}$ and the harmonic's indefinite norm
$(2-\alpha)/3$ crosses zero at $\alpha=2$ — a concrete "harmonic representative
becomes null," residual $3.36\times10^{-15}$.)

### rank $=$ Schmidt rank $=$ connectivity

The engine of H1 is map–state duality (Choi–Jamiołkowski "bending"):
$\operatorname{rank}(U)$ equals the Schmidt rank of $\operatorname{vec}(U)$, which
equals the connectivity of $W_{AB}$. A rank-1 $U$ factors through the unit object,
so its cobordism is **disconnected** (a separable bent state); a full-rank $U$
gives a **connected** cobordism (an entangled bent state). The realized witnesses
instantiate it directly — the rank-1 zero mode is realized at minimal interior
complexity ($\texttt{interior\_vertex\_count}=0$), the separable end of the
realizable spectrum.

## H2 — the boundary is the two states

### Independent boundary synthesis (§4b)

The boundary objects $\mathrm{geo}(\psi_A)$, $\mathrm{geo}(\psi_B)$ are built
**independently**, by solving the inverse eigenvector problem: given a target
state, find the *simplest* complex whose Laplacian has it as an eigenvector
(`realizability_report.py`). A general-amplitude qubit
$\psi=(\sqrt{0.8},\sqrt{0.2})$ with $\lvert c_0\rvert\neq\lvert c_1\rvert$ is **not**
a balanced two-vertex eigenvector — the two-vertex seed $K_2$ floors at
$r=3.60\times10^{-3}$ — and is synthesized as $\mathrm{geo}(\psi)$ on the
**minimal** complex $K_3$: $\lvert V\rvert=3$, $\lvert E\rvert=3$, one cone,
$r=9.39\times10^{-10}$, $\lambda=0.199908$. The minimal vertex count is the
state's recorded combinatorial complexity.

### $\partial W$ is exactly the two boundary states

The fill rewrites **only interior edges**, so $\partial W$ is byte-identical
before and after synthesis. On the bipyramid witness `getBoundary()` returns
exactly the pinned boundary $\partial W=\{(0,2),(0,3),(1,2),(1,3)\}$ (the four
boundary edges) and **excludes** the interior edge $(0,1)$, which stays pinned.
Read with orientation, this is
$\partial W_{AB}=\overline{\mathrm{geo}(\psi_B)}\sqcup\mathrm{geo}(\psi_A)$ — the
outgoing $\mathrm{geo}(\psi_A)$ and the orientation-reversed incoming
$\mathrm{geo}(\psi_B)$.

## H3 — the value is the amplitude

### The DW–spectral bridge: three independent readings agree (and only on the lattice)

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

### The sign carries an invariant (T3), and the holonomy is consistent (T4)

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

## The realizable gate set: $S_3$ is the pinned operation image

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
$\sqrt{\mathrm{SWAP}}$). Realizability is a property of the holonomy action,
indifferent to a gate's name:

| gate | realizable? | reason (action on the holonomy classes) |
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
(homology classes on $\partial W$), not for gate operations. The H3 quantized-shadow
reading says the same from the value side: the $\mathbb{Z}_2$-DW operation image is a
discrete lattice that a generic superposition/entangling $U$ is uniformly bounded
away from (gap median $1.03$ over 200 Haar $U(4)$), and the continuous freedom the hole
supplies lives in the *state* — the spectral $Z$ — not in new realizable maps.

## The gate set with the boundary synthesized: $S_3 + H{\otimes}H + \sqrt{\mathrm{SWAP}}$

The $S_3$ image above pins the cobordism's boundary bit-exact — it is the restriction of
one global form on a fixed twisted cylinder, and that integrality is exactly what holds
the realizable set down to the six holonomy permutations. A third construction
(`spectral_gate_realizability.py`) **synthesizes the boundary instead of pinning it**: it
runs each gate through a *staged spectral synthesis* and decides realizability by the
**continuous spectral method** — the genuine Hodge Laplacian spectrum, $\ker L_1$ of the
surgery-grown bulk read by eigendecomposition — with **no** Levenberg–Marquardt
weight/topology fill. The hypothesis it tests directly: synthesizing the boundary too,
rather than pinning it, relaxes the integrality over-constraint and so realizes *more*
gates.

**The construction — a 3-stage staged spectral synthesis (per gate $U$).** The register
is the torus holonomy $\mathbb{C}^4=\mathbb{C}[H^1(T^2;\mathbb{Z}_2)]$; the three
non-trivial classes $\{[a],[b],[a{+}b]\}$ are carried as three vertex-disjoint boundary
1-cycles on an **$S^2$ bulk** (a triangulated icosahedron), and a gate acts on the
register by its $\{[a],[b],[a{+}b]\}$ block.

1. **Synthesize each state independently** (the §4b boundary synthesis). $\mathrm{geo}(\psi_A)$
   and $\mathrm{geo}(\psi_B)$ are grown *separately* into the minimal complex whose metric
   Hodge $L_1$ carries the register state as a **harmonic** ($\psi\in\ker L_1$), confirmed
   by the genuine spectral residual $\lVert(I-\psi\psi^\dagger)L_1\psi\rVert^2\to0$ — each
   on its own, the input and output geometries never meeting at this stage.
2. **Fix their union as the boundary.** $\partial W=\mathrm{geo}(\psi_A)\sqcup\mathrm{geo}(\psi_B)$
   is held as the (pinned) boundary. Because the two were grown apart, $\partial W$ is
   **not** the bit-exact restriction of one global form — the relaxation the hypothesis
   turns on.
3. **Grow the bulk to the known post-interaction state, with surgery, and decide by the
   spectrum.** The topology-changing surgery move `removeInteriorCell` opens the three
   holonomy holes so $b_1$ and $\ker L_1$ **emerge** $0\to2$ (the closed sphere has
   $\ker L_1=0$): the carried register $V$ becomes the 2-dimensional $S_3$ standard
   representation. Realizability is then the **spectral** statement of
   $Z_{\text{spec}}(W)=\langle\psi_A|U|\psi_B\rangle$: form the post-interaction state
   $U|\psi_B\rangle$ as a boundary 1-form and measure its genuine Hodge residual
   $r(U)=\lVert(I-\psi\psi^\dagger)L_1\psi\rVert^2$ on the grown bulk. $r\to0$ **iff**
   $U|\psi_B\rangle$ lies in $\ker L_1$ — i.e. iff the post-interaction state is *carried*.
   The decision is the eigendecomposition of the real $L_1$, continuous and exact (no
   restart noise).

The carried register is the $\Sigma=0$ subspace of the three holonomy-cycle periods (the
boundary periods of a 3-hole sphere sum to zero, with the induced-orientation signs
$(+,+,-)$ read off the bulk and symmetrized). A single $(\psi_A,\psi_B)$ probes $U$ on one
register vector, so the spectral test is driven on a **$V$-generic input** ($\Sigma=0$,
all components non-zero), whose $U$-image leaks for *any* $U$ that does not preserve the
whole register; the leakage $|\Sigma(U|\psi_B\rangle)|$ is reported alongside as the
analytic cross-check.

**The identity sanity check (the falsifiable core), decided spectrally.** Surgery grows
$\ker L_1$ as $0\to0\to1\to2$ (the closed $S^2$, then the disk, the annulus, the 3-hole
sphere). The identity post-interaction state ($U=I$, $Z_{\text{spec}}=\langle\psi_A|\psi_B\rangle$)
**floors** on every seed with $\ker L_1<2$ ($r\approx1.1\times10^{1}$ — no register to
carry it) and **realizes** only once surgery opens $b_1\,0\to2$ ($r\approx1\times10^{-29}$,
machine zero): the emergent register is load-bearing, exactly the state-test mechanism.
Stage 1 likewise reads $\lVert(I-\psi\psi^\dagger)L_1\psi\rVert^2\approx10^{-29}$ for each
synthesized $\mathrm{geo}(\psi)$ — carried.

**The realizable set (the spectral output).** Scored by the genuine $L_1$ residual of
$U|\psi_B\rangle$ on the surgery-grown register ($b_1=2$ throughout):

| gate | family | spectral residual $r(U)$ | leak $\lvert\Sigma\rvert$ | realizes? |
|---|---|---|---|---|
| Identity | $S_3$ control | $1.1\times10^{-29}$ | $0$ | **yes** |
| `SWAP` | $S_3$ control | $2.9\times10^{-29}$ | $0$ | **yes** |
| `CNOT` | $S_3$ control | $7.8\times10^{-29}$ | $0$ | **yes** |
| reversed-`CNOT` | $S_3$ control | $2.9\times10^{-29}$ | $0$ | **yes** |
| 3-cycle $(0231)$ | $S_3$ control | $5.7\times10^{-29}$ | $0$ | **yes** |
| 3-cycle $(0312)$ | $S_3$ control | $5.9\times10^{-29}$ | $0$ | **yes** |
| $H{\otimes}H$ | superposition | $3.7\times10^{-29}$ | $0$ | **yes** |
| $\sqrt{\mathrm{SWAP}}$ | superposition | $2.8\times10^{-29}$ | $0$ | **yes** |
| $I{\otimes}H$ | superposition | $0.54$ | $0.28$ | no |
| $T{\otimes}I$ | phase | $2.87$ | $0.77$ | no |
| $\sqrt{\mathrm{iSWAP}}$ | superposition | $4.21$ | $1.00$ | no |
| `CPHASE`$(\pi/4)$ | phase/entangler | $4.34$ | $1.00$ | no |
| $H{\otimes}I$ | superposition | $5.62$ | $1.20$ | no |
| $S{\otimes}I$ | phase | $6.96$ | $1.41$ | no |
| `CZ`, $Z{\otimes}Z$ | phase / sign | $7.30$ | $2.60$ | no |
| `iSWAP` | phase/entangler | $8.16$ | $1.84$ | no |
| $X{\otimes}X$ | Pauli perm | $8.44$ | $1.30$ | no |

The realizable set is $\{I,\,\mathrm{SWAP},\,\mathrm{CNOT},\,\text{reversed-CNOT},\,
(0231),\,(0312),\,H{\otimes}H,\,\sqrt{\mathrm{SWAP}}\}$ — **8 gates**, the full $S_3$ plus
$H{\otimes}H$ plus $\sqrt{\mathrm{SWAP}}$. The realize/floor split is $\sim28$ orders of
magnitude: the carried gates hit machine zero (genuine harmonics of $L_1$), the others
floor at $0.5$–$8.4$, each a certified obstruction (its post-interaction state leaks out
of $\ker L_1$, $\Sigma\neq0$).

**The verdict (boundary synthesized).** The staged spectral synthesis realizes
**$S_3 + H{\otimes}H + \sqrt{\mathrm{SWAP}} = 8$** — *one more* than the pinned
fixed-boundary $S_3 + H{\otimes}H = 7$, and far more than the topology-free $\{I\}$. The
extra gate is $\sqrt{\mathrm{SWAP}}$: a **non-integer** register automorphism whose
post-interaction state still lands in the carried register $\ker L_1$, admissible only
once the boundary is *synthesized* rather than pinned to an integer monodromy. This is
exactly the relaxation the hypothesis predicted — synthesizing the boundary, not pinning
it bit-exact, lifts the integrality constraint and enlarges the realizable set. What still
floors is genuine register *leakage* ($\Sigma\neq0$): a holonomy superposition or
off-lattice phase that no emergent $b_1$ can carry, the $k=1$ analogue of the sign-flipped
meridian that floors on every filling.

## Figures

A per-output force-directed render of the simplicial complexes the experiments
build (`force_layout_3d` / `layout_from_spacetime`, vertices colored by
amplitude magnitude where a state is carried).

![A realizable gate's cobordism W: the SWAP twisted cylinder](https://github.com/akellehe/tessera/releases/download/issue-attachments/cobordism_results_gate_W.png)
*Figure 1 — A realizable gate as a cobordism.* The `SWAP` gate's bulk
$W=\texttt{twistedCylinder}(T^2,\,\text{swap})$ — the $(1\,2)$ holonomy
permutation. `SWAP` is realizable because it fixes the trivial class and exchanges
$[a]\leftrightarrow[b]$.

![A synthesized boundary state geo(psi)](https://github.com/akellehe/tessera/releases/download/issue-attachments/cobordism_results_geo_psi.png)
*Figure 2 — A synthesized boundary state $\mathrm{geo}(\psi_A)$.* The minimal
Hermitian-weighted complex whose $k=0$ Laplacian carries $\psi_A$ as an
eigenvector; vertices are colored by $\lvert\psi\rvert$ (phase $\to$ hue),
auxiliary zero-amplitude vertices desaturated.

![The disk filling, b1=0](https://github.com/akellehe/tessera/releases/download/issue-attachments/cobordism_results_disk.png)
*Figure 3 — The disk filling ($b_1=0$).* The octahedron minus one face: the
boundary meridian bounds the still-filled antipodal cell, so it floors.

![The annulus filling, b1=1](https://github.com/akellehe/tessera/releases/download/issue-attachments/cobordism_results_annulus.png)
*Figure 4 — The annulus filling ($b_1=1$).* The octahedron minus two antipodal
faces: the meridian survives in $H_1$, so the cobordism carries it as a bulk
harmonic.

![The surgery-grown bulk](https://github.com/akellehe/tessera/releases/download/issue-attachments/cobordism_results_surgery.png)
*Figure 5 — The surgery-grown bulk.* From the disk seed, the boundary-fixed remove
opens the handle ($b_1:0\to1$) on its own under residual minimization, and the
floored meridian realizes.

## Method, conventions, and reproduction

All experiments are pure orchestration of separately-tested classes —
`ChoiJamiolkowski` (the bend), `HodgeLaplacian` (the $k=0$ and $k\ge1$
Laplacians), `EigenstateSynthesis` / `RealizabilityOracle` (the fixed-boundary
interior fill and its `decideHarmonic` / surgery modes), `DijkgraafWitten` (the
$\mathbb{Z}_2$ state sum + `map` / `amplitude`), `BoundaryStateSpace` (the
$\ker L_1(\Sigma)\to Z(\Sigma)$ preparation), and `Cobordism.twistedCylinder` (the
realizable maps). Conventions held throughout:

- Operators are flat **row-major**; $\operatorname{vec}(U)$ is the row-major
  flatten (`ChoiJamiolkowski.vectorize`), the Choi bend.
- The $k=0$ Laplacian keeps the **magnitude** degree convention
  $D_{ii}=\sum_j\lvert A_{ij}\rvert$ (Hermitian $L$, unitary evolution); the
  eigenvalue is the Rayleigh quotient $\lambda=\psi^\dagger L\psi$. The $k\ge1$
  metric weights are honest signed simplex volumes (Lorentzian preserves the
  sign).
- A target is **realizable** iff the residual is driven below
  $\epsilon=10^{-10}$; an obstruction is a residual floor bounded above $10^{-2}$
  and cross-checked against an independent numpy global-min. The boundary
  $\partial W$ is **pinned** — the fill rewrites only interior edges.
- The DW reading is **metric-free** (topology + cocycle only); the Frobenius norm
  $\lVert\cdot\rVert$ scores $\texttt{gap\_to\_S3}=\min_{g\in S_3}\lVert U-g\rVert$,
  with $\texttt{realizable}:=\texttt{gap\_to\_S3}<10^{-7}$.

To reproduce (default Release build; the fast linker is auto-gated off Release so
`bfd` links the LTO `_tessera`):

```
pip install -e ".[dev]" -C cmake.define.TESSERA_CUDA=OFF
python examples/cobordism/topological_correspondence.py   # T1–T5 + Lorentzian
python examples/cobordism/realizability_report.py         # boundary synthesis + realizability
python examples/cobordism/dw_spectral_bridge.py           # the DW–spectral bridge
python examples/cobordism/realizable_image_sweep.py       # the pinned gate set (S₃)
python examples/cobordism/emergent_bulk_realizability.py  # b₁ as an output
python examples/cobordism/loosened_gate_retest.py         # the loosened gate re-test (k=0)
python examples/cobordism/spectral_gate_realizability.py  # the staged gate set (S₃ + H⊗H + √SWAP)
python -m pytest tests/cobordism
```

Parameter sweeps, raw tables, and figures are written to `/tmp/cobordism/` and are
**not committed** — the example scripts and this report are the committed
artifacts; the figures above are uploaded to the
[`issue-attachments`](https://github.com/akellehe/tessera/releases/tag/issue-attachments)
release and embedded by URL. The 10-CPU cap is honored (thread env set at launch).
