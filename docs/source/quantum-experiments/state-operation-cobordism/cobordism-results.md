# The State–Operation–Cobordism Correspondence: results

> The consolidated results companion to the specification
> [`cobordism.md`](cobordism.md). The spec (§1) states the correspondence as
> three hypotheses **H1/H2/H3**; this report organizes the numerical evidence
> under each — all of it read off the **continuous spectral method** (the
> genuine Hodge spectrum) — ending with the realizable gate set and a
> per-output gallery. Every number below names the runnable example that
> produced it: `realizability_report.py` (boundary synthesis + bulk
> realizability) and `spectral_gate_realizability.py` (the staged spectral
> synthesis: the charge-conservation gate criterion, 13 named gates; its
> `--h3` mode validates the value equation
> $Z_{\text{spec}}=\langle\psi_A|U|\psi_B\rangle$ on the realized set) — both
> under `examples/cobordism/`. The Dijkgraaf–Witten layer that scaffolded and
> calibrated this method — the state-sum functoriality checks, the sign
> invariant, the bridge, and the pinned $S_3$ image — is recorded in
> [the DW scaffold](../earlier-work/dijkgraaf-witten-scaffold.md).

## The correspondence

A quantum operation $U:\mathcal H_B\to\mathcal H_A$ between two states is read as a
**cobordism** whose boundary is the two states and whose TQFT value is their
transition amplitude — the functorial reading of field theory
{cite}`R-Atiyah1988TQFT, R-Segal2004Definition`, with states prepared by bounding
geometries and amplitudes assigned to general boundaries
{cite}`R-HartleHawking1983WaveFunction, R-Oeckl2003GeneralBoundary`. The manifold
is the operation; the amplitude is the number
it computes. Concretely (spec §1):

- **H1.** $W_{AB}=\mathrm{geo}(U)$ is an $n$-cobordism.
- **H2.** $\partial W_{AB}=\overline{\mathrm{geo}(\psi_B)}\sqcup\mathrm{geo}(\psi_A)$.
- **H3.** $Z(W_{AB})=\langle\psi_A|\,U\,|\psi_B\rangle$, with
  $\operatorname{rank}(U)=\text{Schmidt rank of }\operatorname{vec}(U)=\text{connectivity of }W_{AB}$.

**What $\mathrm{geo}(\cdot)$ denotes.** $\mathrm{geo}(\psi)$ is a **carrier**
of the state $\psi$ — a Hermitian-weighted simplicial complex whose (Hodge)
Laplacian has $\psi$ as a distinguished eigenvector (a *harmonic*,
$\psi\in\ker L_k$, on the register layer) — and $\mathrm{geo}(U)$ is a carrier
of the bent state $\operatorname{vec}(U)$, synthesized with its boundary
pinned; the overline in H2 is orientation reversal. The notation is ours, and
it is not a canonical map: carriers need not exist — H1 *is* the existence
question, and the residual floors below are its certified failures — and they
are never unique (all 1443 genuine registers of the topology search below
carry the same gate set). $\mathrm{geo}(\cdot)$ names the **witness the
synthesis returns** (the spec's §4b inverse eigenvector problem for
boundaries; the fixed-boundary interior fill for bulks). What *is* canonical
runs the other way — the field-theoretic assignment of a boundary state to a
bounding geometry
{cite}`R-Atiyah1988TQFT, R-HartleHawking1983WaveFunction, R-Oeckl2003GeneralBoundary`
— and a carrier is a pointwise section of it, found variationally; choosing
*among* the carriers of a realizable operation is exactly the
geometry-selection question the Regge-mediation track takes up. Categorically,
the bent state $\operatorname{vec}(U)$ is the **name** of the morphism $U$
{cite}`R-AbramskyCoecke2004Categorical`; the carrier of a state has no
standard counterpart in the literature.

**The falsifiable core, read spectrally.** The correspondence is *supported*
iff (i) the trivial cobordism reproduces the inner product, (ii) the value is
invariant under interior re-triangulation (the Pachner-move test
{cite}`R-Pachner1987Bistellar, R-Pachner1991Shellings`), and (iii) an obstructed
class is genuinely distinguished — and *refuted* if any fails. All three hold on the
spectral data (`spectral_gate_realizability.py --h3`): (i) the identity's
spectral value equals $\langle\psi_A|\psi_B\rangle$ on every carried pair
(worst $4.5\times10^{-16}$), and it realizes *only* once surgery has grown the
full register — on every smaller seed it floors; (ii) the value carries over
**exactly** to a symmetry-preserving re-triangulation (drift
$9.7\times10^{-16}$), and a generic re-grown bulk deviates by precisely its
register Gram defect, $a^\dagger(G-I)b$, matched to $\sim10^{-15}$; (iii) a
leaked class has **no value at all** — every floored gate's post-state leaves
the carried register ($\lvert\Sigma\rvert=0.21$–$2.60$), the value-level
obstruction certificate. (The original Dijkgraaf–Witten
{cite}`R-DijkgraafWitten1990Topological` formulation of this
core — T1/T2 on the state sum and the T3 sign invariant — is recorded in
[the DW scaffold](../earlier-work/dijkgraaf-witten-scaffold.md).) The sections
below place the rest of the evidence under the hypothesis it bears on.

## H1 — the operation is a cobordism

### Synthesis realizes the bulk, and an obstruction certifies the rest

Bending $U$ to its Choi state $\operatorname{vec}(U)$ and **synthesizing the bulk
spectrally** — pin the boundary, fill the interior, drive the residual
$r(W)=\lVert(I-\psi\psi^\dagger)L\psi\rVert^2$ to zero — cleanly separates the two
outcomes the hypothesis predicts. A **realizable** $U$ is realized with its bulk
$W_{AB}$ ($r\to0$); an **obstructed** $U$ is certified non-realizable by a
**residual floor** bounded away from zero — a spectral obstruction under the
fixed-boundary constraint, not a failure to search hard enough
(`realizability_report.py`; the growable witnesses run with **additions as
well as surgical cuts**, the added vertices capped by
`--max-additional-vertices`, default 20, while the floor control stays pinned
at fixed complexity):

| operation $U$ | bulk | verdict | $r$ / floor | added $\lvert V\rvert$ | cuts | $\lambda$ |
|---|---|---|---|---|---|---|
| $\left(\begin{smallmatrix}1&1\\1&1\end{smallmatrix}\right)$ (zero mode) | bipyramid | **realizable** | $r=9.49\times10^{-11}$ | 0 | 0 | $0.000000$ |
| $\left(\begin{smallmatrix}1&0.3{+}0.5i&-0.8{+}0.2i\end{smallmatrix}\right)$ ($1\times3$) | triangle | **realizable** | $r=3.52\times10^{-11}$ | 1 | 0 | $3.955356$ |
| $\left(\begin{smallmatrix}1&2\\3&4\end{smallmatrix}\right)$ (generic) | bipyramid | **obstructed** | floor $=2.35\times10^{-2}$ | 0 | 0 | $1.176471$ |

The realizable residuals ($\sim10^{-11}$) sit **nine orders of magnitude**
below the obstruction floor ($2.35\times10^{-2}$): the
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
$9.6\times10^{-1}$ with no budget and drops to $3.5\times10^{-11}$ after a single
boundary-fixed cone.

Triangulation invariance of the bulk is certified at the value level by the
H3 bulk-independence result below (the value carries exactly over a
symmetry-preserving re-triangulation, and a generic re-grown bulk deviates by
precisely its measured Gram defect); the state-sum functoriality checks that
first established it (T1/T2/T5 on the Dijkgraaf–Witten functor) are recorded
in [the DW scaffold](../earlier-work/dijkgraaf-witten-scaffold.md).

(The §5.6 Lorentzian variant corroborates the spectral layer
(`topological_correspondence.py`): on the 3-cycle with one timelike edge the
spectrum is exactly $\{0,3,1-2/\alpha\}$ and the harmonic's indefinite norm
$(2-\alpha)/3$ crosses zero at $\alpha=2$ — a concrete "harmonic
representative becomes null," residual $3.36\times10^{-15}$.)

### rank $=$ Schmidt rank $=$ connectivity

The engine of H1 is map–state duality (Choi–Jamiołkowski "bending"
{cite}`R-Choi1975CompletelyPositive, R-Jamiolkowski1972LinearTransformations`, the
wire-bending of categorical quantum mechanics
{cite}`R-AbramskyCoecke2004Categorical`):
$\operatorname{rank}(U)$ equals the Schmidt rank of $\operatorname{vec}(U)$
(the operator-Schmidt rank {cite}`R-Nielsen2003DynamicsResource`), which
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

### H3 on the spectral data: $Z_{\text{spec}}$ equals the amplitude on every realized gate

The value equation is validated on the staged-synthesis register itself —
no DW input anywhere (`spectral_gate_realizability.py --h3`). The spectral
value $Z_{\text{spec}}(W;\psi_A,U\psi_B)$ is the Hodge pairing of the
carried harmonic representatives — harmonic cochains in the sense of
discrete Hodge theory {cite}`R-Eckmann1945Harmonische, R-Lim2020HodgeLaplacians`
— on the surgery-grown bulk (the register
bulk carries the unit cochain metric, so the pairing is the plain Hermitian
contraction), with **one** global scale fixed by the T1 anchor
$Z_{\text{spec}}(\psi_B,\psi_B)=\langle\psi_B|\psi_B\rangle$; after that,
every number — every pair, every gate, every re-grown bulk — is a
prediction with no freedom left.

- **The register chart is a scaled isometry.** The Gram of the period map
  $V\to\ker L_1$ on a flat-orthonormal basis of the $\Sigma=0$ subspace is
  the identity to $7.8\times10^{-16}$. By Schur's lemma this is exactly the
  $S_3$-equivariance of the carried register: $V$ is the irreducible
  standard representation, so any invariant inner product on it is
  proportional to the flat one — the proportionality constant being the one
  scale T1 fixes.
- **$Z_{\text{spec}}=\langle\psi_A|U|\psi_B\rangle$ for every realized
  gate.** Worst pair deviation $5.0\times10^{-16}$ across the 13 criterion
  gates $\times$ 9 carried pairs (the $V$-generic input plus 8 random
  carried $\psi_A$); the independent Choi/operator reading
  (`quantum::ChoiJamiolkowski.transitionAmplitude` on the $\mathbb{C}^4$
  holonomy embedding) agrees to $1.6\times10^{-16}$.
- **A floored gate has no spectral value.** Its post-interaction periods
  leak out of $V$ ($\lvert\Sigma\rvert=0.21$–$2.60$ across the 39 floored
  gates), so no carried representative exists — the value-level
  obstruction certificate, the same mechanism as the residual floor.
- **Bulk independence, with its mechanism.** The symmetry-preserving
  re-triangulation (one geodesic subdivision, each holonomy hole re-placed
  on the central child of its original hole face) carries the value
  *exactly*: Gram defect $7.6\times10^{-16}$, value drift
  $9.7\times10^{-16}$. A generic vertex-disjoint hole draw still realizes
  the same gates, but its chart is anisotropic (Gram defects $0.11$ and
  $0.25$ on two draws) and its value deviation ($0.14$, $0.26$) equals the
  Gram-defect prediction $a^\dagger(G-I)b$ to $\sim10^{-15}$. **The
  value-level H3 is the charge-conservation criterion plus the isometric
  register chart**: surgery decides *which* gates are carried
  (topology-free), and equivariance makes the carried pairing *be* the
  amplitude.

Read at the value level, the falsifiable core says: the identity reproduces
the inner product only once surgery has grown the full register (the T1
anchor), and a leaked class produces *no* value at all rather than a wrong
one.

Before this validation existed, the value equation was certified by
cross-calibrating three independent readings — the DW state sum, the operator
amplitude, and the spectral harmonic — on the torus cylinder, where the DW
invariant turned out to be the **quantized shadow** of the continuous spectral
$Z$. That bridge, together with the T3 sign invariant, is recorded in
[the DW scaffold](../earlier-work/dijkgraaf-witten-scaffold.md); H3 on this
page no longer rests on it.

## The realizable gate set: the charge-conservation criterion

The first construction pinned the cobordism's boundary bit-exact — the restriction of
one global form on a fixed twisted cylinder — and that integrality held its realizable
image down to the six holonomy permutations ($S_3$; recorded with the
$b_1$-hole retest in
[the DW scaffold](../earlier-work/dijkgraaf-witten-scaffold.md)). The final
construction (`spectral_gate_realizability.py`) **synthesizes the boundary instead of pinning it**: it
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
analytic cross-check. Because $V$ is exactly the $\Sigma=0$ subspace, $U$ preserves it
**iff** its $\{[a],[b],[a{+}b]\}$ block conserves total charge — the block's three column
sums are equal (the all-ones covector $c=[1,1,1]$ is a left-eigenvector). This
**closed-form criterion** (`conserves_charge`) agrees with the spectral residual on
*every* gate in the battery, so the realizable set is a *criterion*, not a hand-listed
number — and being a property of $U$'s action, it is independent of the bulk topology.

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
| $\sqrt{\mathrm{SWAP}}^\dagger$ | superposition | $2.8\times10^{-29}$ | $0$ | **yes** |
| `CSX` (ctrl-$\sqrt{X}$) | controlled | $4.4\times10^{-29}$ | $0$ | **yes** |
| `CSXdg` | controlled | $5.5\times10^{-29}$ | $0$ | **yes** |
| `rev-CSX` | controlled | $2.2\times10^{-29}$ | $0$ | **yes** |
| `rev-CSXdg` | controlled | $1.7\times10^{-29}$ | $0$ | **yes** |
| the other **39** (Paulis, $H/S/T$ singles, `CZ`/`CY`/`CH`/`CS`, `iSWAP`, $\sqrt{\mathrm{iSWAP}}$, Magic, Mølmer–Sørensen, …) | superposition / phase / entangler | $0.2$–$8.5$ | $0.2$–$2.6$ | **no** (certified) |

The realizable set is a **criterion**, not a number: $U$ realizes iff its
$\{[a],[b],[a{+}b]\}$ block conserves total charge (equal column sums), and that closed
form agrees with the spectral residual on every gate. The criterion cuts out a continuous
group; among the standard named gates **13** satisfy it — the full $S_3$ (6),
$H{\otimes}H$, the controlled-$\sqrt{X}$-power family on either qubit ($\mathrm{CSX}$,
$\mathrm{CSX}^\dagger$, and their control-$B$ mirrors), and the $\sqrt{\mathrm{SWAP}}$
roots ($\sqrt{\mathrm{SWAP}}$, $\sqrt{\mathrm{SWAP}}^\dagger$). The realize/floor split is
$\sim28$ orders of magnitude: the carried gates hit machine zero (genuine harmonics of
$L_1$), the other 39 floor at $0.2$–$8.5$, each a certified obstruction (its
post-interaction state leaks out of $\ker L_1$, $\Sigma\neq0$).

**The verdict (boundary synthesized).** The staged spectral synthesis realizes exactly the
gates whose holonomy-class block **conserves charge** — a closed-form criterion, not a
count. Synthesizing the boundary (rather than pinning it bit-exact) lifts the integrality
constraint that held the DW image down to the six permutations, and what replaces it is
not a slightly longer list but a **continuous group**: every charge-conserving gate. Among
standard named gates that is the full $S_3$, $H{\otimes}H$, the controlled-$\sqrt{X}$-power
family on either qubit, and the $\sqrt{\mathrm{SWAP}}$ roots — **13** in this battery, with
the count tracking the battery, not the physics. (An earlier 18-gate battery reported
"$S_3 + H{\otimes}H + \sqrt{\mathrm{SWAP}} = 8$"; that was its realizable *subset*, missing
$\mathrm{CSX}$ and $\sqrt{\mathrm{SWAP}}^\dagger$.) What still floors is genuine register
*leakage* ($\Sigma\neq0$): a holonomy superposition or off-lattice phase that no emergent
$b_1$ can carry — the $k=1$ analogue of the sign-flipped meridian that floors on every
filling
([the DW scaffold](../earlier-work/dijkgraaf-witten-scaffold.md) records that
result).

**The bigger search confirms the criterion.** A parallelized high-retry surgery-topology
search (`spectral_gate_realizability.py --retries N --jobs 10` — ten worker processes, each
pinned to one BLAS thread, so $\text{procs}\times\text{threads}\le10$) asks directly whether
a *richer* emergent register carries a gate beyond the criterion set. Each retry re-runs the
identical staged synthesis on a randomized surgery-grown bulk: a different triangulated-$S^2$
seed (the icosahedron and its geodesic subdivisions), a different vertex-disjoint
holonomy-hole triple, extra `removeInteriorCell` surgeries that grow $b_1$ (up to $5$
here), and the move-set's **additive half** — up to `--max-additional-vertices` (default
$20$) vertices added per draw by boundary-fixed stellar subdivision, composed from the two
surgery primitives (the cone fan attached over an interior top cell's edges, the face then
removed: $\partial W$ bit-exact, $\ker L_1$ preserved, the bulk re-pinned to the unit
cochain metric by construction). Over **3 000** cuts-only topologies (seeds up to
$|V|=162$) the realizable set never grows. The
**1 282 genuine** registers — a *proper* carried subspace ($\mathrm{rank}\,P<\#\text{holes}$,
with the identity and all six $S_3$ controls still realizing) — each realize *exactly* the
same charge-conserving gates (the 13). The other **1 718** draws **saturate** the
holonomy-period space ($\mathrm{rank}\,P=\#\text{holes}$): $V$ becomes the whole period
space, so every gate trivially "realizes" — but that is the **dissolution** of the register,
not the carrying of a new gate (no proper subspace is left for a state to leak out of, so the
obstruction, and with it the discriminating content of "realizable", is gone). With the
additive move enabled the conclusion is unchanged: over **300** further draws mixing cuts
with up to $20$ added vertices (one draw using all $20$; refined bulks reaching
$|V|=182$), the **161 genuine** registers again realize exactly the 13 and the remaining
**139** saturate. This is no
accident: the criterion is a property of $U$'s holonomy-class block, *not* of the bulk
topology, so no surgery — subtractive or additive — can change it. Growing $b_1$, or
refining the bulk with added vertices, buys no new *operations* — the
operation-side echo of the $b_1$-hole result in
[the DW scaffold](../earlier-work/dijkgraaf-witten-scaffold.md): surgery
enlarges the realizable *state* space, not the realizable *gate* set.

## Relation to the literature

The correspondence itself is the functorial reading of field theory
{cite}`R-Atiyah1988TQFT, R-Segal2004Definition` instantiated on discrete data;
what this report tests is whether tessera's machinery genuinely realizes it.
The gate-set question — *which unitaries does a topological theory realize?*
— is the founding question of topological quantum computation
{cite}`R-Freedman2003TQC, R-FreedmanKitaevWang2002Simulation, R-FreedmanLarsenWang2002Universal`, where the known answer-shape for abelian
theories is a **finite** braiding/mapping-class image (the Property-F
circle {cite}`R-NaiduRowell2011PropertyF`): the pinned $S_3$ image recorded in
[the DW scaffold](../earlier-work/dijkgraaf-witten-scaffold.md) is exactly
that expectation. The move that enlarged it — topology change as the
computational operation itself — has established cousins in lattice surgery
{cite}`R-Horsman2012LatticeSurgery` and in twist defects and genons enlarging
the power of abelian phases
{cite}`R-Bombin2010Twist, R-BarkeshliJianQi2013TwistDefects`, formulated there
in stabilizer-code rather than spectral language.

The spectral value reading has a precise neighborhood without, to our
knowledge, a direct precedent. Laplacian data enter TQFT classically as
*determinant prefactors*: abelian BF/Chern–Simons partition functions equal
Ray–Singer analytic torsion {cite}`R-RaySinger1971RTorsion, R-Schwarz1978Torsion, R-Witten1989Jones`, reproduced on triangulations by simplicial Hodge theory
{cite}`R-Adams1996RTorsion, R-Adams1997DoubledCS`. Harmonic zero modes serve as
the abelian Chern–Simons state space in canonical quantization
{cite}`R-BosNair1989Blocks`; cellular BF theory builds boundary state spaces
from the cohomology of a cellular cobordism
{cite}`R-CattaneoMnevReshetikhin2020Cellular`; and projections onto
$\ker L_k$ with overlaps against harmonic representatives are the
operational core of quantum topological-data analysis
{cite}`R-Lloyd2016TopologicalData`. What this report adds to that neighborhood
is the amplitude itself read as a T1-anchored Hodge pairing of carried
harmonic representatives on a *synthesized* triangulated cobordism, validated
against the operator amplitude at machine precision.

The realizability methodology likewise has strong cousins: which boundary
*operators* admit an interior geometry, with non-existence certified by
obstructions rather than enumeration, is classical for circular planar
resistor networks {cite}`R-CurtisIngermanMorrow1998Circular`; existence-as-
optimization with rigorous non-existence certificates is the shape of the
quantum marginal hierarchy {cite}`R-Yu2021MarginalHierarchy`; *which boundary
data admit a bulk geometry* is the holographic entropy cone
{cite}`R-Bao2015EntropyCone`; and least-squares formulations of inverse
spectral problems are canonical {cite}`R-ChuGolub2005InverseEigenvalue`. The
instantiation here — an eigenvector-residual floor on synthesized simplicial
bulks, cross-checked against independent global minima, as the
non-realizability certificate — is the report's own combination of those
ingredients.

## Figures

A per-output force-directed render of the simplicial complexes the experiments
build (`force_layout_3d` / `layout_from_spacetime`, vertices colored by
amplitude magnitude where a state is carried).

![A synthesized boundary state geo(psi)](https://github.com/akellehe/tessera/releases/download/issue-attachments/cobordism_results_geo_psi.png)
*Figure 1 — A synthesized boundary state $\mathrm{geo}(\psi_A)$.* The minimal
Hermitian-weighted complex whose $k=0$ Laplacian carries $\psi_A$ as an
eigenvector; vertices are colored by $\lvert\psi\rvert$ (phase $\to$ hue),
auxiliary zero-amplitude vertices desaturated.

The staged spectral synthesis renders its own outputs with the same idiom
(`spectral_gate_realizability.py --all-plots`): holonomy-hole edges thickened, edges
colored by the carried 1-form (hue $=$ phase, brightness $=|\text{amp}|$).

![The surgery-grown S^2 bulk](https://github.com/akellehe/tessera/releases/download/issue-attachments/spectral_gate_grown_bulk.png)
*Figure 2 — The surgery-grown bulk $W$ (staged synthesis).* The icosahedral $S^2$ with the
three holonomy holes opened by `removeInteriorCell` ($b_1=2$); the thick edges are the three
boundary 1-cycles $\{[a],[b],[a{+}b]\}$.

![The emergent register ker L_1](https://github.com/akellehe/tessera/releases/download/issue-attachments/spectral_gate_register.png)
*Figure 3 — The emergent register $V=\ker L_1$.* The 2-dimensional $S_3$ standard
representation read from the spectrum (one carried harmonic shown on the register edges).

![A synthesized boundary state geo(psi_B)](https://github.com/akellehe/tessera/releases/download/issue-attachments/spectral_gate_geo_psiB.png)
*Figure 4 — A synthesized boundary state $\mathrm{geo}(\psi_B)$.* The generic register
input, carried as a harmonic of $L_1$ (the stage-1 synthesis).

![A realized gate sqrt-SWAP](https://github.com/akellehe/tessera/releases/download/issue-attachments/spectral_gate_realized_sqrtswap.png)
*Figure 5 — A realized gate, $\sqrt{\mathrm{SWAP}}$.* The post-interaction state
$U|\psi_B\rangle$ stays in $\ker L_1$ ($r\to0$) — the non-integer register automorphism the
synthesized boundary admits.

(The renders of the pinned twisted-cylinder cobordism and the disk/annulus/
surgery meridian fillings live with their experiments in
[the DW scaffold](../earlier-work/dijkgraaf-witten-scaffold.md).)

## Method, conventions, and reproduction

All experiments are pure orchestration of separately-tested classes —
`ChoiJamiolkowski` (the bend and the operator amplitude), `HodgeLaplacian`
(the $k=0$ and $k\ge1$ Laplacians, `weights`, `harmonics`), and
`EigenstateSynthesis` / `RealizabilityOracle` (the fixed-boundary interior
fill and its `decideHarmonic` / surgery modes). Conventions held throughout:

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

To reproduce (default Release build; the fast linker is auto-gated off Release so
`bfd` links the LTO `_tessera`):

```
pip install -e ".[dev]" -C cmake.define.TESSERA_CUDA=OFF
python examples/cobordism/realizability_report.py         # boundary synthesis + realizability
python examples/cobordism/spectral_gate_realizability.py                   # the staged gate set (charge-conservation criterion, 13 named)
python examples/cobordism/spectral_gate_realizability.py --h3              # H3 at the value level: Z_spec = <psi_A|U|psi_B> on the realized set
python -m pytest tests/cobordism/test_spectral_h3_python.py                # the pinned H3 invariants
python examples/cobordism/spectral_gate_realizability.py --gate sqrt-SWAP   # solve for one named gate (52-gate battery)
python examples/cobordism/spectral_gate_realizability.py --retries 10000 --jobs 10  # parallel surgery-topology search, cuts + additions (--max-additional-vertices caps the added vertices, default 20)
python examples/cobordism/spectral_gate_realizability.py --all-plots        # force-directed renders → issue-attachments
python -m pytest tests/cobordism
```

Parameter sweeps, raw tables, and figures are written to `/tmp/cobordism/` and are
**not committed** — the example scripts and this report are the committed
artifacts; the figures above are uploaded to the
[`issue-attachments`](https://github.com/akellehe/tessera/releases/tag/issue-attachments)
release and embedded by URL. The 10-CPU cap is honored (thread env set at launch).

## References

```{bibliography}
:filter: docname in docnames
:keyprefix: R-
:labelprefix: R
:style: unsrt
```
