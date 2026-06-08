# Cobordism experiment — the DW–spectral bridge: results

> What the bridge layer (the v0.4 capstone, #174 step 3 / #177) built, and what it
> shows: three *independent* readings of a cobordism's value on a shared boundary
> surface, and an honest characterization of where they agree. Companion to the
> runnable capstone `examples/cobordism/dw_spectral_bridge.py`; mirrors the
> realizability report (`cobordism-realizability-report.md`) and the Stage-2 report
> (`cobordism-stage2-report.md`); grounds in the charter `cobordism.md` §5.

## Verdict

The bridge **holds on the DW-representable subset, and only there** — exactly as
the hypothesis predicts. On a shared boundary surface $\Sigma=T^2$ the value of a
cobordism at a pair of prepared boundary states is computed three independent ways
— the topological Dijkgraaf–Witten state sum, the operation's transition
amplitude, and the spectral Hodge harmonic — and **all three agree to machine
precision** when the operation is the one the cobordism actually implements
($U=\mathrm{id}=Z(T^2\times[0,T])$). For a *generic* $U$ they **disagree**: the
$\mathbb{Z}_2$ DW maps are a discrete, integer-quantized family, so a generic $U$
is not $Z_{DW}$ of any $\mathbb{Z}_2$ cobordism. The spectral oracle, by contrast,
realizes a continuum (the longitude and a neighborhood of it; only the meridian is
obstructed) and therefore **strictly extends** the topological theory. The DW
invariant is the *quantized shadow* of the spectral $Z$. The capstone runs clean,
deterministic, and `exit 0`:

```
DW-spectral bridge -- three independent readings of a cobordism's value Z(W_AB)

  Sigma = T^2  (b_1 = 2);  Z(Sigma) = C[H^1(T^2;Z_2)] = C^4;  spectral qubit ker L_1(T^2) = C^2
  W_AB(topological) = T^2 x [0,T] cylinder  ->  Z(W) = id_4  (metric-free state sum)
  W_AB(spectral)    = solid torus D^2 x S^1  ->  carries the longitude harmonic (k=1)

  spectral mode certification (RealizabilityOracle.decideHarmonic on the solid torus):
    longitude  REALIZABLE   r=9.65e-29  lambda=-4.25e-17  witness d-overlap=1.000000
    meridian   OBSTRUCTED   floor=1.526e+01                       (bounds a disk in W -> dies in H_1(W))
    meridian floor cross-check vs numpy Hodge oracle: 15.2567 vs 15.2567  (Delta 5.3e-15)  AGREE

  three readings on longitude-aligned states  psi_A = a * l_hat,  psi_B = b * l_hat:

  (a, b)                       Z_DW (topo)          <A|U=id|B> (op)      Z_spec (Hodge)        max Delta
  ------------------------------------------------------------------------------------------------------
  1, 1                         +1.000000+0.000000i  +1.000000+0.000000i  +1.000000+0.000000i     1.1e-15
  e^{i pi/3}, 0.6+0.8i         +0.992820-0.119615i  +0.992820-0.119615i  +0.992820-0.119615i     7.8e-16
  0.5-0.5i, i                  -0.500000+0.500000i  -0.500000+0.500000i  -0.500000+0.500000i     7.9e-16
  2, 0.25+0.97i                +0.500000+1.940000i  +0.500000+1.940000i  +0.500000+1.940000i     1.4e-15
  ------------------------------------------------------------------------------------------------------

  bridge on the DW-representable U vs generic U  (psi_A = e^{i pi/3} l_hat, psi_B = (0.6+0.8i) l_hat):

  operation U                    representable? <A|U|B>              Delta = |amp - Z_DW|  verdict
  ------------------------------------------------------------------------------------------------
  id_4  (= Z(T^2 x [0,T]))       yes            +0.992820-0.119615i          0.000000e+00  AGREE
  Hadamard on the qubit block    no             +0.521099-0.062782i          4.751330e-01  DIFFER
  Haar-random U(4)  (seed 7)     no             -0.291344+0.172837i          1.317044e+00  DIFFER
  Haar-random U(4)  (seed 23)    no             -0.553146+0.040816i          1.554268e+00  DIFFER
  ------------------------------------------------------------------------------------------------
  Z_DW = +0.992820-0.119615i  (the cobordism's metric-free topological value)
```

## The bridge claim

The realizability report (hypothesis (3)) established $Z(W_{AB})=\langle\psi_A|U|\psi_B\rangle$
only in its **spectral / Choi reading**: the synthesized $W_{AB}$ carries
$\mathrm{vec}(U)$ as its boundary-Laplacian eigenvector, so contracting it returns
the amplitude — *partly by construction*. The **strong** form requires the
*independent topological invariant* — the Dijkgraaf–Witten state sum, computed from
$W_{AB}$'s topology and cocycle **alone, no edge weights** — to *also* equal the
amplitude. This experiment closes that gap by computing the cobordism's value three
ways on a shared $\Sigma=T^2$ and testing

$$ Z_{DW}(W_{AB}) \;=\; \langle\psi_A|U|\psi_B\rangle \;=\; Z_{\text{spec}}. $$

The three readings are genuinely independent machinery:

1. **Topological** $Z_{DW}=\langle\psi_A|Z(W)|\psi_B\rangle$ —
   `DijkgraafWitten.amplitude(prepare(psi_A), prepare(psi_B))` on the torus cylinder
   $W=T^2\times[0,T]$. A sum over the flat $\mathbb{Z}_2$ connections of $W$ weighted
   by the cocycle, sandwiched between the **prepared** boundary states
   (`BoundaryStateSpace`/`PreparedBoundaryState`). **Metric-free**: it reads only the
   topology and the cocycle, never an edge weight. $Z(W)=\mathrm{id}$ on
   $Z(T^2)=\mathbb{C}[H^1(T^2;\mathbb{Z}_2)]=\mathbb{C}^4$.
2. **Operation** $\langle\psi_A|U|\psi_B\rangle$ — `ChoiJamiolkowski.transition
   Amplitude`, the Hilbert–Schmidt / Choi map–state duality
   $\langle\psi_A|U|\psi_B\rangle=\langle\mathrm{vec}(U_T)\mid\mathrm{vec}(U)\rangle$.
3. **Spectral** $Z_{\text{spec}}$ — the value read off the $k=1$ boundary harmonic of
   the *synthesized* $W_{AB}$. The solid torus $W=D^2\times S^1$ (boundary $T^2$)
   carries the **longitude** as a genuine $\ker L_1(W)$ harmonic;
   `RealizabilityOracle.decideHarmonic` certifies it ($r\to0$, $\lambda\to0$, witness
   boundary block $\parallel$ target), and $Z_{\text{spec}}$ is the Hodge harmonic
   overlap (`Cochain.innerProduct`) of the two boundary states' coordinates along
   that certified spectral mode in $\ker L_1(\Sigma)$ — the $b_1$-dimensional
   spectral qubit, kept distinct from the $2^{b_1}$-dimensional DW count (spec §5.2).

## The honest finding: the quantized shadow

The $\mathbb{Z}_2$ DW functor's cobordism maps are a **discrete / finite** family —
the $\mathbb{C}[\mathbb{Z}_2]$ Frobenius algebra (cylinder, cap/cup, pair-of-pants,
twisted by the cocycle) — and every one of them is **integer-quantized** in the
flat-connection basis. So a generic operation $U$ is **not** $Z_{DW}$ of any
$\mathbb{Z}_2$ cobordism, and the bridge $Z_{DW}=\langle\psi_A|U|\psi_B\rangle$ can
only hold on that discrete subset. The experiment exhibits both sides of the boundary,
and **the negative case is as important as the positive**:

- **In the DW-representable set** ($U=\mathrm{id}=Z(T^2\times[0,T])$): the three
  readings agree to $\sim10^{-15}$ on every test pair, each cross-checked against an
  independent numpy oracle. The bridge holds.
- **Outside it** (a Hadamard mix on the qubit block, a Haar-random $U(4)$): $Z_{DW}$
  is fixed at the cobordism's value $\langle\psi_A|\psi_B\rangle$ while
  $\langle\psi_A|U|\psi_B\rangle$ moves off it by $0.48$–$1.55$. Over 200 Haar-random
  $U(4)$ the bridge gap has minimum $0.218$ and median $1.03$: a generic $U$ is
  uniformly bounded away from the DW image, not a measure-zero accident.

The DW maps form a discrete integer lattice; the operation amplitudes range over a
continuum. The DW invariant is the **quantized shadow** of the continuous spectral
$Z$ — the two coincide exactly where the operation lands on the lattice, and the
spectral oracle (which realizes the longitude *and a neighborhood of it*, obstructing
only the meridian) **strictly extends** the topological theory.

## Method

The capstone is pure orchestration of the merged, separately-tested classes —
`DijkgraafWitten` (the state sum + `amplitude`/`map`), `BoundaryStateSpace` /
`PreparedBoundaryState` (the $\ker L_1(\Sigma)\to Z(\Sigma)$ preparation),
`ChoiJamiolkowski` (the bend + the amplitude), and `RealizabilityOracle.decide
Harmonic` (the $k=1$ boundary-harmonic synthesis). The recipe:

1. **Prepare.** Build the boundary states from the longitude $\hat\ell\in\ker
   L_1(T^2)$ (the spectrally-realizable mode), $\psi_A=a\,\hat\ell$, $\psi_B=b\,\hat\ell$,
   and `prepare` them into $Z(T^2)=\mathbb{C}^4$ (scatter the harmonic coordinates
   onto the single-generator classes $2^0,2^1$).
2. **Read topologically.** $Z_{DW}=\texttt{amplitude}(\texttt{prepare}(\psi_A),
   \texttt{prepare}(\psi_B))$ on the cylinder — the metric-free state sum.
3. **Read as an operation.** $\langle\psi_A|U|\psi_B\rangle$ via `transitionAmplitude`,
   for $U=\mathrm{id}$ (representable) and for generic $U$ (Hadamard, Haar).
4. **Read spectrally.** Certify with `decideHarmonic` that the solid torus carries
   $\hat\ell$ ($r\to0$, $\lambda\to0$), then $Z_{\text{spec}}=\langle\psi_A,\psi_B
   \rangle$ as the Hodge overlap of the two forms.

Every reading is cross-checked against an **independent numpy computation**, not just
against the others:

- $Z_{DW}$ vs the **GF(2)/holonomy state sum**: an independent numpy reimplementation
  (GF(2) cohomology, holonomy binning, the $\omega$ product) reproduces the C++ map's
  amplitude multiset (convention-free), $\Delta=0$; the contraction then matches.
- $\langle\psi_A|U|\psi_B\rangle$ vs the **direct** $\overline{\psi_A}\,U\,\psi_B$
  (`np.vdot`), $\Delta=0$.
- $Z_{\text{spec}}$ vs the **numpy $\ker L_1(\Sigma)$ projection** (an independent
  eigendecomposition of the symmetric metric Hodge Laplacian), $\Delta\sim10^{-15}$.

## Results

The three readings on longitude-aligned boundary states $\psi_A=a\hat\ell$,
$\psi_B=b\hat\ell$ (the agreement is $\overline a\,b$ to machine precision):

| $(a,b)$ | $Z_{DW}$ (topo) | $\langle\psi_A|\,\mathrm{id}\,|\psi_B\rangle$ (op) | $Z_{\text{spec}}$ (Hodge) | max $\Delta$ |
|---|---|---|---|---|
| $1,\,1$ | $+1.000000+0.000000i$ | $+1.000000+0.000000i$ | $+1.000000+0.000000i$ | $1.1\times10^{-15}$ |
| $e^{i\pi/3},\,0.6{+}0.8i$ | $+0.992820-0.119615i$ | $+0.992820-0.119615i$ | $+0.992820-0.119615i$ | $7.8\times10^{-16}$ |
| $0.5{-}0.5i,\,i$ | $-0.500000+0.500000i$ | $-0.500000+0.500000i$ | $-0.500000+0.500000i$ | $7.9\times10^{-16}$ |
| $2,\,0.25{+}0.97i$ | $+0.500000+1.940000i$ | $+0.500000+1.940000i$ | $+0.500000+1.940000i$ | $1.4\times10^{-15}$ |

The bridge on a representable vs generic $U$ (fixed $\psi_A=e^{i\pi/3}\hat\ell$,
$\psi_B=(0.6{+}0.8i)\hat\ell$, so $Z_{DW}=+0.992820-0.119615i$):

| operation $U$ | representable? | $\langle\psi_A|U|\psi_B\rangle$ | $\Delta=|\,\text{amp}-Z_{DW}|$ | verdict |
|---|---|---|---|---|
| $\mathrm{id}_4\;(=Z(T^2\times[0,T]))$ | **yes** | $+0.992820-0.119615i$ | $0.0$ | **agree** |
| Hadamard on the qubit block | no | $+0.521099-0.062782i$ | $4.75\times10^{-1}$ | **differ** |
| Haar-random $U(4)$ (seed 7) | no | $-0.291344+0.172837i$ | $1.32\times10^{0}$ | **differ** |
| Haar-random $U(4)$ (seed 23) | no | $-0.553146+0.040816i$ | $1.55\times10^{0}$ | **differ** |

The DW-representable maps are integer-quantized: $Z(T^2\times[0,T])=\mathrm{id}_4$ for
both the trivial and sign cocycles, and the solid-torus boundary state
$Z(D^2\times S^1)=[1,0,1,0]$ — all integer. The spectral certification: the longitude
is realized at $r=9.65\times10^{-29}$, $\lambda=-4.25\times10^{-17}$ (a genuine
harmonic) with witness boundary-block overlap $1.000000$; the meridian floors at
$15.2567$, matching the independent numpy Hodge oracle to $\Delta=5.3\times10^{-15}$.

## Key findings

1. **Three independent readings agree on the representable point.** The state sum
   (GF(2) flat connections + cocycle), the Choi/Hilbert–Schmidt amplitude, and the
   Hodge harmonic overlap — entirely different machinery — land on the same complex
   value $\overline a\,b$ to $\sim10^{-15}$ for every test pair. The agreement is a
   genuine cross-validation, not a tautology: $Z_{DW}$ is metric-free, $Z_{\text{spec}}$
   is a metric eigenproblem, and they coincide because the cylinder's $Z(W)=\mathrm{id}$
   and `prepare` is an isometry.
2. **The bridge fails for a generic $U$, by design.** With the cobordism (hence $Z_{DW}$)
   held fixed, $\langle\psi_A|U|\psi_B\rangle$ departs from $Z_{DW}$ as soon as $U$
   leaves the discrete DW point — gap $0.48$ for the Hadamard mix, $1.3$–$1.6$ for
   Haar $U(4)$, minimum $0.218$ over 200 Haar samples. The DW-representable set is a
   discrete island in the continuum of operations.
3. **The DW maps are integer-quantized — the discrete family is literal.** $\mathrm{id}_4$
   (both cocycles) and the solid-torus cap $[1,0,1,0]$ are exact integer objects in the
   flat-connection basis. The $\mathbb{C}[\mathbb{Z}_2]$ Frobenius structure forces
   quantization; a generic complex amplitude cannot be hit.
4. **The spectral oracle is a continuum that strictly contains the DW set.** It realizes
   the longitude ($r\to0$) and — by the realizability report's interior-fill result — a
   continuum of nearby targets, obstructing only the topologically-forced meridian (the
   disk-bounding cycle). So $\{Z(W)\}_{\mathbb{Z}_2\text{-DW}} \subsetneq
   \{\text{spectrally realizable}\}$: the topological invariant is the *quantized shadow*
   of the geometric/spectral $Z$.
5. **The bridge is the strong form of realizability hypothesis (3).** The independent
   topological state sum — no edge weights — equals the operation amplitude on the
   representable subset, lifting the v0.3 spectral/Choi reading (true partly by
   construction) to an honest agreement of two separately-computed invariants.

## Conventions

- Operators are flat **row-major**; $\mathrm{vec}(U)$ is the row-major flatten
  (`ChoiJamiolkowski.vectorize`).
- $Z(\Sigma)=\mathbb{C}[H^1(\Sigma;\mathbb{Z}_2)]$ is the $2^{b_1}$-dimensional
  flat-connection basis; the spectral qubit $\ker L_1(\Sigma)$ is $b_1$-dimensional.
  `prepare` embeds the latter into the former (harmonic $i\to$ class $2^i$), an
  isometry, so $\texttt{amplitude}$ on the cylinder is the harmonic overlap.
- The DW reading is **metric-free** (topology + cocycle only); the spectral reading is
  a **metric** Hodge eigenproblem on the pinned solid torus (the boundary $\partial W$
  byte-fixed, only the interior filled).
- Agreement tolerance $10^{-7}$; a generic $U$ must differ by more than $10^{-2}$.
- Seeded and reproducible; the 10-CPU cap is honored (thread env set at launch).

## Reproduce

```
pip install -e ".[dev]"
python examples/cobordism/dw_spectral_bridge.py            # the tables above
python -m pytest tests/cobordism/test_dw_spectral_bridge_python.py
```

`--seed` (default 0), `--restarts` (default 8), and `--no-plot` are available; the
section 7 sweeps (the bridge gap $U(t)$ from $\mathrm{id}$ to a Haar $U(4)$, and the
spectral oracle's longitude$\to$meridian residual) and the section 8 figures are
written to `/tmp/cobordism/` and are **not committed** — the script is the committed
artifact. Attach them to the issue/PR to pin a result.
