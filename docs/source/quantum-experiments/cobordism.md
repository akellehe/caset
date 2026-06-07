# Testing the State–Operation–Cobordism Correspondence in `tessera`

## 0. Objective

Test, numerically, whether a quantum operation between two states can be realized as a **cobordism** whose boundary is the two states and whose TQFT value is their transition amplitude — with all data (states, operation, signs/phases) determined by a **Hermitian-weighted simplicial complex**. The experiment has two layers: an algebraic layer that is pure linear algebra (validates the correspondence), and a topological layer built on `tessera`'s simplicial machinery (tests whether the sign/phase carries a genuine topological invariant).

This document specifies the **mathematics only**. All data structures, language bindings, parallelism, and module layout are left to the implementer.

---

## 1. Hypothesis under test

Let $\mathrm{geo}(\cdot)$ map a finite-dimensional Hilbert space to a simplicial complex carrying a Hermitian weighted adjacency, and map a state to a (weighted) subcomplex. The claim, for systems $A$ and $B$ with an operation $U:\mathcal{H}_B\to\mathcal{H}_A$:

$$ W_{AB} = \mathrm{geo}(U) \quad\text{is an } n\text{-cobordism}, $$

$$ \partial W_{AB} = \overline{\mathrm{geo}(\psi_B)}\ \sqcup\ \mathrm{geo}(\psi_A), $$

$$ Z(W_{AB}) = \langle \psi_A \,|\, U \,|\, \psi_B \rangle . $$

The manifold is the operation; the amplitude is the number it computes. The engine is map–state duality (Choi–Jamiołkowski / "bending"), under which

$$ \operatorname{rank}(U) \;=\; \text{Schmidt rank of } \operatorname{vec}(U) \;=\; \text{connectivity of } W_{AB}. $$

**Falsifiable core.** The hypothesis is *supported* iff (i) the cylinder cobordism reproduces the inner product, (ii) $Z(W)$ is invariant under interior re-triangulation, and (iii) the nontrivial sign class produces a $Z(W)$ distinct from the trivial one. It is *refuted* if any of these fails.

---

## 2. What to build on in `tessera`

Reuse:

- **Simplicial mesh** (oriented, with per-simplex vertex ordering / branching structure).
- **All four Pachner moves** (add / remove / flip / shift) — the vehicle for the triangulation-independence test; map to the correct bistellar moves for the working dimension.
- **`Cylinder` topology** $\Sigma\times[0,T]$ with open time boundaries — supplies boundary geometry and the **trivial** cobordism (the identity, $Z=\mathrm{id}$); the README notes it exists "for transition amplitudes." It is the $\mathrm{id}$ checkpoint, **not** the cobordism for a nontrivial operation — that is found by synthesis (§5.0).
- **Lorentzian signature** and the CDT causal foliation (parameter $\alpha$ for the time/space edge-length-squared ratio) — supplies the indefinite-signature variant.
- **Spectral tooling** already used for spectral dimension.

New mathematics to add on top (no implementation prescription):

1. A **Hermitian connection** on edges (complex weights with $A=A^\dagger$).
2. **Hodge Laplacian** eigen-decomposition on $k$-cochains, and extraction of harmonic representatives.
3. **Dijkgraaf–Witten** tetrahedron weights from a group cocycle, and the associated state-sum.

---

## 3. Notation

- $V$, $E$ : vertex and edge sets; $N=|V|$ held fixed within a run.
- $C^k$ : complex vector space of $k$-cochains; $d_k:C^k\to C^{k+1}$ the coboundary.
- $A$ : weighted adjacency, $A_{ij}=w_{ij}\,e^{i\theta_{ij}}$, with $w_{ij}=w_{ji}\in\mathbb{R}$ (magnitudes; sign sets signature) and $\theta_{ij}=-\theta_{ji}$ (a $U(1)$ connection). Hermitian: $A=A^\dagger$.
- $D$ : diagonal degree matrix. Use the **magnitude convention** $D_{ii}=\sum_j |A_{ij}|=\sum_j w_{ij}$ so that $L$ is Hermitian.
- $L=D-A$ : graph (0-)Laplacian; $L_k$ : Hodge Laplacian on $k$-cochains, $L_k = d_{k-1}d_{k-1}^\dagger + d_k^\dagger d_k$.
- $b_k=\dim\ker L_k$ : $k$-th Betti number (harmonic-cochain dimension).
- $\Phi_\gamma=\sum_{(ij)\in\gamma}\theta_{ij}\ (\mathrm{mod}\ 2\pi)$ : holonomy (flux) around cycle $\gamma$.
- $G=\operatorname{diag}(e^{i\alpha_v})$ : a vertex-phase gauge transformation, acting by $A\mapsto GAG^\dagger$.
- $Z$ : the TQFT functor; $Z(W)$ the linear map / amplitude assigned to cobordism $W$.

Two-qubit computational basis is written $\{00,01,10,11\}$, identifying $\mathbb{C}^4\cong\mathcal{H}_A\otimes\mathcal{H}_B$.

---

## 4. Stage 1 — Algebraic layer ($n=1$), pure linear algebra

States live on vertices; the cobordism is an arc. This stage is fast, exact, and validates the correspondence and the gauge structure before any mesh is built.

### 4.1 Hermitian-weighted complex

Construct a Hermitian-weighted graph on $N$ vertices: choose magnitudes $w_{ij}=w_{ji}$ (allow negative for indefinite signature) and phases $\theta_{ij}=-\theta_{ji}$. Form $A$, then $D$ (magnitude convention), then $L=D-A$. Confirm $L=L^\dagger$ and that $e^{-iLt}$ is unitary.

### 4.2 States as harmonic eigenvectors

Eigendecompose $L=\sum_k \lambda_k\, v_k v_k^\dagger$, $\lambda_k\in\mathbb{R}$. A state is a chosen unit eigenvector $\psi=v_k$. Its geometric image $\mathrm{geo}(\psi)$ is the **weighted support**: the subcomplex of simplices on which $\psi$ is nonzero, decorated with the amplitudes. For $\mathrm{geo}(\psi)$ to be admissible as a closed boundary, require $\psi$ harmonic (a cycle): $\psi\in\ker L_{n-1}$.

### 4.3 The operation and its bending

The operation between $A$ and $B$ is the **entangling cross-cut coupling** — the both-bits-flip link $00\!\leftrightarrow\!11$, i.e. the off-diagonal block $\gamma_{AB}$ (the Weyl off-diagonal of $\gamma^0$) rendered geometrically. Compare it against the **transition operator** $U_T=|\psi_A\rangle\langle\psi_B|$.

Bend (vectorize):

$$ \operatorname{vec}(U) \;=\; \sum_{i,j} U_{ij}\,|i\rangle_A\otimes|j\rangle_B \;\in\; \mathcal{H}_A\otimes\mathcal{H}_B . $$

Compute the Schmidt rank of $\operatorname{vec}(U)$ by SVD of the matrix $U$.

### 4.4 The cobordism reading

$W_{AB}$ is the support graph of $U$; its boundary components are $\mathrm{geo}(\psi_A)$ and $\mathrm{geo}(\psi_B)$. A rank-1 $U$ factors through the unit object,

$$ \mathcal{H}_B \xrightarrow{\ \langle\psi_B|\ } \mathbb{C} \xrightarrow{\ |\psi_A\rangle\ } \mathcal{H}_A , $$

so its cobordism is **disconnected** (a separable bent state); full-rank $U$ gives a **connected** cobordism (an entangled bent state, e.g. the cup $|00\rangle+|11\rangle$).

### 4.5 Checks

- **C1 (value = amplitude).** Verify numerically $\langle\psi_A|U|\psi_B\rangle$ equals the contraction of $\operatorname{vec}(U)$ with the bent boundary states.
- **C2 (rank = Schmidt = connectivity).** Verify $\operatorname{rank}(U)$ equals the Schmidt rank of $\operatorname{vec}(U)$; confirm rank $1\Rightarrow$ separable/disconnected, rank $\ge 2\Rightarrow$ entangled/connected. Confirm $U_T$ is rank 1 and $\gamma_{AB}$ is full rank.
- **C3 (gauge invariance).** Apply random $G=\operatorname{diag}(e^{i\alpha_v})$. Verify $\operatorname{spec}(L)$ is unchanged, eigenvectors are rephased $v_k\mapsto Gv_k$, and every cycle flux $\Phi_\gamma$ is unchanged.
- **C4 (flux lives in the spectrum).** On a bigon — two parallel weighted edges between the $00$ and $11$ vertices, $b_1=1$ — the effective coupling is $z=e^{i\theta_1}+e^{i\theta_2}=2\cos(\Phi/2)\,e^{i\bar\theta}$ with $\Phi=\theta_1-\theta_2$. Verify the eigenvalue gap scales as $|z|=2|\cos(\Phi/2)|$ (a gauge-invariant interference signature) while the Bell relative phase $\bar\theta$ is gauge-dependent.
- **C5 (tree vs cycle).** On a tree ($b_1=0$) verify the spectrum is independent of all $\theta$ (phases are pure gauge); on a graph with $b_1\ge1$ verify residual $\theta$-dependence through $\Phi_\gamma$.

A representative cyclic testbed honoring both requirements: the square $00\text{–}01\text{–}11\text{–}10\text{–}00$ plus the diagonal $00\text{–}11$ ($|E|=5,\ b_1=2$); the diagonal is the entangling coupling, the cycles carry the flux.

---

## 4b. Boundary-state synthesis (inverse eigenvector problem)

This builds $\mathrm{geo}(\psi_A)$ and $\mathrm{geo}(\psi_B)$ **independently** by solving the inverse problem: given a target state, find the *simplest* complex whose Laplacian has it as an eigenvector. Written here for the 0-Laplacian $L=D-A$ (vertex states); the same procedure runs at degree $k$ by replacing vertices with $k$-simplices and $L$ with $L_k$.

### 4b.1 Target and embedding

Draw a random unit target $\psi$. To keep it a **qubit** on $N\ge 2$ vertices, designate two **logical** vertices carrying the amplitudes $(c_0,c_1)$ and let the remaining **auxiliary** vertices carry zero amplitude — they exist only to supply combinatorial freedom:

$$ \psi = (c_0,\ c_1,\ 0,\ \dots,\ 0),\qquad \|\psi\|=1 . $$

(Alternatively treat the full $\mathbb{C}^N$ vector as an $N$-level state and drop "qubit." Decide which.)

### 4b.2 Why auxiliary vertices are necessary

Under the pure-edge constraint ($A_{ii}=0$, no on-site potentials) with the magnitude degree convention, a single edge between the two logical vertices has Laplacian

$$ L=\begin{pmatrix} w & -w e^{i\theta}\\ -w e^{-i\theta} & w\end{pmatrix}, $$

whose eigenvectors are $\tfrac{1}{\sqrt2}(e^{i\theta},\pm 1)$ — **balanced only** ($|c_0|=|c_1|$). A general-amplitude qubit ($|c_0|\ne|c_1|$) therefore cannot be a two-vertex eigenvector; it requires auxiliary scaffolding. The minimal number of auxiliary vertices is the state's **combinatorial complexity** — the quantity this stage measures.

### 4b.3 Objective

Solve for Hermitian edge weights on the current complex minimizing the eigenvalue-agnostic residual

$$ r(A) = \big\|\,(I-\psi\psi^\dagger)\,L\,\psi\,\big\|^2 , $$

with $A_{ii}=0$ and $D_{ii}=\sum_j w_{ij}$. Then $\psi$ is an eigenvector iff $r=0$ (i.e. $L\psi\parallel\psi$); the realized eigenvalue is the Rayleigh quotient $\lambda=\psi^\dagger L\psi$. The landscape is non-convex in $\{w_{ij},\theta_{ij}\}$; use multiple restarts.

### 4b.4 Seed and the coning loop

- **Seed.** Start each state on a single 4-simplex $\Delta^4$ — 5 vertices, 1-skeleton $K_5$.
- **Cone-and-retry.** If no restart drives $r<\epsilon$ within budget, **cone in** one vertex and re-optimize. Use the topology-preserving stellar / Pachner-add operation (`tessera`'s `Spacetime` and `Simplex` expose coning methods): it adds an apex and its incident edges, enlarging the parameter space **without changing the homotopy type**. Add one vertex at a time.
- **Stop / minimality.** Accept the first complex reaching $r<\epsilon$; record $(|V|,|E|)$ as the state's complexity. This is the simplest complex representing $\psi$.

### 4b.5 Notes

- **Coning vs the cone.** The growth move must be the topology-preserving subdivision (Pachner add), **not** the full cone $CX$ — a cone is contractible and would trivialize all harmonic/topological content. Confirm `tessera`'s coning primitive is the subdivision.
- **Degree convention.** Keep the magnitude convention $D_{ii}=\sum_j|A_{ij}|$ (Hermitian $L$, unitary evolution). The signed convention restores the constant zero mode but breaks Hermiticity for complex weights — do not mix them.
- **Feasibility.** On an unrestricted complete graph, parameters far outnumber constraints and a solution is generic; the search is nontrivial only because the complex must be a valid simplicial complex. Coning is the canonical simplicial way to add freedom, so the loop terminates.
- **Output per state.** $\mathrm{geo}(\psi_A)$ and $\mathrm{geo}(\psi_B)$: their minimal complexes, edge weights, and realized $\lambda$. These are the boundary objects consumed by Stage 2.

---

## 5. Stage 2 — Topological layer ($n=3$), simplicial via `tessera`

Now states live on closed 2-surfaces and the cobordism is a 3-manifold. This is where the sign becomes a topological invariant.

### 5.0 Synthesis is the task

Given the operation $U:\mathcal{H}_B\to\mathcal{H}_A$ (e.g. the entangling coupling $\gamma_{AB}$ relating the synthesized boundaries), **find a bulk** 3-manifold $W_{AB}$ realizing it — do not assume one. Bend $U$ to a boundary *state* via Choi–Jamiołkowski ($\mathrm{vec}(U)$, the operator-as-state), then **synthesize the bulk spectrally**: holding the synthesized boundaries $\mathrm{geo}(\psi_A)$, $\mathrm{geo}(\psi_B)$ and the output surface *fixed*, fill the interior of $W_{AB}$ — its topology and Hermitian edge weights — so the final state's graph-Laplacian eigenvector(s) match the bent target (drive the §4b residual $r=\lVert(I-\psi\psi^\dagger)L\psi\rVert^2\to 0$). The operation lives in the bulk, not in the cylinder.

**Realizability is itself the test.** The boundaries are *pinned* (the synthesized $\mathrm{geo}$'s), so this is **not** the free coning of §4b — where parameters outrun constraints and any single eigenvector is eventually realizable — but an *interior fill with fixed ends*. A target is **realizable** iff the residual can be driven to zero, and **unrealizable** iff it floors away from zero: a spectral/topological obstruction under the fixed-boundary constraint (the analogue of §4b's two-vertex floor $w_{\min}^2(|c_0|^2-|c_1|^2)^2$). Non-existence is thus certified by an *obstruction* (a residual floor), **not** by exhausting triangulations. Whether such obstructions exist — which operations $U$ have no bulk — is the sharpest form of the hypothesis. (An earlier draft framed this as TQFT-membership $Z(W)=U$ generated by the modular $S,T$; that route is set aside — the Dijkgraaf–Witten state-sum of §5.3–5.5 remains the topological invariant tested by T1–T5, while realizability is decided spectrally as above.)

### 5.1 Cobordism and boundary surfaces

A candidate $W$ is an oriented triangulated 3-manifold with

$$ \partial W = \overline{\Sigma_B}\ \sqcup\ \Sigma_A , $$

two closed oriented surfaces. The `Cylinder` topology $\Sigma\times[0,T]$ gives the **trivial** cobordism ($Z=\mathrm{id}$), used only as boundary geometry and the T1 checkpoint; nontrivial candidates come from gluings and other topologies ($S^2\times S^1$, $T^3$, lens spaces, mapping tori).

### 5.2 States as harmonic 1-forms (qubits from $H_1$)

Take $\Sigma$ a genus-$g$ surface. The real harmonic 1-cochains form $\ker L_1(\Sigma)$ of dimension $b_1=2g$. For the torus ($g=1$), $\dim\ker L_1=2$ — a **qubit**. Choose $\psi_A\in\ker L_1(\Sigma_A)$, $\psi_B\in\ker L_1(\Sigma_B)$ as the boundary states. (The discrete-gauge count of flat $\mathbb{Z}_2$ connections is $2^{\,b_1}$; keep the two countings distinct — continuous harmonic forms give the qubit dimension, flat connections index the state-sum.)

### 5.3 The $\mathbb{Z}_2$ connection and Dijkgraaf–Witten weight

Assign a $\mathbb{Z}_2$ gauge field $g\in C^1(W;\mathbb{Z}_2)$, $g_e\in\{0,1\}$, flat on every 2-simplex:

$$ (d g)\big|_{\triangle} = g_{01}+g_{12}-g_{02} \equiv 0 \pmod 2 . $$

Weight each oriented tetrahedron $[v_0v_1v_2v_3]$ by a 3-cocycle $\omega\in Z^3(\mathbb{Z}_2;U(1))$ raised to its orientation sign $\epsilon_t=\pm1$:

$$ \omega\big(g_{01},g_{12},g_{23}\big)^{\epsilon_t}. $$

Use the two classes: trivial $\omega\equiv 1$, and the nontrivial generator

$$ \omega(a,b,c) = (-1)^{abc} \in H^3(\mathbb{Z}_2;U(1)) \cong \mathbb{Z}_2 . $$

### 5.4 State-sum and cobordism map

For a closed complex,

$$ Z(W) = \frac{1}{|\mathbb{Z}_2|^{|V|}} \sum_{\text{flat } g}\ \prod_{\text{tetrahedra } t} \omega(t)^{\epsilon_t}. $$

For $W$ with boundary, hold the boundary field $g|_{\partial W}$ fixed; summing over interior fields yields a vector indexed by boundary flat connections, i.e. an element of $Z(\partial W)$. Reading it as a map gives

$$ Z(W) : Z(\Sigma_B)\to Z(\Sigma_A), \qquad \dim Z(\Sigma_g)=2^{\,b_1(\Sigma_g)} . $$

The amplitude under test is $\langle\psi_A\,|\,Z(W)\,|\,\psi_B\rangle$ for the chosen boundary states.

### 5.5 Checks

- **T1 (cylinder = identity).** For $W=\Sigma\times[0,T]$ verify $Z(W)=\mathrm{id}_{Z(\Sigma)}$, hence $\langle\psi_A|Z(W)|\psi_B\rangle=\langle\psi_A|\psi_B\rangle$.
- **T2 (triangulation independence — the central test).** Apply interior bistellar (Pachner) moves with $\partial W$ fixed; verify $Z(W)$ is invariant to machine precision. Failure here refutes the construction (or signals $\omega$ is not a cocycle).
- **T3 (the sign carries an invariant).** On a 3-manifold with nontrivial topology, verify $Z_{\omega=(-1)^{abc}}(W)\ne Z_{\omega\equiv1}(W)$. Equality everywhere would refute "the sign carries an invariant" at $n=3$.
- **T4 (cross-layer consistency).** Verify the bulk $\mathbb{Z}_2$ holonomies equal the Stage-1 cycle fluxes restricted to $\{0,\pi\}$.
- **T5 (composition / functoriality).** Glue $W_1:\Sigma_A\to\Sigma_C$ and $W_2:\Sigma_C\to\Sigma_B$ along $\Sigma_C$; verify $Z(W_2\cup_{\Sigma_C} W_1)=Z(W_2)\,Z(W_1)$.

### 5.6 Lorentzian variant

Repeat the Hodge analysis in CDT's Lorentzian signature (parameter $\alpha$). The metric is indefinite, so $L_k$ becomes a discrete d'Alembertian and the clean identification $\ker L_k\cong H_k$ degrades to a pseudo-Hodge decomposition. Record where harmonic representatives become null and how the invariant (T2, T3) behaves relative to the Euclidean run. This is the signature in which $\gamma^0$ natively lives.

---

## 6. Falsifiable predictions

- **P1.** Cylinder reproduces the inner product (T1, C1).
- **P2.** $Z(W)$ invariant under interior Pachner moves (T2) — make-or-break.
- **P3.** Nontrivial $\omega$ distinguishable from trivial on some $W$ (T3).
- **P4.** $\operatorname{rank}(U)=$ Schmidt rank $=$ connectivity; transition operator separable, coupling entangling (C2).
- **P5.** Flux is gauge-invariant and visible in the spectrum; only cycles ($b_1\ge1$) carry physical phase (C3, C4, C5).

Hypothesis **supported** iff P1, P2, P3 hold; **refuted** if any of P1–P3 fails. P4, P5 are structural consistency conditions that should hold in every run.

---

## 7. Parameter sweeps

- Flux $\Phi\in[0,2\pi)$: spectrum and amplitude versus flux (Stage 1).
- Signature: Euclidean versus Lorentzian (CDT $\alpha$).
- Boundary genus $g$: qubit dimension $b_1=2g$; state-sum dimension $2^{2g}$.
- Cocycle class: trivial versus nontrivial $\mathbb{Z}_2$.
- Bulk refinement / Pachner depth: invariance stress test (T2).
- 3-manifold topology: $\Sigma\times S^1$, $T^3$, lens spaces.

---

## 8. Outputs

- A pass/fail report on P1–P5 / C1–C5 / T1–T5, with residuals.
- Tables of $Z(W)$ across triangulations at fixed boundary (demonstrating, or breaking, invariance).
- Side-by-side $Z_{\text{nontrivial}}$ versus $Z_{\text{trivial}}$.
- Spectrum-versus-flux and amplitude-versus-flux curves.
- Cobordism mesh exports (the existing GIF / GraphML / DOT writers) for the worked $W$.

---

### Notes for the implementer

- Stages are independent: Stage 1 needs no mesh and should run first as a correctness oracle for the bending/amplitude relations and the gauge structure.
- The only nonstandard numerical objects are: the magnitude-convention Hermitian Laplacian, harmonic-cochain extraction (kernel of $L_k$), and the cocycle-weighted state-sum. Everything else is existing `tessera` machinery.
- Keep the two notions of "dimension" separate throughout: the manifold dimension $n$ (here $1$ then $3$) versus the Hilbert-space dimension (here $2$ per qubit, $2^{b_1}$ for a boundary surface).
