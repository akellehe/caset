# Charged Cartan — when does $H_{\mathrm{pair}}$ actually engage? A first-cell derivation

> **Status**: design exploration
> **Motivation**: [Experiment 05](../experiments/05-v0.2-h-param-sweep.md)
> found that $J_s = 0$ gives peak $D_S \approx 4.08$, indistinguishable
> from the v0.2 default's $T \to \infty$ asymptote. Tracing through the
> code revealed the initial states are charge eigenstates within their
> sector, which makes $\hat{Q} \otimes \hat{Q}$ act as a global phase
> and means the dynamics may be degenerate at $J_s = 0$. This document
> works the first interaction cell by hand to characterise *when*
> $H_{\mathrm{pair}}$ has dynamical content vs. when it doesn't, and
> proposes physically meaningful manipulations to increase
> "engagement."

## 1. Setup

Each vertex carries a four-dimensional [qudit](../../overview/intellectual_lineage.md#5-the-qudit-basis-charge-intrinsic-to-the-state)
state $\rho_v \in \mathbb{C}^{4 \times 4}$ on the basis

$$
\mathcal{B} \;=\; \bigl\{\ket{+\,0},\;\ket{+\,1},\;\ket{-\,0},\;\ket{-\,1}\bigr\}
\;\equiv\;\{\ket{e_0},\ket{e_1},\ket{e_2},\ket{e_3}\}.
$$

The first index is the **charge** sign (sector label, $\pm$); the
second is an internal **spin-like** index ($0/1$). We treat them as
two qubits, with the convention that the *charge* qubit is the outer
slot in tensor products and the *spin* qubit is the inner slot.
Compactly, $\mathbb{C}^4 = \mathbb{C}^2_q \otimes \mathbb{C}^2_s$.

### 1.1 Single-vertex operators

The charge operator is diagonal in $\mathcal{B}$:

$$
\hat{Q} \;=\; \mathrm{diag}(+1, +1, -1, -1) \;=\; \sigma^z_q \otimes I_s.
$$

The spin operators are identity on the charge sector and Pauli on the
spin sector:

$$
\sigma^x \;=\; I_q \otimes \sigma^x_{\text{Pauli}}, \qquad
\sigma^y \;=\; I_q \otimes \sigma^y_{\text{Pauli}}.
$$

In $\mathcal{B}$ this means $\sigma^x$ swaps $\ket{+\,0}\leftrightarrow\ket{+\,1}$ and
$\ket{-\,0}\leftrightarrow\ket{-\,1}$ (and similarly for $\sigma^y$ with
$\pm i$ phases). The CP-violation generator is the charge-flip:

$$
\hat{C} \;=\; \sigma^x_q \otimes I_s,
$$

which swaps $\ket{+\,s}\leftrightarrow\ket{-\,s}$. (This is the only
operator in $H_{\mathrm{pair}}$ that does not commute with $\hat{Q}$.)

### 1.2 The pair Hamiltonian

For two vertices $A, B$ (sites in $\mathbb{C}^4 \otimes \mathbb{C}^4 = \mathbb{C}^{16}$):

$$
\boxed{\;
\hat{H}_{\mathrm{pair}} \;=\;
\underbrace{J_c\,(\hat{Q}_A \otimes \hat{Q}_B)}_{\text{charge--charge}} \;+\;
\underbrace{J_s\,(\sigma^x_A \sigma^x_B + \sigma^y_A \sigma^y_B)}_{\text{spin--spin } (XX+YY)} \;+\;
\underbrace{\delta_m\,(\hat{Q}_A \otimes I + I \otimes \hat{Q}_B)}_{\text{mass shift}} \;+\;
\underbrace{\gamma_{\mathrm{CP}}\,(\hat{C}_A \otimes I + I \otimes \hat{C}_B)}_{\text{CP-violation}}\;}
$$

The per-event unitary is $\hat{U} = \exp(-i\,\hat{H}_{\mathrm{pair}}\,\Delta t)$.

We work at the **v0.2 defaults**:

| | value |
|---|---:|
| $J_c$ | $1.0$ |
| $J_s$ | $0.25$ |
| $\delta_m$ | $0.0$ |
| $\gamma_{\mathrm{CP}}$ | $0.0$ |
| $\Delta t$ | $0.25$ |

So $\delta_m = \gamma_{\mathrm{CP}} = 0$ and

$$
\hat{H}_{\mathrm{pair}}\big|_{\text{v0.2}} \;=\; J_c \,\hat{Q}\otimes\hat{Q} \;+\; J_s\,(\sigma^x\sigma^x + \sigma^y\sigma^y).
$$

### 1.3 Conservation and block structure

At $\gamma_{\mathrm{CP}} = 0$, every term in $H_{\mathrm{pair}}$ commutes
with the total-charge operator $\hat{Q}_{\mathrm{tot}} = \hat{Q}_A + \hat{Q}_B$.
So $\hat{U}$ is block-diagonal in the $\hat{Q}_{\mathrm{tot}}$ eigenspaces:

| $Q_{\mathrm{tot}}$ | sub-block | dim |
|---:|---|---:|
| $+2$ | $(+,+)$ | $4$ |
| $0$  | $(+,-) \oplus (-,+)$ | $8$ |
| $-2$ | $(-,-)$ | $4$ |

In each sub-block, $\hat{Q}\otimes\hat{Q}$ acts as the constant
$q_A q_B \in \{+1, -1\}$ — i.e., $\pm I_4$ on the sub-block. So inside
a fixed-charge-pair sub-block, the only operator with non-trivial
matrix structure is $\sigma^x\sigma^x + \sigma^y\sigma^y$, which lives
entirely in the spin sector.

### 1.4 $XX + YY$ in the spin sector

Acting on the $\{\ket{00}, \ket{01}, \ket{10}, \ket{11}\}$ spin-pair basis:

$$
\sigma^x \otimes \sigma^x \;=\; \begin{pmatrix} 0 & 0 & 0 & 1\\0 & 0 & 1 & 0\\0 & 1 & 0 & 0\\1 & 0 & 0 & 0\end{pmatrix},
\qquad
\sigma^y \otimes \sigma^y \;=\; \begin{pmatrix} 0 & 0 & 0 & -1\\0 & 0 & 1 & 0\\0 & 1 & 0 & 0\\-1 & 0 & 0 & 0\end{pmatrix},
$$

so

$$
\sigma^x\sigma^x + \sigma^y\sigma^y \;=\; \begin{pmatrix} 0 & 0 & 0 & 0\\0 & 0 & 2 & 0\\0 & 2 & 0 & 0\\0 & 0 & 0 & 0\end{pmatrix}.
$$

That is: $(XX+YY)$ **annihilates** $\ket{00}$ and $\ket{11}$, and acts
as $2\,\sigma^x$ on the $\{\ket{01}, \ket{10}\}$ subspace.

### 1.5 The unitary on a fixed-charge-pair sub-block

Take $A = +$, $B = -$ (the relevant block for an alternating-charge
initial layer). On the 8-dimensional $(+,-)$ block:

$$
\hat{H}_{\mathrm{pair}}\big|_{(+,-)} \;=\; -J_c\,I_8 \;+\; J_s\,(\sigma^x\sigma^x + \sigma^y\sigma^y).
$$

Decomposing further into the spin sub-blocks:

- On $\ket{+\,0,\,-\,0}$ and $\ket{+\,1,\,-\,1}$: $H = -J_c$ (eigenvalue), so $U = e^{+i J_c \Delta t}$.
- On $\{\ket{+\,0,\,-\,1},\,\ket{+\,1,\,-\,0}\}$: $H = -J_c\,I + 2 J_s\,\sigma^x$,
  with eigenvalues $-J_c \pm 2 J_s$ and eigenvectors
  $(\ket{+\,0,\,-\,1} \pm \ket{+\,1,\,-\,0})/\sqrt{2}$.

So inside this 2-dim spin sub-block,

$$
\hat{U} \;=\; e^{+i J_c \Delta t}\bigl[\cos(2 J_s \Delta t)\, I \;-\; i\sin(2 J_s \Delta t)\,\sigma^x\bigr].
$$

At v0.2 defaults: $2 J_s \Delta t = 0.125\,\mathrm{rad}$, so
$\cos(2 J_s \Delta t) \approx 0.99219$, $\sin(2 J_s \Delta t) \approx 0.12467$.

That's our complete picture of $\hat{U}$ on the $(+,-)$ block:

$$
\boxed{\quad
\begin{aligned}
\hat{U}\ket{+\,0,\,-\,0} &\;=\; e^{+i J_c \Delta t}\,\ket{+\,0,\,-\,0}\\
\hat{U}\ket{+\,1,\,-\,1} &\;=\; e^{+i J_c \Delta t}\,\ket{+\,1,\,-\,1}\\
\hat{U}\ket{+\,0,\,-\,1} &\;=\; e^{+i J_c \Delta t}\bigl[\,c\,\ket{+\,0,\,-\,1} \;-\; i s\,\ket{+\,1,\,-\,0}\,\bigr]\\
\hat{U}\ket{+\,1,\,-\,0} &\;=\; e^{+i J_c \Delta t}\bigl[\,c\,\ket{+\,1,\,-\,0} \;-\; i s\,\ket{+\,0,\,-\,1}\,\bigr]
\end{aligned}\quad}
$$

with $c = \cos(2J_s\Delta t)$, $s = \sin(2J_s\Delta t)$. The
$(-,+)$ block is identical under $A \leftrightarrow B$.

## 2. The first interaction cell

### 2.1 Generation 0 → generation 1

Two initial-layer vertices $X, Y$ (the *0-simplices* of generation 0)
get picked from the frontier and interact via $\hat U$. The
construction produces three new generation-1 vertices — $X', Y'$
(the worldline continuations of $X, Y$) and $AB$ (the entangling-
core $\Sigma_{AB}$ that carries the Choi state of $\hat U$). The
five vertices $\{X, Y, X', AB, Y'\}$ together form a **4-simplex**
(a $(2,3)$-Pachner cell) with $\binom{5}{2} = 10$ edges:

![Two gen-0 vertices interact via $\hat U$ to produce three gen-1
vertices; the five vertices form a 4-simplex with 10 labeled
edges](../../figures/h_pair_first_cell_diagram.svg)

The 10 edges fall into three classes by MI semantics:

- **2 worldline self-info edges** (green): $X{-}X'$ carries
  $S_X = S(\rho_X)$, $Y{-}Y'$ carries $S_Y$. These connect each
  input to its own product.
- **4 hub-spoke edges** (red): each connects a corner to the
  $\Sigma_{AB}$ vertex. All four carry the post-event MI
  $I_{\mathrm{joint}} = I(X' : Y')$ (the "joint" because it
  measures how much $\hat U$ correlated the products).
- **4 pair-MI edges** (blue): the input edge $X{-}Y$ and three
  more $X'{-}Y'$, $X{-}Y'$, $Y{-}X'$. All four carry MI = either
  $I_{\mathrm{input}} = I(X{:}Y)$ (input joint) or
  $I_{\mathrm{joint}} = I(X'{:}Y')$ (post-event joint).

The causal/temporal structure is cleanest in the auxiliary
schematic:

![Discrete event step $n=0 \to n=1$: input $\rho_X \otimes \rho_Y$
goes through $\hat U$ producing $\rho_{X'}, \rho_{Y'}$, and the Choi
state $\rho_{AB}$](../../figures/h_pair_first_cell_schematic.svg)

### 2.2 The three product states

The new vertices' density matrices come from the post-event joint:

- $\rho_{X'} = \mathrm{Tr}_Y\bigl(\hat{U}(\rho_X \otimes \rho_Y)\hat{U}^\dagger\bigr)$
- $\rho_{Y'} = \mathrm{Tr}_X\bigl(\hat{U}(\rho_X \otimes \rho_Y)\hat{U}^\dagger\bigr)$
- $\rho_{AB} =$ the $\Sigma_{AB}$ Choi-state vertex carrying $J(\hat{U})$ (post-#16)

Each edge gets a mutual information $I_e$, which becomes an edge
length $\ell_e = -\ln(I_e / I_{\max})$ (with $I_{\max} = 2 \ln 2$)
and squared length
$\ell_e^2 = \mathrm{signedSquaredLength}(\ell_e, \text{spacelike})$.

### 2.3 The 10 MI assignments

From `src/quantum/interaction_simulation.cpp` (lines 786–795):

| edge | endpoints | MI value | meaning |
|---|---|---|---|
| 1 | $X$–$Y$ | $I_{\mathrm{input}}$ | input spatial |
| 2 | $X$–$X'$ | $S_X$ | worldline self-info |
| 3 | $Y$–$Y'$ | $S_Y$ | worldline self-info |
| 4 | $X$–$AB$ | $I_{\mathrm{joint}}$ | hub spoke |
| 5 | $Y$–$AB$ | $I_{\mathrm{joint}}$ | hub spoke |
| 6 | $X'$–$AB$ | $I_{\mathrm{joint}}$ | hub spoke |
| 7 | $AB$–$Y'$ | $I_{\mathrm{joint}}$ | hub spoke |
| 8 | $X'$–$Y'$ | $I_{\mathrm{input}}$ | cross-pair |
| 9 | $X$–$Y'$ | $I_{\mathrm{input}}$ | cross-pair |
| 10 | $Y$–$X'$ | $I_{\mathrm{input}}$ | cross-pair |

where

- $S_X = S(\rho_X)$, $S_Y = S(\rho_Y)$ — input marginal entropies
- $I_{\mathrm{input}} = I(X : Y) = S_X + S_Y - S(\rho_{XY})$
- $I_{\mathrm{joint}} = I(X' : Y')$ in the **post-** $\hat{U}$ joint $\rho_{AB} = \hat{U}\rho_{XY}\hat{U}^\dagger$

By unitary invariance of total entropy, $S(\rho_{AB}) = S(\rho_{XY})$,
so the question of $I_{\mathrm{joint}}$ is entirely about how
$\hat{U}$ rearranges marginal entropies between $X$ and $Y$.

## 3. Case A — Bell-like input (concentrated information)

Take a charge-Bell input

$$
\ket{\psi_{XY}^{(A)}} \;=\; \tfrac{1}{\sqrt 2}\bigl(\ket{+\,0,\,-\,0} \;+\; \ket{-\,0,\,+\,0}\bigr).
$$

### 3.1 Marginals and entropies

$\rho_{XY} = \ket{\psi}\bra{\psi}$, pure, so $S(\rho_{XY}) = 0$. Trace
out $Y$:

$$
\rho_X^{(A)} \;=\; \tfrac{1}{2}\ket{+\,0}\bra{+\,0} \;+\; \tfrac{1}{2}\ket{-\,0}\bra{-\,0}.
$$

Off-diagonals would require $\ket{+\,0, y}\bra{-\,0, y}$ pairs with
matching $y$ — they're absent because in $\ket{\psi}$ the $A$-side
$\ket{+\,0}$ correlates with $B$-side $\ket{-\,0}$ and vice versa.

$S(\rho_X^{(A)}) = \ln 2$ (one-bit charge entropy). Similarly
$S(\rho_Y^{(A)}) = \ln 2$.

$$
I^{(A)}_{\mathrm{input}} \;=\; S_X + S_Y - S(\rho_{XY}) \;=\; \ln 2 + \ln 2 - 0 \;=\; 2\,\ln 2.
$$

So the initial MI equals the maximum a 2-dim charge-bit can carry —
$\ket{\psi_{XY}^{(A)}}$ is **maximally entangled in the charge
register** and **trivial in the spin register** ($s = 0$ frozen).

### 3.2 Action of $\hat U$

The two basis states $\ket{+\,0,\,-\,0}$ and $\ket{-\,0,\,+\,0}$ live in
the $(+,-)$ and $(-,+)$ blocks respectively. From the boxed expression
above, both get the same global phase $e^{+i J_c \Delta t}$:

$$
\hat U\,\ket{\psi^{(A)}_{XY}} \;=\; e^{+i J_c \Delta t}\,\ket{\psi^{(A)}_{XY}}.
$$

The $(XX+YY)$ term contributes **nothing** because $\ket{\psi^{(A)}}$
sits entirely on the $\ket{00}$ spin sector, which is annihilated by
$XX+YY$.

So $\rho_{AB} = \rho_{XY}$ (the global phase cancels in $\rho$), and:

$$
I^{(A)}_{\mathrm{joint}} \;=\; I^{(A)}_{\mathrm{input}} \;=\; 2\,\ln 2.
$$

### 3.3 Per-edge MI

| edge | $I_e$ | $I_e / I_{\max}$ | $\ell_e = -\ln(I_e/I_{\max})$ |
|---|---:|---:|---:|
| $X$–$Y$ | $2\ln 2$ | $1.0$ | $0$ |
| $X$–$X'$ | $\ln 2$ | $0.5$ | $\ln 2 \approx 0.693$ |
| $Y$–$Y'$ | $\ln 2$ | $0.5$ | $\ln 2$ |
| 4 hub spokes | $2\ln 2$ each | $1.0$ | $0$ |
| 3 cross-pair | $2\ln 2$ each | $1.0$ | $0$ |

(Recall $I_{\max} = 2\ln 2$.)

So **8 of the 10 edges have length 0** (saturated at $I_{\max}$); the
other 2 are the worldline-self edges at $\ell = \ln 2$. The new cell
has a near-degenerate geometry — most edges are "as short as
possible" — and the Regge action $\Delta S$ is determined almost
entirely by the two worldline edges.

### 3.4 What happened dynamically

**Nothing.** $\hat U$ acted as the identity (up to global phase) on
this input. The cell has rich MI structure, but all of it was
**already present in the input**. The Hamiltonian did not generate
any of it.

This is the **eigenstate trap**: $\ket{\psi^{(A)}}$ is a simultaneous
eigenstate of $\hat Q\otimes\hat Q$ (eigenvalue $-1$, in both blocks)
and of $XX+YY$ (eigenvalue $0$), hence of $\hat H_{\mathrm{pair}}$
itself. Eigenstates pick up phases under $\hat U$; their MI
structure is rigid.

## 4. Case B — distributed mixed input (maximally distributed information)

Now take

$$
\rho_X^{(B)} \;=\; \tfrac{1}{2}\bigl(\ket{+\,0}\bra{+\,0} + \ket{+\,1}\bra{+\,1}\bigr) \;=\; \tfrac{1}{2}\,P_+,
$$

i.e., the maximally mixed state on the $+$ charge sector ($P_+$ is the
sector projector). Similarly $\rho_Y^{(B)} = \tfrac{1}{2} P_-$. The
joint input is a product:

$$
\rho_{XY}^{(B)} \;=\; \rho_X^{(B)} \otimes \rho_Y^{(B)} \;=\; \tfrac{1}{4}\,P_+ \otimes P_-.
$$

This idealises the actual code path (which uses a random rank-≤2
state in each sector, see `src/quantum/interaction_simulation.cpp:495`)
— maximally distributed information *within* each sector, no
sector-superposition, no cross-vertex correlation.

### 4.1 Marginals and entropies

$S(\rho_X^{(B)}) = S(\rho_Y^{(B)}) = \ln 2$ (a maximally-mixed state
on a 2-dim subspace).

$S(\rho_{XY}^{(B)}) = S(\rho_X^{(B)}) + S(\rho_Y^{(B)}) = 2\ln 2$
(product state, additive entropy).

$$
I^{(B)}_{\mathrm{input}} \;=\; \ln 2 + \ln 2 - 2\ln 2 \;=\; 0.
$$

Two pure-distributed-uncorrelated systems carry the same per-vertex
entropy ($\ln 2$ each) as the Bell case, but **none of it is shared**
— hence $I = 0$.

### 4.2 Action of $\hat U$

$\rho_{XY}^{(B)} = \tfrac{1}{4} P_+ \otimes P_-$ sits entirely in the
$(+,-)$ block. Within that block:

$$
\rho_{XY}^{(B)}\bigg|_{(+,-)\text{-block}} \;=\; \tfrac{1}{4}\,I_4
$$

— the maximally mixed state on the 4-dim $(+,-)$ block. **Any unitary
commutes with the identity**:

$$
\hat U\bigl(\tfrac{1}{4}\,I_4\bigr)\hat U^\dagger \;=\; \tfrac{1}{4}\,\hat U\,\hat U^\dagger \;=\; \tfrac{1}{4}\,I_4.
$$

So $\rho_{AB} = \rho_{XY}^{(B)}$, **exactly the input**.
$\hat U$ acted invisibly.

Therefore:

$$
I^{(B)}_{\mathrm{joint}} \;=\; I^{(B)}_{\mathrm{input}} \;=\; 0.
$$

### 4.3 Per-edge MI

| edge | $I_e$ | $\ell_e = -\ln(I_e/I_{\max})$ |
|---|---:|---:|
| $X$–$Y$ | $0$ | $-\ln \varepsilon_I$ (clamped) |
| $X$–$X'$ | $\ln 2$ | $\ln 2 \approx 0.693$ |
| $Y$–$Y'$ | $\ln 2$ | $\ln 2$ |
| 4 hub spokes | $0$ each | clamped |
| 3 cross-pair | $0$ each | clamped |

With the simulator's $\varepsilon_I = 10^{-10}$, the clamp is
$\ell = -\ln(10^{-10}) = 10 \ln 10 \approx 23$.

**8 of 10 edges are saturated at the maximum length** (the opposite
extreme from Case A's saturation at length 0). Only the two
worldline-self edges, with $\ell = \ln 2$, carry any geometric
information.

### 4.4 What happened dynamically

Again, nothing. $\hat U$ commutes with the maximally-mixed state on
the active sub-block — every unitary does. The Hamiltonian
contributed neither phase structure nor entanglement.

## 5. Reading: when does $H_{\mathrm{pair}}$ engage?

Both extremes — pure Bell-eigenstate input and maximally-mixed
product input — leave $\hat U$ inert. The Hamiltonian engages only in
the **middle**: inputs that are simultaneously

1. **Not eigenstates of $\hat H_{\mathrm{pair}}$** (otherwise $\hat U$
   is a global phase), and
2. **Not maximally mixed on the active sub-block** (otherwise
   $\hat U I \hat U^\dagger = I$).

Concretely, "engagement" requires *coherences* in the spin index
within an allowed sub-block: e.g., off-diagonal terms
$\ket{+\,0,\,-\,1}\bra{+\,1,\,-\,0}$ or pure-basis-state inputs like
$\ket{+\,0,\,-\,1}$ that the $(XX+YY)$ rotor can mix with
$\ket{+\,1,\,-\,0}$.

A clean intermediate sanity check: take
$\rho_{XY} = \ket{+\,0,\,-\,1}\bra{+\,0,\,-\,1}$ (pure product, basis
state, not eigenstate). Then

$$
\hat U \ket{+\,0,\,-\,1} \;=\; e^{+i J_c \Delta t}\bigl[c\,\ket{+\,0,\,-\,1} - i s\,\ket{+\,1,\,-\,0}\bigr]
$$

with $c \approx 0.992$, $s \approx 0.125$. Tracing out $Y$:

$$
\rho_X' \;=\; c^2\,\ket{+\,0}\bra{+\,0} \;+\; s^2\,\ket{+\,1}\bra{+\,1} \;\approx\; 0.984\,\ket{+\,0}\bra{+\,0} \;+\; 0.016\,\ket{+\,1}\bra{+\,1}.
$$

$S(\rho_X') = -[0.984\ln 0.984 + 0.016\ln 0.016] \approx 0.080$ nats.
$I^{}_{\mathrm{joint}} = 2 \times 0.080 \approx 0.16$ nats — **modest
but nonzero**. This is the actual regime the code occupies (random
rank-2 in-sector inputs sit between this and Case B): some MI gets
generated, but at v0.2 defaults the rotation angle $2 J_s \Delta t =
0.125$ rad keeps it small.

### 5.1 Why the J_s = 0 sweep result was suspicious

At $J_s = 0$ exactly, the $(XX+YY)$ term vanishes and the only
remaining Hamiltonian term, $J_c\,\hat Q\otimes\hat Q$, is constant on
each fixed-charge-pair sub-block (eigenvalue $\pm 1$). So
$\hat U|_{J_s=0}$ is a *global phase* on every sub-block — **any
charge-eigenstate input is an eigenstate of $\hat U|_{J_s=0}$**, and
$I_{\mathrm{joint}} = I_{\mathrm{input}}$ identically.

Since the code's initial inputs are independent random states in
each sector ($I_{\mathrm{input}} = 0$), at $J_s = 0$ every
interaction generates zero new MI. Every edge except the two
worldline-self ones gets clamped at $\ell = -\ln \varepsilon_I$.
The peak $D_S \approx 4.08$ reported in [Experiment
05](../experiments/05-v0.2-h-param-sweep.md) reflects the bare
topology of the (2,3)-Pachner-stacked cell complex with degenerate
edge weights — *not* a Hamiltonian-shaped geometry. Experiment 05's
finding "J_s = 0 is the H_DS4 working point" should be downgraded
until the dynamics-on regime is also checked at the same lattice
size.

## 6. Physically meaningful manipulations to increase engagement

The above analysis suggests six directions, each with a clean
physical interpretation. They're listed in increasing order of how
disruptive they are to the current v0.2 picture.

### 6.1 Lower initial-state entropy

The current `buildInitialLayer` produces random rank-$\leq 2$ mixed
states in each sector by sampling 4×4 complex Gaussians, projecting
to the sector, and trace-normalising. The result has typical
entropy $\sim 0.5\,\ln 2$, which puts it most of the way to
maximally mixed in the sector — too close to Case B for $\hat U$ to
move it strongly.

**Try**: project onto a *random pure state* in the sector instead
(rank 1, $S = 0$). This makes each vertex maximally distinguishable
in its sector and gives $\hat U$ full purchase via $(XX+YY)$
rotation. Cost: trivial code change in
[`buildInitialLayer`](https://github.com/akellehe/tessera/blob/main/src/quantum/interaction_simulation.cpp#L495).

Interpretation: replaces "random spinor in this charge sector"
(thermal-ish vacuum) with "definite single-particle state".

### 6.2 Charge-sector superposition inputs

Lift the off-sector zeroing in `buildInitialLayer`. The vertex state
then has support on **both** charge sectors, with
$\bra{+}\rho\ket{-}$ off-diagonal terms. Now $\hat Q\otimes\hat Q$
is no longer constant — its $\pm 1$ eigenvalues mix coherently
across blocks, and the input is no longer an eigenstate. Bell-like
entanglement starts being generated.

Interpretation: relaxes the strict sector-confinement convention.
Physically, this is the "no longer in a definite charge
eigenstate" regime — appropriate if we want a quantum *number*
(charge) to be conserved on average but not exactly in the wave
function. Tradeoff: $Q$-conservation as a sharp invariant is lost
in the input; it would still be preserved as an expectation value.

### 6.3 Increase $\Delta t$

The rotation angle in $(XX+YY)$ is $2 J_s \Delta t = 0.125$ rad at
defaults. Doubling $\Delta t$ to $0.5$ gives angle $0.25$ rad
($\approx 14^\circ$, $\sin \approx 0.247$) — about 4× more MI
generated per event. Cost: nothing, just a config value.

Interpretation: longer "exposure" per interaction event. The
unitary is still the same form but takes a bigger Trotter step in
the underlying continuous-time evolution.

### 6.4 Add a spin–charge cross term

The current $H_{\mathrm{pair}}$ has no operator that couples the
*spin* and *charge* indices on the same vertex. So a vertex in a
charge eigenstate stays in that charge eigenstate, and the spin
register evolves only via the bipartite $(XX+YY)$. Adding a single-
site term

$$
H_{\mathrm{SOC}} \;=\; \lambda\,(\sigma^x_q \otimes \sigma^x_s)_A \;+\; \lambda\,(\sigma^x_q \otimes \sigma^x_s)_B,
$$

i.e., a *spin–charge correlated flip* on each vertex, breaks both
charge conservation and the charge/spin block independence. Inputs
that started as charge eigenstates develop charge-coherences under
$\hat U$, and $(XX+YY)$ then bipartitely entangles in the wider
state space.

Interpretation: spin–orbit-like coupling. The physical analog is a
term that mixes spin and "internal flavor" on each vertex — making
the qudit basis less reducible than a tensor product.

### 6.5 Replace $(XX+YY)$ with a full Cartan entangler

The general SU(4) Cartan entangling content of any pair unitary is
$\exp(i (c_x XX + c_y YY + c_z ZZ))$, with three independent
coupling constants. The current $H_{\mathrm{pair}}$ uses only the
$c_x = c_y = J_s$ slice (an XXZ-like coupling with $c_z = 0$). The
$ZZ$ term — diagonal in the spin index — would lift the
$\ket{00}/\ket{11}$ degeneracy that currently makes those states
$(XX+YY)$-stationary.

Interpretation: extends the spin-sector dynamics from "XY model" to
"XXZ model" (or fully Heisenberg). Standard in lattice spin physics;
the additional coupling is just one more knob.

### 6.6 Time-dependent / kicked $\hat U$

Currently $\hat U$ is the same matrix at every event. Allowing a
slowly varying $\hat U$ (e.g., a Floquet drive where $J_s, J_c$
modulate with cell index) would prevent the input from settling
into any eigenstate of a single $\hat U$.

Interpretation: an *open* quantum system whose environment evolves.
Most ambitious; touches the autonomy of $H_{\mathrm{pair}}$ as a
single fundamental object.

## 7. Recommendations

In rough order of "do this first":

1. **(6.1)** Random *pure* states in each sector, instead of random
   mixed. Trivial code change; immediately distinguishes the
   $J_s = 0$ inert regime from non-trivial dynamics. Predicted: peak
   $D_S$ at $J_s = 0$ should *rise* above 4.08 once the inputs aren't
   maximally distributable, because the cross-pair edges will now
   carry non-zero $I_{\mathrm{joint}}$.
2. **(6.3)** Increase $\Delta t$ to $\sim 0.5$ in a control run, see
   whether peak $D_S$ moves systematically.
3. **(6.2)** A separate experiment with sector-superposition
   inputs. This is the cleanest way to actually engage $\hat
   Q\otimes\hat Q$, which is currently almost entirely cosmetic at
   the chosen initial conditions.

Each of (6.1)–(6.3) is a one-line config change or a few-line
patch to `buildInitialLayer`. They can be run as small sweeps
against the same N=8, T=2500 lattice the H-parameter sweep used,
and the resulting $D_S$ surfaces should be qualitatively different
from the current ones — *that* would mean we've moved out of the
"dynamics inert" regime and into "Hamiltonian-shaped geometry."

## 8. A second interaction event

To see how state propagates through the history and how the
Regge action accumulates, work the next event. After cell 1 has
fired on inputs $X, Y$ producing $X', Y', AB_1$, the frontier
contains $\{X', Y', AB_1\}$ plus whatever initial-layer vertices
weren't consumed.

We'll work the "**intermediate engagement**" case from §5 — pure
basis-state inputs, where $\hat U$ actually does something
non-trivial — so the propagation is visible. Cell 1 inputs:

$$
\rho_X^{(1)} = \ket{+\,0}\bra{+\,0}, \qquad \rho_Y^{(1)} = \ket{-\,1}\bra{-\,1},
\qquad \rho_{XY}^{(1)} = \ket{+\,0,\,-\,1}\bra{+\,0,\,-\,1}.
$$

### 8.1 Cell 1 outputs (recap of §5's mid-engagement case)

From the boxed action on the $(+,-)$ block:

$$
\ket{\psi_1} \;=\; \hat U\ket{+\,0,\,-\,1} \;=\; e^{+i J_c \Delta t}\bigl[c\,\ket{+\,0,\,-\,1} \;-\; i s\,\ket{+\,1,\,-\,0}\bigr],
$$

with $c = \cos(2 J_s \Delta t) \approx 0.99219$, $s = \sin(2 J_s \Delta t) \approx 0.12467$.

The post-event joint is pure: $\rho_{AB_1} = \ket{\psi_1}\bra{\psi_1}$.
Marginals:

$$
\rho_{X'}^{(1)} \;=\; c^2\,\ket{+\,0}\bra{+\,0} \;+\; s^2\,\ket{+\,1}\bra{+\,1}, \qquad
\rho_{Y'}^{(1)} \;=\; c^2\,\ket{-\,1}\bra{-\,1} \;+\; s^2\,\ket{-\,0}\bra{-\,0},
$$

with $c^2 \approx 0.9844$, $s^2 \approx 0.0156$. Per-vertex entropies:

$$
S(\rho_{X'}^{(1)}) \;=\; S(\rho_{Y'}^{(1)}) \;=\; -[\,c^2 \ln c^2 + s^2 \ln s^2\,] \;\approx\; 0.0801\ \text{nats}.
$$

And $I^{(1)}_{\mathrm{joint}} = 2 S(\rho_{X'}^{(1)}) \approx 0.160$ nats
(since $\rho_{AB_1}$ is pure).

The joint $\rho_{AB_1}$ is also stored as `quditJointOf_[(X', Y')]`
(post-#16 Choi path) and carries the full coherence:

$$
\rho_{AB_1} \;=\; \begin{pmatrix}
c^2 & 0 & 0 & i s c\\
0 & 0 & 0 & 0\\
0 & 0 & 0 & 0\\
-i s c & 0 & 0 & s^2
\end{pmatrix}
$$

in the $\{\ket{+\,0,\,-\,1},\,\ket{+\,0,\,-\,0},\,\ket{+\,1,\,-\,1},\,\ket{+\,1,\,-\,0}\}$
basis (filling the active $(+,-)$ block, zero elsewhere).

### 8.2 Cell 2 setup — reuse the two products

The most informative second event reuses both products of cell 1
together: $X^{(2)} = X'$ from cell 1, $Y^{(2)} = Y'$ from cell 1.
The two cells then share the $X'$-$Y'$ edge — one physical edge in
the spacetime, referenced by both cells' MI tables:

![Two generations of (2,3) Pachner cells stitched together via
the shared $X'$-$Y'$ edge; cell 1 produces $X', AB_1, Y'$ from
$X, Y$ at generation $n=1$, and cell 2 produces $X'', AB_2, Y''$
from $X', Y'$ at generation
$n=2$.](../../figures/h_pair_two_generations_diagram.svg)

The input joint to cell 2 is then **not** a product of marginals —
it's the stored $\rho_{AB_1}$:

$$
\rho_{XY}^{(2)} \;=\; \rho_{AB_1} \;=\; \ket{\psi_1}\bra{\psi_1}.
$$

This is the case the v0.2 code distinguishes (via `quditJointOf_`):
when two frontier vertices share an interaction history, the
simulator uses the **stored 16-dim joint**, not the tensor product
of marginals. The path-integral picture below makes this matter.

### 8.3 $\hat U$ on cell 2

We're applying $\hat U$ to a state that is *itself* the output of
$\hat U$ acting on $\ket{+\,0,\,-\,1}$ at cell 1. Stacking:

$$
\hat U \hat U \ket{+\,0,\,-\,1} \;=\; \hat U^2 \ket{+\,0,\,-\,1}.
$$

In the $\{\ket{+\,0,\,-\,1},\,\ket{+\,1,\,-\,0}\}$ 2-dim sub-block,
$\hat U$ acts as $e^{+i J_c \Delta t}\,R$ where

$$
R \;=\; \cos(2 J_s \Delta t)\,I \;-\; i\sin(2 J_s \Delta t)\,\sigma^x.
$$

Stacking is a rotation-angle doubling:

$$
R^2 \;=\; \cos(4 J_s \Delta t)\,I \;-\; i\sin(4 J_s \Delta t)\,\sigma^x.
$$

At v0.2 defaults: $4 J_s \Delta t = 0.25$ rad, so $\cos(0.25) \approx 0.96891$, $\sin(0.25) \approx 0.24740$.

Cell 2's output:

$$
\ket{\psi_2} \;=\; \hat U^2 \ket{+\,0,\,-\,1} \;=\; e^{+2 i J_c \Delta t}\bigl[\cos(4 J_s \Delta t)\,\ket{+\,0,\,-\,1} \;-\; i\sin(4 J_s \Delta t)\,\ket{+\,1,\,-\,0}\bigr].
$$

Marginals of $\rho_{AB_2} = \ket{\psi_2}\bra{\psi_2}$:

$$
\rho_{X''}^{(2)} \;=\; \cos^2(4 J_s \Delta t)\,\ket{+\,0}\bra{+\,0} \;+\; \sin^2(4 J_s \Delta t)\,\ket{+\,1}\bra{+\,1},
$$

with $\cos^2 \approx 0.9388$, $\sin^2 \approx 0.0612$.

$S(\rho_{X''}^{(2)}) \approx 0.230$ nats. And $I^{(2)}_{\mathrm{joint}} \approx 0.46$ nats — **about 2.9× larger than cell 1's $I^{(1)}_{\mathrm{joint}}$**.

### 8.4 Per-edge MI in cell 2

The inputs to cell 2 are not maximally mixed, and the input joint
already carries MI $I^{(1)}_{\mathrm{joint}} \approx 0.161$ nats from cell 1.
So for the new cell's 10 edges:

| edge | $I_e$ (nats) | $\ell_e = -\ln(I_e/I_{\max})$ |
|---|---:|---:|
| $X^{(2)}$–$Y^{(2)}$ (= $X'$–$Y'$ of cell 1) | $0.160$ | $2.155$ |
| $X^{(2)}$–$X^{(2)\prime}$ | $S(\rho_{X^{(2)}}) = 0.080$ | $2.852$ |
| $Y^{(2)}$–$Y^{(2)\prime}$ | $S(\rho_{Y^{(2)}}) = 0.080$ | $2.852$ |
| 4 hub spokes ($AB_2$) | $I^{(2)}_{\mathrm{joint}} \approx 0.461$ | $1.102$ |
| 3 cross-pair | $I^{(2)}_{\mathrm{input}} = 0.160$ | $2.155$ |

(With $I_{\max} = 2 \ln 2 \approx 1.386$.)

The geometry of cell 2 is *richer* than cell 1's:

- Cell 1 had only the 2 worldline edges at finite length ($\ln 2$);
  the other 8 edges were clamped at the $\varepsilon_I$ floor.
- Cell 2 has **all 10 edges at finite, distinct lengths**, because
  the input is now correlated and U has rotated further.

### 8.5 What the second cell taught us

1. **Stacking $\hat U$ doubles the rotation angle** in the active
   spin sub-block. So if we want richer per-event dynamics, we
   can either (a) make $J_s \Delta t$ larger (§6.3) or (b) chain
   events through the same worldlines so the angle accumulates.
2. **Cell 2 inherits cell 1's joint state**, not just marginals.
   This is exactly the v0.2 / #16 storage convention paying off
   — `quditJointOf_[(X', Y')]` was set in cell 1 and is read here.
3. **The MI signal compounds**. $I_{\mathrm{joint}}$ at cell 2
   was ≈ 3× cell 1's, because cell 2 starts from an entangled input
   rather than a product input.

If we'd worked Case A (Bell-like input) or Case B (max-mixed input)
instead, $\hat U^2$ would still be a global phase or trivial, and
cell 2 would inherit cell 1's degeneracy. The "intermediate
engagement" regime is the one where the history does any work.

## 9. How this all fits into the path integral

The Monte Carlo samples a partition function

$$
Z \;=\; \sum_{\text{histories } h}\, \pi(h),
$$

where each *history* $h$ is a sequence of accepted moves $(m_1, m_2,
\dots, m_T)$ that grew the initial layer into a final cell complex
$\mathcal{K}_T$. The equilibrium distribution is

$$
\pi(h) \;\propto\; \exp\!\bigl(-\beta\, S_{\mathrm{Regge}}[\mathcal K_T]\bigr) \;=\; \exp\!\Bigl(-\beta\,\sum_{\text{hinges } H \subset \mathcal K_T} A_H \cdot \varepsilon_H\Bigr).
$$

### 9.1 What's summed over

The configuration space is **the set of growable cell complexes**
reachable from the initial layer by a sequence of `interact`
(and optionally `annihilate`, `pairCreate`) moves. Crucially:

- $\mathcal K_T$ is a 4-dimensional simplicial complex of (2,3)
  cells, not a continuum metric.
- Each cell $c$ contributes its 10 edges; each edge $e$ has a
  *length* $\ell_e$ determined by the MI of the pair-state that
  the cell was constructed from (§2, §3.3, §4.3, §8.4).
- The action $S_{\mathrm{Regge}} = \sum_H A_H \varepsilon_H$ is a
  function of those edge lengths via the hinge areas and deficit
  angles.

So $S$ is *not* a separate quantum-mechanical functional layered on
top of a fixed metric. The edge lengths, hence $A_H$ and
$\varepsilon_H$, are *outputs* of the quantum dynamics
($\hat U \rho \hat U^\dagger \rightarrow$ MI $\rightarrow \ell$).
The path integral is over histories that *jointly specify the
geometry and the quantum state evolution*.

### 9.2 Conditional structure of the Boltzmann weight

The cumulative action telescopes:

$$
S_{\mathrm{Regge}}[\mathcal K_T] \;=\; \sum_{n=1}^{T} \Delta S_n,
$$

where $\Delta S_n$ is the *new* hinge contribution from event $n$ —
i.e., the action of the hinges added by cell $n$ that weren't in
$\mathcal K_{n-1}$. From §2, $\Delta S_n$ is computed from the 10
edge lengths of cell $n$, which depend on:

- The marginal states $\rho_{X_n}, \rho_{Y_n}$ of the chosen
  inputs at step $n$;
- The stored input joint $\rho_{X_n Y_n}$, if any, propagated
  forward from earlier events through `quditJointOf_`.

So the Boltzmann weight factorises *along the history*:

$$
\pi(h) \;\propto\; \prod_{n=1}^{T} e^{-\beta\,\Delta S_n(\rho_{X_n Y_n}^{\text{state at }n-1})},
$$

with the $n$-th factor *conditional* on the quantum state at step
$n-1$. Each accepted move multiplicatively reweights the history.

### 9.3 The Metropolis sampler

The Metropolis-Hastings acceptance for proposed move $m_n$ is

$$
P_{\mathrm{accept}} \;=\; \min\Bigl(1,\;\; \exp\bigl[-\beta\,\Delta S_n + \log P_{\mathrm{prop}}(\text{reverse}) - \log P_{\mathrm{prop}}(\text{forward})\bigr]\Bigr),
$$

where the proposal-probability prefactor handles the asymmetry of
"propose this cell" vs "un-propose it". The detailed-balance
condition ensures that as $T \to \infty$, samples are drawn from
$\pi(h) \propto \exp(-\beta S_{\mathrm{Regge}})$ as desired.

At $\beta \to 0$, the action is irrelevant and any topologically
legal move is accepted — the chain wanders freely in configuration
space. At large $\beta$, the chain freezes near the action minima.
The H_DS4 hypothesis (§3 of the [v0.2 design](v0.2.md)) is the
claim that *some* $\beta$ exists where the equilibrium ensemble's
geometry has emergent spectral dimension $D_S = 4$.

### 9.4 Putting cells 1 and 2 into the path integral

For our 2-cell intermediate-engagement example:

$$
\pi(h_2) \;\propto\; e^{-\beta\,(\Delta S_1 + \Delta S_2)}.
$$

With Wick-rotated Euclidean squared lengths $\ell_e^2$, the
per-hinge action $A_H \varepsilon_H$ is a definite (computable)
function of the 10 edges of each cell. Sketching the dependence:

$$
\Delta S_1 \;=\; \mathcal F\bigl(\,c, s, S_X^{(1)}, S_Y^{(1)}, I^{(1)}_{\mathrm{input}}, I^{(1)}_{\mathrm{joint}}\,\bigr),
$$

and

$$
\Delta S_2 \;=\; \mathcal F\bigl(\,c^{(2)}, s^{(2)}, S_X^{(2)}, S_Y^{(2)}, I^{(2)}_{\mathrm{input}}, I^{(2)}_{\mathrm{joint}}\,\bigr),
$$

where the $^{(2)}$-superscripted quantities depend on the $^{(1)}$-quantities
via the propagation rules above ($S_X^{(2)} = S(\rho_{X'}^{(1)})$
etc., and $I^{(2)}_{\mathrm{input}} = I^{(1)}_{\mathrm{joint}}$).

In numbers (v0.2 defaults, pure-basis-state inputs):

| $n$ | $S_X^{\text{in}}$ | $S_Y^{\text{in}}$ | $I_{\mathrm{input}}$ | $I_{\mathrm{joint}}$ |
|---:|---:|---:|---:|---:|
| 1 | $0$       | $0$       | $0$       | $0.160$ |
| 2 | $0.080$   | $0.080$   | $0.160$   | $0.461$ |
| 3 | $0.230$   | $0.230$   | $0.461$   | $0.788$ |
| 4 | $0.394$   | $0.394$   | $0.788$   | $1.078$ |
| 5 | $0.539$   | $0.539$   | $1.078$   | $1.285$ |
| 6 | $0.643$   | $0.643$   | $1.285$   | $1.381$ |
| 7 | $0.691$   | $0.691$   | $1.381$   | $1.354$ |
| 8 | $0.677$   | $0.677$   | $1.354$   | $1.208$ |
| 9 | $0.604$   | $0.604$   | $1.208$   | $0.961$ |
| 10 | $0.480$  | $0.480$   | $0.961$   | $0.648$ |
| 11 | $0.324$  | $0.324$   | $0.648$   | $0.322$ |
| 12 | $0.161$  | $0.161$   | $0.322$   | $0.063$ |

**Important**: the MI signal does **not** saturate at $I_{\max}$ — it
**oscillates coherently**. Each application of $\hat U$ rotates the
joint state further around the $\sigma^x$ axis of the spin sub-block.
Three key angles, parameterised by $n\theta$ where $\theta = 2 J_s \Delta t$:

| event $n$ | $n\theta$ | population $(\cos^2, \sin^2)$ | $S(\rho_X)$ | $I_{\mathrm{joint}}$ |
|---:|---|---|---:|---:|
| $0$ | $0$ | $(1, 0)$ — pure input $\ket{+0,-1}$ | $0$ | $0$ |
| $\approx 6$ | $\pi/4$ | $(0.5, 0.5)$ — first MI **peak** | $\ln 2$ | $I_{\max} = 2\ln 2$ |
| $\approx 13$ | $\pi/2$ | $(0, 1)$ — fully swapped to pure $\ket{+1,-0}$ | $0$ | $0$ (first MI **node**) |
| $\approx 19$ | $3\pi/4$ | $(0.5, 0.5)$ — second MI peak | $\ln 2$ | $I_{\max}$ |
| $\approx 25$ | $\pi$ | $(1, 0)$ — back to input | $0$ | $0$ |

so MI is a periodic function of $n$ with period
$T_{\mathrm{MI}} = 2\pi/(2\theta) = \pi/\theta \approx 25$ cells at
v0.2 defaults (the *state* itself has period $2\pi/\theta \approx 50$
cells, but the marginal entropy is even in the deviation from a
pure state, so it has half the period). The first MI peak is at

$$
n_{\mathrm{peak}} \;\approx\; \frac{\pi}{4\theta} \;=\; \frac{\pi}{8 J_s \Delta t} \;\approx\; 6.28
$$

cells along a chained worldline at v0.2 defaults — six events before
the unitary rotation drives $I_{\mathrm{joint}}$ to its maximum,
and another six before it falls back to zero.

> **Caveat.** This coherent oscillation only applies along an
> *uninterrupted* chain that keeps reusing the same products. In
> the actual simulation, the second cell rarely picks both of the
> first cell's products: the frontier has many vertices and
> `getRandomTopSimplex` / `interact` chooses among them. So most
> worldlines see only a few stacked rotations before their
> products get consumed by interactions with other vertices,
> which mixes their state and breaks the coherence. The estimate
> $n_{\mathrm{peak}} \approx 13$ is therefore an upper bound on
> coherent buildup along a single chain, not a typical correlation
> length in the simulated complex.

### 9.5 What "geometry from entanglement" really means here

Read this all backward, and the path integral is doing something
specific:

1. The simulator proposes a sequence of moves $\{m_n\}$.
2. Each move generates a small block of edge-MI from $\hat U$
   acting on the local pair-state.
3. Those MI values become edge *lengths*.
4. The lengths feed the Regge action $\Delta S_n$.
5. Metropolis accepts/rejects based on $e^{-\beta \Delta S_n}$.
6. The equilibrium ensemble samples geometries weighted by
   $e^{-\beta S_{\mathrm{tot}}}$.

There is no separate "quantum field on a fixed background." The
*quantum information dynamics generate the metric* (via MI →
length) and the *Regge action filters geometries* (via Metropolis).
The H_DS4 question — does this generate a 4-dimensional spacetime
— is a question about the joint structure of these two
ingredients.

Conversely, the things this derivation makes vivid:

- If $\hat U$ is dynamically inert on the chosen inputs (§3 / §4),
  every $\Delta S_n$ is approximately a *constant* — driven only
  by the worldline-self edges with $\ell = -\ln(S/I_{\max})$, and
  independent of which $(X, Y)$ pair was chosen at step $n$. The
  geometry then samples a *single* effective lattice up to
  combinatorial choices.
- The "interesting" regime is the one where $\Delta S_n$ varies
  strongly across move proposals — i.e., where the H-engagement
  conditions of §5 are met.
- Manipulations from §6 are all ways to make $\Delta S_n$ depend
  more strongly on the quantum state of the chosen pair, so the
  Metropolis sampler can actually select among quantum-distinct
  geometries rather than a near-uniform topological prior.

## See also

- [v0.2 design note](v0.2.md) — sets up $H_{\mathrm{pair}}$ and
  introduces the Choi-state $\Sigma_{AB}$.
- [Experiment 05: H-parameter sweep](../experiments/05-v0.2-h-param-sweep.md)
  — the surface result this derivation reinterprets.
- [Intellectual lineage](../../overview/intellectual_lineage.md#5-the-qudit-basis-charge-intrinsic-to-the-state)
  — the qudit-basis story this builds on.
- `src/quantum/interaction_simulation.cpp` lines 455–525
  (`buildInitialLayer`) and 760–810 (`computeInteractionQudit`):
  the code paths this analysis matches.
