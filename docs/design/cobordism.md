# MergeCobordism Algorithm

Our goal is to build an emergent operator from starting and ending states in a pair of pants through a cobordism bulk 
mediated by the Regge action on the _dual_ complex while maintaining a valid simplicial manifold in both spaces. Let $ 
\psi_A \in \mathscr{H}_A $ and $ \psi_B \in \mathscr{H}_B $ represent qubit states of each of two quantum systems, 
respectively. Let these be two input states to a cobordism $ W_{AB} $. Let $ \psi_{AB} $ be the expected output/final 
state of the same cobordism.

## The functor $\operatorname{geo}()$

Let $ \operatorname{geo}() $ be a functor that brings a quantum state $ \psi \in \mathscr{H} $ to a simplicial complex 
$ K $. $\operatorname{dim}(\psi_A) = 2 $ because $ \psi_A $ represents a qubit.

$$
\operatorname{geo} : \mathscr{H} \rightarrow K
$$

$ \operatorname{geo}(\psi) $ is the 2-complex (a surface) whose degree-1 Hodge kernel realizes $ \psi : \psi \in 
\ker(L_1(\operatorname{geo}(\psi))) $ with $b_1 = \operatorname{dim}(\psi) $ where $b_1$ is the first betti number.

Let $ L_{1}^A $ and $L_1^B$ be Hodge Laplacians of $ \operatorname{geo}(\psi_A) $ and $ \operatorname{geo}(\psi_B) $ 
respectively.

Now define a cobordism boundary $ \partial W_{AB} $ as $ \operatorname{geo}(\psi_A) \sqcup \operatorname{geo}(\psi_B) 
\sqcup \operatorname{geo}(\psi_{AB}) $. 

Let $U^{Choi}_{AB}$ be the Choi decomposition (Choi state) of operator $U_{AB} = \operatorname{unvec}(U^{Choi}_{AB}), $ 
with $ (U_{AB})_{ij} = (U^{Choi}_{AB})_{2i+j} $. 

Let the bulk be defined as 

$$
\operatorname{geo}(U^{Choi}_{AB}) = W_{AB} - \partial W_{AB}, \qquad U^{Choi}_{AB} \in
\ker \big(L_1^{U^{Choi}_{AB}}\big)
$$

where the four Choi components are constrained by charge conservation $\sum_k (U^{Choi}_{AB})_k = 0$, so the realizable space has $\operatorname{dim}(U^{Choi}_{AB}) = 3$.

This $\sum = 0$ charge-conserving structure is the same one that carries the color singlet $[1, \omega, \omega^2]$ in the color-singlet (register) construction: there three holes give $b_1 = 2$ on the $\sum = 0$ hyperplane, and here four vertex-disjoint holes give $b_1 = 3 = \operatorname{dim}(U^{Choi}_{AB})$ — the bare icosahedron's capacity ($12$ vertices over $3$).

Fix $ \psi_A \in \ker(L_1^A) $ so that the state, $ \psi_A $ is represented by $ L_1^A$. Similarly fix $ \psi_B \in 
\ker(L_1^B) $ as well as $ \psi_{AB} \in \ker(L_1^{AB})$ be the expected final state.

Let $ W_{AB} = \operatorname{geo}(U^{Choi}_{AB}) \cup \partial W_{AB}$ with $U^{Choi}_{AB} \in \ker(
L_1^{U^{Choi}_{AB}}) $ emerge from the optimization process described below.

The holes above live on $\partial W$, so forming the bulk $W - \partial W$ deletes them and $\ker L_1(W - \partial W) = 0$. The emergent operator must be a genuinely interior cycle. So $W$ is a 3-manifold whose boundary $\partial W$ is the 2-complex states, and the operator is an interior 1-handle; $\ker L_1(W - \partial W)$ counts the handles. Build it from the closed $S^2 \times S^1$ (the handle, $\ker L_1 = 1$, every vertex interior): enrich the interior with boundary-fixed Pachner adds, then open the $\partial W$ cavities (the states) by gated `removeInteriorCellChecked`, rolling back any cut that drops the handle. Use REGGE, not CDT, so the edge lengths are free and the causal type is emergent, with no preferred foliation.

## Residuals

Let's construct an objective function, $ F $, that we can minimize to build the bulk of our cobordism, $ W_{AB} - 
\partial W_{AB}$, from the boundary $ \partial W_{AB} $.

The first term will involve the Regge Action, $S_{Regge}$. We want to extremize it such that $ \delta S_{Regge} = 
0 $ on the space _dual_ to the primal complex $W_{AB}$, namely $ \star W_{AB} $. We use the gradient as the part to 
minimize: 

$$ 
r_{S_{Regge}} = \beta\|\nabla S_{Regge}\|^2.
$$

Where $ \beta $ is a weight on the impact of the action.

Next we add residuals that force $ \psi_i \in \ker(L_1^i) $ for each $i \in \{A, B, AB\} $.

$$
r_\psi = \sum_i \big\|\psi_i - P_{\ker L_1^i} \psi_i \big\|^2 = \sum_i \big\|(I - P_{\ker L_1^i}) \psi_i \big\|^2
$$

where $P_{\ker L_1^i}$ is the orthogonal projector onto the kernel and the element is the projection $ P_{\ker L_1^i} 
\psi_i $. Then the total residual is

$$
r = r_{S_{Regge}} + r_{\psi}
$$

and then we use the exact analytic Jacobian to do a least squares gradient descent over the residual.

$$
J = \frac{\partial f}{\partial x} = \begin{bmatrix} \sqrt{\beta}\, \nabla^2 S_{Regge} \\ \partial_x (I - P_i) \psi_i \end{bmatrix} 
$$

The least-squares solver (Levenberg-Marquardt) consumes the exact analytic Jacobian above. Its stationarity block is the 
exact action Hessian $\nabla^2 S_{Regge}$; its state constraint block is the analytic derivative of projected periods.

## Optimization

The algorithm converges when $ r < \epsilon$, where $\epsilon$ is a user-defined tolerance, or when MAX_ATTEMPTS is 
reached, defaulting to 100.

1. Start with an icosahedron connecting the boundary as the minimal manifold/complex representing the bulk of $W_{AB}$ connecting the starting states $\operatorname{geo}(\psi_A) \sqcup \operatorname{geo}(\psi_B) $ to the ending state $ \operatorname{geo}(\psi_{AB}) $. Open the bulk's holes by boundary-fixed surgery: remove $\operatorname{dim}(U^{Choi}_{AB}) + 1$ vertex-disjoint faces (`removeInteriorCell`), since removing $k$ vertex-disjoint faces from a $2$-sphere yields $b_1 = k - 1$; for the $\sum = 0$ constrained operator this is $4$ faces giving $b_1 = 3 = \operatorname{dim}(U^{Choi}_{AB})$. This surgery is the only operation that creates the operator's holes; the RemoveMove/AddMove Pachner moves below are topology-preserving and only relax the metric. 
2. Now relax edge lengths by minimizing $r$ using the analytic gradient/hessian and a least squares optimizer. 
3. Does the optimizer converge?
   1. Yes? Execute one RemoveMove (a pachner move, you can find in our code base) and relax again, and go to 3. 
   1. No? If there was a previous convergence; return. Otherwise execute an AddMove, relax again, and go to 3.

## Implementation

This should all be implemented in a C++ class, `MergeCobordism(inputStates, outputStates, U)`. The **primary
emergent quantity is the one the caller did NOT supply**:

 - **Output supplied** (no $U$): pin `inputStates` *and* `outputStates` together as $\partial W$, relax, and the
   **operator** $U$ is the primary emergent quantity (`operatorU` — see the deferral note below).
 - **$U$ supplied** (no `outputStates`): apply $U$ to the inputs to get the expected output and pin it as the
   output target, relax, and the **output state** is the primary emergent quantity, read over the output cycles
   (`outputState`) — the $\#353$ inputs-$\to$-emergent-output flow. (Making the transport itself realise $U$ —
   "$U$ as the bulk constraint" — is deferred; see the note below.)

`MergeCobordism` should have several members used for later introspection and analysis. 
 - `inputStates`
 - `outputStates` (as supplied, or computed from $U$)
 - `cobordism` ($W$)
 - `boundary` ($\partial W$)
 - `bulk` ($W - \partial W$)
 - `operatorU` / `choiState` — the emergent operator $U = \operatorname{unvec}(\ker L_1(W - \partial W))$ (deferred; see below)
 - `outputState` — the emergent final state $\psi_{AB}$: the inputs carried through the relaxed geometry, read over the output cycles

As well as any useful statistics about the convergence process. We should especially make note of topological parameters, 
and call out the observed topologies.

### Emergence modes (which quantity is primary)

`outputState` is populated in **both** modes — the primary emergent quantity when $U$ was supplied, and a
consistency read when the output was supplied. It is the $\#353$ flow: the minimum-norm metric $L_1(W)$ harmonic
matching the **input** periods, read (as periods) over the output cycles. Because the read carries *only* the
inputs, it is input-dominated and independent of the pinned target (read from the relaxed geometry, not echoed
from the seed — supplying a different output with the same inputs returns the same `outputState`). It is returned
unnormalised and up to a global phase (the period scale); normalise to recover the qubit amplitudes.

Note that on the current $(T^2 - 3\,\text{holes}) \times S^1$ topology the input-to-output transport is
$\approx$ identity, so `outputState` tracks the (transported) inputs and does **not** yet reflect $U$'s action.
Making the transport realise $U$ is the operator-as-bulk-constraint — the same interior-handle operator-topology
rework deferred for the operator read-out below.

The operator read-out $U = \operatorname{unvec}(\ker L_1(W - \partial W))$ is **deferred**. On the current
$(T^2 - 3\,\text{holes}) \times S^1$ topology $\ker L_1(W - \partial W)$ is a $(d^2 - 1)$-dimensional subspace of
the interior $1$-cochains, and there is no basis-independent map from it to the $d \times d$ operator: the kernel
basis is fixed only up to an $O(d^2 - 1)$ rotation, so a reshape is frame-dependent. A principled read needs
distinguished interior **Choi-cycles** the topology does not yet supply — the interior-handle operator-topology
rework (building $W$ from the closed $S^2 \times S^1$ handle so the operator is a genuine interior $1$-cycle).
Until then `operatorU` / `choiState` stay **empty** rather than report a frame-dependent value.

## Notes

### The Period Matrix
A period is the integral of a harmonic form around a closed cycle: its circulation (aka holonomy). If the 1-form is $ \omega $ and $ \gamma $ is the loop then it's $ \oint_{\gamma} \omega$.

In the context of simplicial complexes; the harmonic 1-form is a 1-cochain, $\psi$, which assigns a complex number $ \psi(e) $ to each edge and lives in $ \ker L_1 $.

A cycle is a closed loop of oriented edges.

The _period_ of $\psi$ over that cycle is the signed sum along the loop.

$$
\text{period} = \oint_\gamma \psi = \sum_{e \in \gamma} \pm \psi(e)
$$

The period only sees topology, not the specific loop you chose. It's the same for any two loops in a homology class. So the (harmonic form + cycle) maps to $b_1$ independent values in $ \mathbb{C} $

In our context the state, $ \psi$ IS its periods. $ \psi \in \ker L_1 $ with $b_1 = \operatorname{dim}(\psi) $ ($b_1$ is the $1^{th}$ betti number: holes). The harmonic space is $b_1$ dimensional and a harmonic form's coordinates in that space are its periods over the $b_1$ independent cycles. So "pinning a qubit state" is "forcing the carried harmonic to have these target periods over these cycles".

Each torus has two independent cycles; the meridian and the longitude. 

$H$ is the "harmonic matrix". i.e. `H = HodgeLaplacian.harmonicMatrix(1, ...)`, which is a `dim x n` table where $dim = b_1$ (number of harmonic 1-forms, n = number of edges). So each row, $r$, is one basis harmonic 1-form $ h_r \in \ker L_1$. Each column, $i$, is an edge. So $H [rn + i ] = h_r(e_i)$ is the complex number harmonic $r$ assigns to edge $e_i$, which is a measure of how much the harmonic "flows" along that edge.

The $\operatorname{sign}$ of an edge is the edge's orientation in its loop $\gamma$. The product $ \operatorname{sign} H[r, e]$ is one edge's signed contribution to harmonic $r$'s circulation. Summing over the loop is the discrete line integral:

$$
P[r,q] = \sum_{e \in \text{cycle }q} \text{sign}(e)\cdot h_r(e) = \oint_{\text{cycle }q} h_r
$$

So; rows = harmonics, columns = cycles, entry = the period of that harmonic over that cycle. That's the period matrix.

