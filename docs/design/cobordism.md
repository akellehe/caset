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

This should all be implemented in a C++ class, `MergeCobordism(inputStates, outputStates, U)`. If the user supplies $U$; 
the `outputStates` should be calculated. If not then we require `outputStates` and $U$ will be emergent. If the user 
passes $U$ we should use the algorithm above, ignoring $U$ except in the calculation of the final states used in the 
whole setup.

`MergeCobordism` should have several members used for later introspection and analysis. 
 - `inputStates`
 - `outputStates`
 - `cobordism` ($W$)
 - `boundary` ($\partial W$)
 - `bulk` ($W - \partial W$)

As well as any useful statistics about the convergence process. We should especially make note of topological parameters, 
and call out the observed topologies.

## Realized topology and initialization

The bulk $W_{AB}$ is realized as $(T^2 - 3\,\text{holes}) \times S^1$, i.e. $T^3$ with three $(\text{hole} \times S^1)$ solid tori removed.

- $\partial W = T^2_A \sqcup T^2_B \sqcup T^2_{AB}$: three tori, one per state, each with $b_1 = 2$ — the hole-circle and the $S^1$, the two cycles a qubit register carries.
- $\ker L_1(W - \partial W) = 3$ interior handles (the two genus cycles and the $S^1$): the operator, $\operatorname{dim}(U^{Choi}) = 3$. A genus-1 base is required — the interior count is $2g + h - 2$, which equals $3$ only for $g = 1, h = 3$; a sphere base ($g = 0$) gives $1$.

Initialization:

1. Build $T^2 = \operatorname{SimplicialProduct}(S^1, S^1)$, subdivide once (the minimal torus admits only two vertex-disjoint holes; one subdivision admits three), remove three vertex-disjoint faces, and staircase the remainder over $S^1$ (three layers, looped).
2. Seed the edge lengths slightly off uniform $\ell^2 = 1$. At $\ell^2 = 1$ a non-null metric-Hodge eigenvalue sits at zero and the period-residual gradient diverges; a small spread removes it. $\ell^2$ stays free to go negative — the causal type is emergent, not clamped.

The states stay pinned (all three, $\psi_A$, $\psi_B$ and $\psi_{AB}$); the operator is what emerges. The state residual reads periods over the torus cycles via $\operatorname{residualForLoops}$ over signed edge-loops, because the $S^1$ cycle is no triangle boundary. The six boundary cycles span $b_1(W) = 5$, so the joint read-out is over-determined ($6 > 5$) and its gradient is non-singular; a single torus's two cycles ($2 < 5$) would not be.

The search executes a batch of boundary-fixed Pachner moves (flip, inverse flip, shift, add — topology-preserving, so $\ker L_1(W - \partial W)$ is held) before each relaxation, keeping the lowest-residual operator: the moves change the triangulation and hence the operator, and the relaxed residual scores each.

