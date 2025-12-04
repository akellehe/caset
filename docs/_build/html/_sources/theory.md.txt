# Theory

## Why the Lorentzian Metric is Important

Lorentzian path integrals tend to integrate over an infinite domain, so they tend to "blow up". Euclidean path integrals 
are easier to solve, but they obscure the causal structure of spacetime.

In Euclidean spacetimes, all points are effectively spacelike separated. Unitarity/probability conservation is tied to 
the Lorentzian structure. 

## Path Integrals and Wick Rotations

The path integral formulation of quantum mechanics, introduced by Richard Feynman, lets us compute the probability 
amplitude for a system to transition from one state to another. It's calculated by summing over all paths the system can 
take, weighted by the exponential of the action (in units of $ \hbar $). The path integral is the foundation for lattice 
gauge theory as well as quantum chromodynamics.

Mathematically, this is expressed as:

$$
\langle x_f, t_f | x_i, t_i \rangle = \int \mathcal{D}[x(t)] e^{\frac{i}{\hbar} S[x(t)]} 
$$

which can be thought of as the "sum over all possible histories".

In field theory the $ x(t) $ becomes a field $ \phi(x) $ and the action $ S[x(t)] $ becomes the action functional 
$ S[\phi] $, leading to

$$
Z = \int \mathcal{D}[\phi] e^{\frac{i}{\hbar} S[\phi]}
$$

where $ Z $ is the partition function, encoding all the dynamics of the quantum field theory.

This is the Lorentzian path integral, because $ S[\phi] $ is computed with the Lorentzian metric, $ g_{\mu\nu} $.

The exponential term in the integral oscillates rapidly for large actions. It's not typically convergent. To make sense 
of this integral, people use a Wick rotation to go from Lorentzian to Euclidean signature:

$$
t \rightarrow -i \tau
$$

which makes the term 

$$
e^{iS} \rightarrow e^{-S_E}
$$

where $ S_E $ is the Euclidean action. Now it acts like a statistical partition function that converges. This new form is
no longer timelike, though. It is a mathematical trick to make the integral manageable, but it obscures the causal 
structure of spacetime.

There is no known general mapping to go to and from Euclidean and Lorentzian metrics, or even equivalence classes of 
diffeomorphic metrics.

## Quantum Gravity

The Lorentzian path integral is particularly important in quantum gravity, where the goal is to quantize the 
gravitational field. In this context, the path integral takes the form:

$$
Z = \int \frac{\mathcal{D}[g_{\mu\nu}]}{Diff(M)} e^{\frac{i}{\hbar} S_{EH}[g_{\mu\nu}]}
$$

Where $ S_{EH} $ is the Einstein-Hilbert action, given by

$$
S_{EH} = \frac{1}{16 \pi G} \int d^4x \sqrt{-g} (R - 2 \Lambda)
$$

This is the integral over all Lorentzian metrics under the equivalence class that differ by a 
diffeomorphism i.e. they are frame independent (since diffeomorphisms allow us to switch between frames). Another common 
way of saying that which is pretty cryptic is "The integral over all Lorentzian geometries modulo diffeomorphisms." This 
is Feynman's "sum over all spacetimes". It requires regularization or contour deformation to make sense of it, though.

## Lattice Gravity Approaches

### Lorentzian Spin Foams

### Loop Quantum Gravity on Graphs

### Picard-Lefschetz Theory

### Regge Calculus

Regge Calculus allows for edge lengths to vary, providing a discrete approximation to General Relativity. In this 
framework, spacetime is represented as a simplicial complex, where curvature is concentrated on the hinges (the 
(D-2)-dimensional simplices). The Einstein-Hilbert action can be expressed in terms of the deficit angles around these 
hinges, leading to a discrete version of the gravitational action.

### CDT (Causal Dynamical Triangulations)

Causal Dynamical Triangulations (CDT) is a non-perturbative approach to quantum gravity that constructs spacetime from 
causally ordered simplices. In CDT, the path integral over geometries is approximated by approximately summing over all 
possible causal triangulations with a MCMC algorithm, ensuring that the causal structure of spacetime is preserved. This 
approach has shown promise in recovering classical spacetime at large scales while incorporating quantum effects at small 
scales.

Note that CDT allows for imaginary edge lengths. This is handled by allowing for spacelike squared edge lengths to be 
some constant $ a^2 $ and time like squared edge lengths to be $ -\alpha a^2 $ for some constant $ \alpha > 0 $. This 
allows for a Wick rotation, mapping [[1]](resources/1905.08669v1.pdf)

$$
l_t^2 = -\alpha a^2 \rightarrow l_t^2 = +\alpha a^2
$$

### Causal Sets

Causal Set Theory posits that spacetime is fundamentally discrete, composed of a set of events with a partial order 
representing causal relationships. In this framework, the geometry of spacetime emerges from the underlying causal 
structure. Causal sets are locally finite, meaning that between any two events, there are a finite number of 
intermediate events. This discreteness leads to a natural ultraviolet cutoff, potentially resolving issues in quantum
gravity.

## Connections Between Theories

These three approaches to quantum gravity share a common theme of discretizing spacetime, albeit in different ways.

- Regge Calculus focuses on varying edge lengths in a simplicial complex, capturing curvature through deficit angles.
- CDT emphasizes the causal structure of spacetime by constructing it from causally ordered simplices.
- Causal Set Theory abstracts spacetime to a set of events with causal relationships, highlighting the fundamental discreteness of spacetime.

## Incompatibilities

While these theories share the goal of discretizing spacetime, they differ in their foundational assumptions and
methodologies. Regge Calculus and CDT both utilize simplicial complexes, but CDT imposes a strict causal structure,
which is not a requirement in Regge Calculus. 

Regge Calculus allows for continuously defined edge lengths, while CDT fixes them. 
Regge Calculus allows for arbitrary triangulations, whereas CDT restricts to those that preserve causality.
CDT allows for re-drawing of triangulation by a set of defined rules that preserve causality, while Regge Calculus does 
not have such freedom.

Causal Set Theory, on the other hand, does not rely on a simplicial complex representation, instead focusing on the 
causal relationships between discrete events. These differences lead to distinct mathematical frameworks and physical 
interpretations, making direct comparisons and integrations between the theories challenging.

Regge Calculus and CDT both sidestep coordinates by discretizing spacetime geometrically instead of analytically. That 
means the structure of spacetime is purely geometric. This gains the benefit of removing diffeomorphism redundancy.

Two triangulations that differ by the vertex labels are considered the same geometry in CDT. The path integral then 
becomes a sum over inequivalent triangulations instead of an integral over $ g_{\mu\nu} $ modulo diffeomorphisms.

### Path Integrals 

So CDT's path integral is (not Wick rotated):

$$
Z = \sum_{T \in \text{Causal Triangulations}} \frac{1}{C(T)} e^{i S_{Regge}(T)}
$$

where $ C(T) $ is a symmetry factor for the triangulation $ T $, and $ S_{Regge}(T) $ is the Regge action, the discrete 
version of the Einstein-Hilbert action.

In Quantum Regge Calculus; you replace the continuum gravitational path integral

$$
Z = \int_{Lor(M)/Diff(M)} \mathcal{D}[g_{\mu\nu}] e^{i S_{EH}[g_{\mu\nu}]\hbar}
$$

with the discrete analog integrating over edge lengths $ l_{ij} $ for all triangulations, $ \mathcal{T} $, of the 
manifold $ M $:

$$
Z = \sum_{T \in \mathcal{T}} \int_{l_{ij} > 0} \prod_{i < j \in T} dl_{ij} \mu(l_{ij}) e^{i S_{Regge}([T,l_{ij}])/\hbar}
$$

where $\mu(l_{ij}) $ is a measure factor for the edge lengths.

### Phase Transitions

The big difference is that CDT fixes edge lengths and only sums over _causal_ triangulations, which are enforced via 
(time) foliation. They both implement a sum over possible geometries, but CDT restricts the geometries to those that 
preserve causality. In either case this provides a partition function in the statistical mechanics sense.

Note that a phase transition occurs when changing a parameter (e.g. temperature, pressure) results in a qualitative 
change in the system's structure or large scale behavior. In CDT, varying the coupling constants in the Regge action can 
lead to different phases of spacetime geometry. For example, one phase may exhibit a well-defined four-dimensional 
spacetime, while another phase may lead to a crumpled or degenerate geometry. Identifying and understanding these phase 
transitions is crucial for exploring the continuum limit of quantum gravity theories.

### Observables 

In Regge Calculus curvature is calculated via deficit angle. In CDT observables are defined across the ensemble of
triangulations. The spatial volume as a function of discrete proper time encodes the effective curvature radius of 
spacetime. 

$$
V_3(t) = the number of 3 simplices in time slice t
$$

the ensemble average is

$$
\braket{V_3(t)} = \frac{1}{Z} \sum_{T \in \text{Causal Triangulations}} V_3(t)_T e^{i S_{Regge}(T) / \hbar}
$$
 
which fits the classical Euclidean de Sitter solution in 4D:

$$
V_3(t) \propto \cos^3\left(\frac{t}{R}\right)
$$

Another approach is with the spectral dimension $D_S(\sigma)$. A diffusion process on triangulated spacetime. A random 
walk with diffusion time $\sigma$ returns a probability $P(\sigma)$ of returning to the starting point that decays with
diffusion time. This is a measure of the effective dimension of spacetime at different scales. I think it also measures 
the holonomy of the triangulation within that region. It is scale dependent. Both of these are described in the 
'Observables' section of ["Quantum Gravity from Causal Dynamical Triangulations: A Review" by R. Loll](http://arxiv.org/abs/1905.08669).

Another approach is Geodesic distance distributions. You measure the volume of a geodesic ball.

## Improvements

 - We should consider how to represent light-like edges.
 - We should decide if there's a more computationally efficient method for solving Regge Calculus
 - We should consider how to represent topology change/gluing/rewriting/retriangulating rules.
 - Implement Chain and Co-Chain definitions
 - Optimize the CDT Markov Chain Monte Carlo algorithms
