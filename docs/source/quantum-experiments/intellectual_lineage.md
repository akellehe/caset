# 4D Charged Cartan Monte Carlo: milestones and motivation

This is the through-line of the project: the sequence of ideas — each
from a piece of established physics — that took the construction from
"spacetime from entanglement" to a Monte Carlo with peak spectral
dimension approaching 4 asymptotically.

Each step replaces one element of a textbook physical model with a
geometric/information-theoretic analog while preserving the property
the textbook treatment used it for. The chain is meant to be read as a
sequence of *principled generalisations*, not as a stack of ad-hoc
choices.

## 1. Van Raamsdonk: spacetime from entanglement

The premise. Mark van Raamsdonk's 2010 essay ["Building up spacetime
with quantum entanglement"](https://arxiv.org/abs/1005.3035), and the
broader holography programme that contains it, observed that **the
entanglement between two regions of a quantum state acts like a metric
distance**: as you disentangle two regions, they become spatially
disconnected; as you entangle them more, they become "closer."
Spacetime, on this view, is built up *from* the pattern of quantum
correlations rather than being a stage on which quantum systems sit.

The construction in this project takes this literally:

- **Vertices** are quantum systems with their own density matrices.
- **Edge lengths** between vertices are computed from their mutual
  information `ℓ = −log(I / I_max)`. Strong correlation ↔ short edge;
  no correlation ↔ infinite edge.
- The simplicial complex of vertices, with these MI-derived lengths,
  *is* the emergent spacetime — there is no separate background metric.

This is the founding principle and remains intact through every
subsequent generalisation.

## 2. Mutual information instead of pure-state entanglement

Van Raamsdonk's original argument is cleanest for pure states, where
entanglement is the bipartite Schmidt content. Real quantum systems
mix. Mutual information

```
I(A : B) = S(A) + S(B) − S(AB)
```

reduces to twice the entanglement for pure states and generalises
cleanly to mixed states, so the construction uses MI throughout. This
also reads as: the *information* shared between two systems is the
geometric distance between them, regardless of whether that information
takes the form of pure entanglement, classical correlation, or any
mixture thereof.

## 3. Causal sets: events as the substrate

Spatial correlation alone gives us a *spatial* metric. To get
spacetime, we need temporal structure too. Sorkin's *causal-set*
programme treats individual events — discrete points in spacetime —
as the fundamental degrees of freedom, with a causal partial order
giving the temporal structure.

In our construction:

- Each **interaction event** (two systems applying a unitary `U`) is
  itself a vertex (the entangling core, what we'd later call `Σ_AB`).
- The worldline products of an interaction are also vertices,
  inheriting causally from the inputs.
- The simplicial complex grows event by event, with each event's
  products causally later than its inputs.

So a vertex is no longer just a "spatial" point; it's a spacetime
event, in the causal-set sense. Mutual information is generalised from
"between two co-existing systems" to "between two events at any
time-slice difference, related by their causal cone." Edge length =
−log(I) still defines distance, but now both spacelike and timelike
distances are encoded in the same object.

## 4. Schwinger model: a set of requirements to satisfy

The natural question is: which `U` to apply at each interaction
event? Rather than commit to a specific model, we read the Schwinger
model — 1+1D QED with massive fermions and a U(1) gauge field — as a
**set of requirements** that any per-event `U` should plausibly
satisfy. Its two-site Hamiltonian fragment

```
H_Schwinger = (1 / 4a) (XX + YY) + (m / 2) (−Z⊗I + I⊗Z) + Coulomb
```

(with the Coulomb term integrated out via Gauss's law in the original
theory) gave us a concrete first `U` to use, but the *requirements*
it embodied were the structurally important content:

1. **A specific 4×4 unitary** `U = exp(−i H · dt)` we can apply on
   each pair of inputs.
2. **A conserved U(1) charge** (staggered fermion number) that `U`
   commutes with — so charge is a real quantum number, not just a
   classical label.
3. **A gauge force** between charges (the Coulomb / σ^z σ^z term) —
   so distant charges feel each other through a propagator.
4. **C and CP as operator-level structures**, not numerical knobs —
   so charge conjugation and CP-violation become statements about
   how `U` interacts with `Q̂` and `Ĉ`, not opt-in parameters.
5. **An out-of-equilibrium growth process** consistent with
   Sakharov's third condition, in case we want to study
   matter-antimatter asymmetry mechanisms.

The original v0 implementation used the Schwinger `U` literally for
(1) and inherited (2) via the staggered fermion number. (3) was
dropped at the two-site fragment level. (4) and (5) were not
addressed in v0; v0.2's qudit basis made (4) operator-level, and
v0.3's gauge mediation aims to address (3).

By v0.2, the Schwinger `U` itself is no longer load-bearing — it
remains as a configurable option, but the *default* is a parametric
ququart-pair Hamiltonian whose constants `(J_c, J_s, δ_m, γ_CP)`
let us tune to satisfy each requirement independently. We treat
Schwinger as the historical anchor that gave us the requirements,
not as the dynamics the model is committed to.

See [from_schwinger_to_lattice.md](from_schwinger_to_lattice.md) for
the detailed argument about why each piece of the Schwinger framing
matters or doesn't.

## 5. The qudit basis: charge intrinsic to the state

In the original implementation each vertex carried a *qubit* state
(2-dim Hilbert), with charge as an external classical label
(`chargeOf_[v]`). This worked but left charge as a separate register
not derivable from the quantum state — meaning charge conjugation,
CP-violation, and other operator-level statements about charge were
imposed via numerical parameters (like a `cpBias` knob), not derived
from the Hilbert-space structure.

The v0.2 move was to promote each vertex to a **4-dim ququart** with a
Dirac-spinor-like basis:

```
{ |+, 0⟩, |+, 1⟩, |−, 0⟩, |−, 1⟩ }
   ↑                          ↑
charge sector            spin / internal
```

with the charge operator `Q̂ = diag(+1, +1, −1, −1)`. Charge is now
*intrinsic to the state*: `⟨q⟩_v = Tr[ρ_v · Q̂]`. Continuous
expectation values arise naturally from mixed states.

This generalisation cost us a property the qubit basis had for free:
the **Cartan / KAK decomposition** of SU(4), which cleanly factors any
4×4 unitary into "local frames × entangling core × local frames"
form. SU(16) has no equally clean canonical decomposition.

## 6. Choi isomorphism: a quantum state for the entangling core

The Cartan core in SU(4) gave us a principled construction for the
`Σ_AB` (entangling-core) vertex's state. In v0.2 we lost that
construction and need an analog for SU(16) interactions on the qudit
basis.

The **Choi-Jamiolkowski isomorphism** is the natural replacement: for
any unitary `U`, the Choi state

```
J(U) = (U ⊗ I) |Ω⟩⟨Ω| (U† ⊗ I)
```

(with `|Ω⟩` the maximally entangled state on the doubled Hilbert)
encodes everything about `U` as a pure quantum state on a larger
space. The Choi state is the principled generalisation of the SU(4)
Cartan core to any 4×4 or 16×16 unitary — it carries the entangling
content of `U` as a state we can put on the Σ_AB vertex.

> **Claim to defend later.** "Cartan core" is the right name here
> because the property that makes the SU(4) Cartan core meaningful —
> being the *non-local* part of `U` that cannot be reduced to local
> rotations on either side — also applies to the Choi-isomorphized
> entanglement core. Concretely: any local-only unitary `U = K_A ⊗
> K_B` has operator-Schmidt rank 1, so its Choi state factors across
> the bipartite cut and carries no genuine entanglement between A
> and B. Any genuinely entangling `U` has operator-Schmidt rank > 1
> (the σ-spectrum from the operator Schmidt decomposition; in SU(4)
> this is exactly the `(c_x, c_y, c_z)` Cartan-core parameters
> reorganised). The Choi state is the unique pure quantum state that
> captures this *non-local* content for arbitrary `U`. A proper
> defense — showing the operator Schmidt rank ≥ 2 condition is
> equivalent to non-locality in the same sense the Cartan core
> captures — should appear in a dedicated note in v0.3 or later.

In v0.2 we use a placeholder (the maximally-mixed `I/4`) for the
Σ_AB marginal, which is what the [finite-size
investigation](v02_finite_size_investigation.md) surfaces as a real
inconsistency. The principled fix — Σ_AB carrying the actual Choi
state of `U` — is [GitHub issue
#16](https://github.com/akellehe/tessera/issues/16).

## 7. Σ_AB as a photon analog

Once Σ_AB carries the Choi state of `U`, it becomes a quantum object
sitting between the two worldline-continuation vertices (`U_A`, `V_B`
— or `xp`, `yp` in code), and inheriting joint correlations with each
of them. This is exactly the role a **virtual photon** plays in QED:
it sits between two charged particles, carries the gauge interaction
between them, and has no independent quantum identity except through
its joint correlations.

Two consequences emerge from this reading:

- **Photon emission on annihilation** is mechanistically natural: when
  a charged pair annihilates, the Σ_AB-like quantum becomes a free
  photon vertex carrying away the joint content. Already implemented
  as `featurePhotonOnAnnihilate`.
- **Gauge force as photon-mediated correlation** is what
  [v0.3](charged_cartan_monte_carlo_v0.3.md) implements: between two
  charged vertices `i`, `j`, the *existence of photon paths in the
  graph* mediates a Coulomb-like amplitude that conditions their
  next interaction's input state.

## 7.25. Post-selection and worldline creation: TSVF inspiration

The mechanism by which Σ_AB's quantum content is reduced to 4 dim
when it later participates in an interaction — *projection onto a
charge-basis subspace matched to the partner vertex* — is a
**post-selection** step in the sense of the Two-State Vector
Formalism (TSVF) of Aharonov, Vaidman, and collaborators. In TSVF
the quantum state of a system is conditioned on both its past
(prepared) state and its future (post-selected) state; the
"effective" state at any intermediate time is computed using both.

In our construction:

- The Σ_AB vertex's "past state" is the Choi state `J(U)` of the
  interaction that created it.
- Its "future state" is fixed by the partner it eventually
  encounters at consumption time — specifically, the partner's
  charge sector selects which subspace of Σ_AB's 256-dim Hilbert is
  realised.
- The "effective" 4-dim ρ that we feed into the next interaction is
  this post-selection: the reduction of `J(U)` conditioned on the
  partner's charge content.

This isn't just an analogy — it's the same mathematical operation.
TSVF gave us the framing: rather than treating the Σ_AB reduction as
an arbitrary contraction choice, view it as a *measurement-like
post-selection* whose outcome is decided by the partner.

Closely related: **Avshalom Elitzur**'s interpretation of how
worldlines are created and removed at interaction events in TSVF
inspired the choice of when worldlines terminate and begin in our
construction. Specifically:

- **Worldlines terminate at consumption.** When a vertex is consumed
  as input to a new interaction, its worldline ends — the next slice
  carries different vertices (the products). In TSVF terms, the
  vertex's two-state vector "closes" at the consumption event.
- **Worldlines begin at creation.** Every product of an interaction
  begins a new worldline, with the Choi-state / marginal content
  determined by the interaction's joint output. In TSVF terms, the
  product carries a fresh past-state vector.
- **Annihilation deactivates both worldlines.** Under
  `featureDeactivateOnAnnihilate`, the matched pair vanishes from the
  frontier — both two-state vectors close simultaneously. The
  optional photon vertex begins a new worldline with no charge,
  inheriting only the released energy budget.

We're using these formalisms as *inspiration* — adopting the
worldline-termination semantics and the partner-conditioned
projection picture — without committing to TSVF as a complete
interpretational framework. The Monte Carlo dynamics doesn't care
which interpretation of quantum mechanics we hold; what TSVF gave
us is a coherent *language* for describing the projections and
worldline boundaries that already had to happen in our model.

## 7.5. Worldline discontinuity and the 4 / 16 / 256 dimensional ladder

The qudit-basis + Choi-state machinery from steps 5–7 has a natural
hierarchy of Hilbert-space dimensions:

| dim | object | when it appears |
|---:|---|---|
| **4** | per-vertex qudit state `ρ_v ∈ ℂ^{4×4}` | frontier vertices, products, marginals |
| **16** | bipartite joint `ρ_{XY} ∈ ℂ^{16×16}` | input to an interaction; output of an interaction; stored joint correlations |
| **256** | Choi state `J(U) ∈ ℂ^{256×256}` | the Σ_AB entangling-core vertex, encoding the full quantum content of U |

A worldline traces a path through this ladder:

- **Birth.** A new vertex (either from the initial layer, from
  `pairCreate`, or as a product of an interact) starts as a 4-dim
  state on the frontier.
- **Interaction (input side).** Two frontier vertices are picked.
  Their 4-dim states (or stored 16-dim joint, if they share one)
  enter a 16-dim joint, U is applied (16×16), and three new 4-dim
  products are extracted. The two *input* worldlines **terminate
  here** — their identity is consumed by the interaction. The
  worldline that "continues" from each input is its product
  worldline (`xp` for `x`, `yp` for `y`), and the entangling-core
  product (`ab`) is a new worldline whose state is the Choi state
  of U (under the principled v0.2/v0.3 treatment).
- **Annihilation.** Under `featureDeactivateOnAnnihilate` (v0.1.B), a
  matched `(+, −)` pair is removed from the frontier — both worldlines
  terminate, no continuation products. In the qudit basis (the
  v0.2-style upgrade to this move, issue
  [#13](https://github.com/akellehe/tessera/issues/13)), the
  bipartite state of the pair is *projected* onto the `Q_total = 0`
  subspace of the 16-dim joint Hilbert (an 8-dim subspace spanned by
  the `q = +1, q = −1` cross-charge basis states). With
  `featurePhotonOnAnnihilate` (v0.1.iii), a neutral "photon" vertex
  is spawned to carry the projected joint's quantum content forward
  — a new worldline begins after the projection.
- **Σ_AB consumption (the Compton-scattering analog).** When a
  Σ_AB vertex (photon-like) is later picked as input to a new
  interaction alongside a charged worldline-vertex, its 256-dim Choi
  state is reduced to a 4-dim qudit via charge-basis projection
  (option A from the v0.2 design), and the standard interact machinery
  runs on the resulting 4×4 joint. The Σ_AB worldline *terminates*
  at this consumption (it's an input), and a *new* Σ_AB'-worldline
  begins as the entangling-core product of the new bowtie — a fresh
  Choi state from the new U.

That last bullet is structurally analogous to Compton scattering: an
incoming photon interacts with a charged particle and an outgoing
photon emerges, with the electron's worldline continuing through the
interaction vertex. In our construction:

- The incoming Σ_AB plays the role of the incoming photon.
- The charged worldline vertex plays the role of the electron.
- The interaction event produces an outgoing Σ_AB' (the outgoing
  photon) and an outgoing charged worldline product (the scattered
  electron).
- The "charge-basis projection" that reduces the incoming Choi state
  to 4-dim corresponds to the photon being "measured" in the charge
  basis defined by the partner — exactly how a real Compton vertex
  selects the photon's polarisation / momentum content along the
  electron's reference frame.

**Tentative.** Whether this analog is exact at the cross-section /
amplitude level is open. The *worldline topology* (one photon-in +
one electron-in → one photon-out + one electron-out, with the
charged worldline continuing through and the photon discontinuous in
identity but continuous in graph-connectivity) matches. The
*dynamics* depend on the specific 16×16 U we choose and how the
charge-basis projection actually selects from the Choi state. A
quantitative check against the Klein-Nishina formula or its low-energy
Thomson limit is a v0.3+ investigation, not currently part of the
construction's design goals.

This worldline-discontinuity picture is the v0.2/v0.3 model's
analogue of Feynman-diagram vertices: the *graph structure* is a
spacetime diagram with charged worldlines (continuous through
multiple events) and photon worldlines (typically short-lived, each
spawning at one event and terminating at the next).

## 8. Spectral dimension as the observable: Loll et al.'s CDT line

What observable do we use to test whether the construction emergent a
*physical* (4-dimensional) spacetime? The Causal Dynamical
Triangulations (CDT) programme of Loll, Ambjørn, and collaborators
uses the **heat-kernel spectral dimension** `D_S(σ)` — the dimension
seen by a random walker diffusing on the lattice for time `σ` — as
the diagnostic for whether their path-integral over geometries
produces 4D macroscopic spacetime.

We borrow exactly this observable. On our MI-weighted simplicial
complex, the heat kernel `P(σ) = (1/N) Σ_v ⟨v| exp(−σ L) |v⟩` gives
us `D_S(σ) = −2 d log P / d log σ`. The H_DS4 hypothesis is: there
exists a regime where this `D_S` reaches 4, signalling an emergent
3+1-dimensional macroscopic phase.

The v0.2 β-scan ([writeup](charged_cartan_v02_beta_scan_writeup.md))
finds a stable plateau at peak `D_S ≈ 4.6 ± 0.1` across a decade of
`β`. The [finite-size
investigation](v02_finite_size_investigation.md) then shows that this
plateau drifts *toward 4* as the lattice grows (T-scaling), with
naive geometric extrapolation projecting to D_S(T → ∞) ≈ 4.07 — i.e.,
**the H_DS4 target is plausibly reached in the asymptotic limit**.

## 9. Where we are now

The current state, as a sequence of substituted ingredients:

| Schwinger model has | We use |
|---|---|
| 1+1D lattice of fermions | MI-weighted simplicial complex of events |
| Wavefunction | Density matrices + joint states |
| `U(1)` charge as fermion number | `Q̂ = diag(+1,+1,-1,-1)` on a 4-dim qudit |
| Two-site `U = exp(−i H dt)` | 16×16 `U` from a parametric pair Hamiltonian |
| Coulomb / gauge term | Photon-mediated input conditioning (v0.3, in design) |
| Hilbert-space partition function | Regge-action Monte Carlo with `e^{−βS}` weight |
| Continuum limit | T → ∞ asymptotic plateau |

The asymptotic plateau is approaching 4 in the right direction. The
remaining work — the [v0.3
milestone](https://github.com/akellehe/tessera/milestone/2) — is
about restoring the gauge/Coulomb piece geometrically and seeing
whether the plateau lands cleanly at 4 with those couplings in place.

## See also

- [van Raamsdonk 2010, "Building up spacetime with quantum
  entanglement"](https://arxiv.org/abs/1005.3035) — the holography
  premise.
- Sorkin's [causal-set
  programme](https://arxiv.org/abs/gr-qc/0309009) — events as the
  substrate.
- Ambjørn-Jurkiewicz-Loll [CDT
  reviews](https://arxiv.org/abs/1203.3591) — spectral dimension as
  the emergent-4D diagnostic.
- The construction's own design notes:
  [v0.1](charged_cartan_monte_carlo_v0.1.md),
  [v0.2](charged_cartan_monte_carlo_v0.2.md),
  [v0.3](charged_cartan_monte_carlo_v0.3.md).
- [Why Schwinger isn't load-bearing](from_schwinger_to_lattice.md).
