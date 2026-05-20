# From the Schwinger model to an abstract lattice construction

A standalone justification for why the current line of work has dropped
every load-bearing aspect of the Schwinger fermion-field construction
and what we keep / lose by doing so. Read this alongside
[../earlier-work/interaction_history_monte_carlo.md](../earlier-work/interaction_history_monte_carlo.md)
and [../charged-cartan/design/v0.1.md](../charged-cartan/design/v0.1.md).

## The starting point

The original `InteractionSimulation` line used the Schwinger
two-site Hamiltonian to drive each interaction:

$$
\begin{aligned}
H_{XY} &= \frac{1}{4a}\,(XX + YY) + \frac{m}{2}\,(-Z \otimes I + I \otimes Z), \\
U &= \exp(-i\, H_{XY}\, dt).
\end{aligned}
$$

This was the 1+1D massive Schwinger model in its staggered-fermion +
Jordan-Wigner form, restricted to a single two-site fragment. Choosing
it had two practical advantages and two structural ones:

- **Practical, concrete**: a textbook QFT we could write down without
  inventing physics ourselves; clean parameters ($a$, $m$, $dt$) with
  established meaning; a well-documented $g \to 0$ (strong-coupling)
  limit where confinement and meson-like bound states appear.
- **Structural**: the Hamiltonian commutes with the staggered fermion
  number $Q$ per site-pair, so charge is a real conserved $U(1)$
  quantum number rather than a bookkeeping label.

## Why we kept it at first

The motivating spec mentioned the Schwinger model as the candidate
two-site dynamics because we wanted a *specific* $U$ whose Cartan core
$(c_x, c_y, c_z)$ corresponded to something physical. The decision was
"use Schwinger now, scan over Hamiltonian parameters later" — i.e.,
treat the Schwinger choice as a baseline rather than as the central
claim. Even in the original spec, the line of work was always going to
be about whether the *Cartan-decomposed entangler picture* generates
emergent dimension, not about whether the Schwinger model in
particular does.

## What the Schwinger truncation was actually doing

Three things, ordered by how essential they were to the experiment:

1. **Providing a $4 \times 4$ unitary $U$** to apply to each pair of
   inputs.
2. **Carrying a conserved $U(1)$ charge** (fermion number) that the
   two-site $U$ commutes with.
3. **Connecting the construction to a textbook 1+1D QFT** so reviewers
   could check the physics interpretation.

Of these, only (1) was load-bearing in the actual computation. Every
edge MI, every cell-action contribution, every charge bookkeeping
step works the same way for *any* $4 \times 4$ unitary — the Schwinger
choice just fixed a specific point in the $SU(4)$ parameter space.

## What we already truncated by going two-site

The Schwinger model in 1+1D has a **non-local Coulomb term** $L_n^2$
(the gauge-field electric energy) that, after Gauss's law is solved,
becomes a long-range $\sigma^z_i \sigma^z_j$ interaction between
fermion sites weighted by $|i - j|$. This term:

- can't be written as a two-site fragment (it's non-local by
  construction), so we dropped it the moment we chose a per-pair
  unitary,
- carries the gauge / "photon" content of the theory,
- conserves energy and momentum during charge interactions because
  it's the channel through which energy is exchanged.

So the moment the original spec restricted to a two-site Hamiltonian
fragment, the Schwinger model was *already* truncated to "fermion
hopping + staggered mass, no gauge dynamics." That truncation was
necessary to keep $U$ local to a pair, but it dropped the energy
conservation that the full theory has during pair annihilation.

The "Schwinger model" label on the original construction was therefore
already misleading: it was running a *two-site fragment* of the
Hamiltonian, not the full model. The choice of fragment didn't change
the geometric content of the experiment; it just gave us a specific
parameterised entangler.

## What we drop in v0.1 (and why it doesn't cost anything)

**The Schwinger Hamiltonian as the default $U$.** Replaced (still
configurable) with a fixed canonical Cartan entangler

$$
U_{\mathrm{Cartan}} = \exp\!\bigl(i\,(c_x\, XX + c_y\, YY + c_z\, ZZ)\bigr).
$$

The Cartan decomposition exists for any $4 \times 4$ unitary, so this
is just choosing a representative point in $SU(4)$ without the
appearance of physical baggage. We keep the Schwinger $U$ as an
alternative option (`cfg.unitaryMode = SCHWINGER`) so backward
compatibility is preserved.

**The Schwinger fermion-number as the charge source.** In v0.1 the
charge is carried as a *separate classical label* `chargeOf_[v]`. It
is no longer derived from the state. We enforce conservation through
the move rules (annihilate / pairCreate) and the eligibility filter
(opposite-sign pairs don't interact, they annihilate). This is a
much weaker statement than "$U$ commutes with the charge operator" —
it's an *imposed* conservation law rather than a *derived* one — but
it's exactly as physical for our purposes, because:

- charge being "real" in our framework means the dynamics conserve
  it, which it does;
- the cost of carrying it as a separate label is that operations that
  modify charge without modifying state (like our partial-annihilation
  move) need the label and the state to stay consistent, which we
  achieve by convention rather than by Hamiltonian symmetry;
- v0.2 promotes charge to be intrinsic to the state by using a
  4-dim qudit basis
  $\{\lvert{+}0\rangle,\, \lvert{+}1\rangle,\, \lvert{-}0\rangle,\, \lvert{-}1\rangle\}$
  with $\hat{Q} = \operatorname{diag}(+1, +1, -1, -1)$, recovering the
  operator-level statement.

**Energy conservation during annihilation.** The Schwinger model
preserved energy because the Coulomb term carried it. Our truncated
two-site $U$ already lost the Coulomb term, so even the original
construction wasn't energy-conserving across annihilation events.
v0.1's `annihilate` move (and v0.1.B/iii variants) makes this
explicit: annihilation produces no energy-carrying photon by default,
or optionally produces a neutral "photon" vertex (the
`featurePhotonOnAnnihilate` flag) that lives on the frontier but
doesn't carry quantitative energy/momentum content.

**The "Schwinger label" on the construction.** The construction in
v0.1 is honest about being "a discrete graph of MI-coupled qubits
with U(1) charge conservation," not "a simulation of the Schwinger
model." This isn't a loss; it's a more accurate description of what
we always had.

## What we keep

The objects that genuinely drive the experiment:

- **Two-site interaction unitaries on density-matrix inputs.** Every
  interaction is
  $\rho_{AB} = U\,(\rho_A \otimes \rho_B)\,U^\dagger$
  with $U$ a fixed $4 \times 4$ unitary. The choice of $U$
  parameterises the $SU(4)$ Cartan core but doesn't change the
  structural content of the experiment.
- **The Cartan / KAK decomposition**

  $$
  U = (K_1 \otimes K_2)\,\exp\!\bigl(i\,\vec{c}\cdot\vec{\sigma}\vec{\sigma}\bigr)\,(K_3 \otimes K_4)
  $$

  as the principled way to split per-cell physics into "local-frame
  carriers" (the $U_A$, $V_B$ worldline products) and "entangling
  core" (the $\Sigma_{AB}$ joint-state carrier).
- **Mutual-information-derived edge lengths**
  $\ell = -\log(I / I_{\max})$ as the way to lift quantum correlation
  onto geometric edges of a simplicial complex.
- **Regge action on hinges of the resulting complex** as the dynamical
  weight, with Monte Carlo sampling at inverse temperature $\beta$.
- **Charge as a conserved quantum number** that organizes the move
  set (same-sign pairs → bowtie; opposite-sign pairs → annihilation).
  The conservation is enforced by the move rules; whether the
  underlying $U$ makes this a derived theorem (Schwinger) or an
  imposed convention (v0.1) does not change the geometry.

## Why this is the right framing for the $H_{D_S=4}$ question

The hypothesis $H_{D_S=4}$ — "is there a $\beta$ at which the
heat-kernel spectral dimension of the interaction-history complex
reaches 4?" — is a *geometric* question. It depends on:

- what cells are eligible to form,
- what the action weight on each candidate cell is,
- how cells share vertices (or faces) with their neighbours,
- the resulting graph topology and edge weights.

None of these depend on whether $U$ is the Schwinger Hamiltonian, a
Cartan-canonical entangler, or any other $SU(4)$ element. They all
depend on the *structure* of how interactions chain together.

The Schwinger framing was load-bearing for the *physical
interpretation* (we could talk about fermions and gauge fields), but
it was scaffolding for the *experiment* (the structural question is
about graphs and actions). Dropping it makes the construction more
honest and more reusable, without changing the content of the
$H_{D_S=4}$ test.

## Glossary: what the current construction "is"

**A construction.** A simplicial-complex Monte Carlo where each cell
records a unitary interaction event between two prior systems, with
edge weights computed from the resulting mutual informations, weighted
by an Euclidean Regge action, and sampled at inverse temperature
$\beta$. Charges are a separately-tracked $U(1)$ quantum number,
conserved by the move rules.

**Not.** A simulation of the Schwinger model, the Standard Model, or
any specific QFT. It's a discrete geometric model that *could* be a
toy version of various theories depending on choices we haven't
committed to yet.

## See also

- [interaction-history-monte-carlo.md](../../interaction-history-monte-carlo.md)
  — the construction's charter (still uses Schwinger language; will be
  updated to match this note in due course).
- [../charged-cartan/design/v0.1.md](../charged-cartan/design/v0.1.md)
  — the v0.1 design including the configurable $U$ modes.
- [../earlier-work/interaction_history_monte_carlo.md](../earlier-work/interaction_history_monte_carlo.md)
  — the original experiment's results, run under the Schwinger $U$
  default.
