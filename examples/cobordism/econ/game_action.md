# A game-theoretic action for the economic cobordism — analysis

> Exploratory theory note for tessera#602, extending gauge_dictionary.md
> §10. Analysis and a concrete candidate definition; no implementation
> claims. Not the official research track.

## 1. The central observation: the machinery is already equilibrium-seeking

The proton builds do not minimize the Regge action. Their outer
objective,

    F = ‖∇S‖² + Γ · Σᵢ r_periods(boundaryᵢ),

drives the **gradient norm** to zero: it searches for *stationary
points* of S subject to the boundary residuals. A stationary point of an
action whose variables are owned by many agents is precisely a **Nash
condition** — no profitable local deviation. So the leap to game theory
is smaller than it looks: swapping the action swaps *which game's
equilibria* the build finds. The machinery's semantics (pin boundaries,
relax to stationarity, floor = no stationary configuration exists,
surgery = discrete move that restores existence) transfer verbatim from
"physics extremal" to "economic equilibrium."

## 2. S_econ is already a game — recognize it, then generalize

The conductance energy Σ_e f_e²/w_e is the **Beckmann potential** of a
congestion game with linear marginal cost f/w: Wardrop equilibria are
exactly its constrained minimizers ([Beckmann et al. 1956; survey](https://theory.stanford.edu/~tim/papers/rg.pdf)).
[Rosenthal (1973)](https://en.wikipedia.org/wiki/Congestion_game) proved
every congestion game admits an exact potential;
[Monderer–Shapley (1996)](https://en.wikipedia.org/wiki/Potential_game)
proved the converse — exact potential games *are* congestion games. The
current spike action is therefore the special case: pure costs, no
benefits, linear marginal cost, no topology moves. The IPF/RAS null —
already proven to be the constrained minimizer of this energy under
pinned margins — is the *zero-benefit Wardrop equilibrium* of the
economy's congestion game. The proposal generalizes within the same
mathematical category, which is why nothing downstream breaks.

## 3. The candidate action

For a bulk flow history f and a topology change set G (opened and
severed relationships):

    S_game[f, G] = Σ_e ∫₀^{f_e} c_e(x) dx        (transaction costs)
                 − Σ_v B_v(y_v(f))               (agent benefits)
                 + Σ_{e ∈ G⁺} κ⁺_e + Σ_{e ∈ G⁻} κ⁻_e   (formation / severance)

with the following commitments:

- **Costs** c_e(x): marginal cost of routing x dollars through
  relationship e. c_e(x) = 2x/w_e recovers S_econ exactly; the general
  form is any increasing c_e (convexity of the integral keeps the
  potential-game structure and single-valued Wardrop selection).
- **Benefits** B_v: concave benefit at vertex v as a function of its
  **gauge-invariant throughput** (net receipts / charge). Restricting
  B_v to gauge-invariant arguments preserves the dictionary: gradient
  re-potentialing and netting moves change no agent's payoff. The data
  supplies calibration targets: the value-added rows are each industry's
  realized net benefit; final-demand columns proxy the closure sectors'.
- **Fixed costs** κ±: the [Jackson–Wolinsky (1996)](https://web.stanford.edu/~jacksonm/netsurv.pdf)
  network-formation term. Surgery acceptance becomes a
  **pairwise-stability test**: open (sever) a relationship when the two
  endpoints' joint benefit differential exceeds (falls below) the fixed
  cost. Known caveat to carry into interpretation: pairwise-stable
  networks need not be efficient and vice versa — the certified
  structural change is an equilibrium statement, not a welfare one.

**Calibration by inverse equilibrium — the game-theoretic anchor.** The
observed year-t economy is, by hypothesis, an equilibrium of whatever
game is being played, so the calibrated action must make f_t stationary:
marginal cost equals marginal benefit at the observed flows. This pins
every linear coefficient exactly from data — the precise analogue of the
paper's anchor condition (the identity transition fixes the one free
scale) — leaving curvatures and κ as the scanned knobs. The anchor test
of the build protocol then has a second, economic reading: the pinned
year must verify as an equilibrium of its own calibrated game at
residual ≈ 0.

## 4. What survives, what changes

- **Certified floors survive untouched.** Floors come from the pinned
  period constraints (homology forcing on the cylinder), not from the
  action. The action selects among feasible relaxations and prices
  surgery. With S_game, the floor statement upgrades from "no flow
  history carries this transition" to: *"no equilibrium path connects
  the two years without structural moves; the cheapest such moves have
  net cost X and sit on circuits Y"* — a certificate stated in
  cost/benefit language.
- **Multiplicity becomes a finding, not a failure.** Non-convex
  cost/benefit actions generically have multiple equilibria. The
  paper's floor discipline (multi-start, restart-independence) already
  detects this: restart-dependent outcomes now *mean* equilibrium
  multiplicity, worth reporting on its own.
- **The temperature knob is the rationality parameter.** The Gibbs
  weight e^{−βS_game} is the
  [logit / quantal-response](https://www.its.caltech.edu/~trp/QRE%20Primer.pdf)
  equilibrium family; logit-response dynamics select potential
  maximizers within exact potential games
  ([logit-response dynamics](https://www.sciencedirect.com/science/article/abs/pii/S0899825609001651);
  [Blume's statistical mechanics of strategic interaction](https://www.researchgate.net/publication/5147486_Regular_Quantal_Response_Equilibrium)).
  This is the principled bridge to the Stage-2 amplitude machinery: the
  partition-function reading of the cobordism arrives with an economic
  interpretation (β = agent rationality / noise) rather than a quantum
  metaphor. Notably, `RealizabilityOracle`'s mediated objective
  F_β = r_U + β·|S| already has exactly this shape.
- **The Regge parallel.** In the proton builds the Regge action supplies
  geometric rigidity against degenerate interiors; in the economic build
  S_game supplies *incentive* rigidity — interiors must be sequences of
  states no agent profitably deviates from. Same role, same slot in the
  objective.

## 5. Consequences for the backtest predictor

The `MatrixStrategy` plug (sibling of the incumbent RAS strategy in
`tvl.backtesting`) becomes: *predict the target period's matrix as the
equilibrium of the calibrated game under the target period's margins*.

- RAS is recovered exactly as the zero-benefit, quadratic-cost,
  no-surgery special case — so the game strategy strictly generalizes
  the incumbent and degrades gracefully to it.
- Parameter schema (what the plug-and-play interface registers): cost
  curvature, benefit curvature, κ (surgery threshold), β (0 = hard
  equilibrium, >0 = logit smoothing). All calibrated on train folds,
  scored out-of-sample by the existing predicted-vs-actual scorecard —
  which is precisely the identification discipline the idea needs: with
  free curvatures *any* transition is rationalizable in-sample, so the
  only honest test of the game action is out-of-sample matrix
  prediction against RAS on frozen folds.
- The register/topology layer adds the piece RAS cannot express:
  predicted *structural* change (surgery decisions from the
  pairwise-stability rule) rather than re-weighting only.

## 6. Risks, stated

1. **Identification** — the flexibility of B_v is the danger; the
   backtest's frozen-fold out-of-sample scoring is the only defensible
   arbiter, and the calibration must be committed before the test
   window (the strategy interface enforces this).
2. **Closure-vertex benefits** — household/government utility proxies
   are the weakest data link; start with VA-anchored industry benefits
   and treat closure benefits as fixed exogenous terms.
3. **Compute** — equilibrium-finding per fold is heavier than one
   Sinkhorn; the quadratic-cost/concave-benefit case stays convex
   (single equilibrium, fast), so the first implementation should stay
   in that regime.
4. **Efficiency vs stability** — certified structural changes are
   equilibrium statements; do not read them as welfare judgments.
