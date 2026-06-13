# Regge-Mediated Bulk Synthesis: results

> The results companion for the **Regge-mediation track** — the geometry-selection
> question raised at the end of
> [`cobordism-results.md`](cobordism-results.md): *among the carriers of a
> realizable operation, which one does the synthesis pick, and does it matter?*
> The base layer answers H1–H3 by the **spectral residual** alone; here the
> **dual Lorentzian Regge action** enters as a mediator, and we sweep its coupling
> $\beta$. Every number below is produced by
> [`examples/cobordism/mediated_gate_battery.py`](https://github.com/akellehe/tessera/blob/main/examples/cobordism/mediated_gate_battery.py)
> (the design is `docs/design/cobordism-implementation-details.pdf` §8–§11). The
> framing is spectral/continuous throughout — no Dijkgraaf–Witten layer.

**Merge framing, and the spacetime as the emergent dual.** The bulk scored
and synthesized here is the **merge** cobordism: two co-incoming states on
slice $t$ are pinned, and the interior is filled so the bulk merges them into a
single object at $t+1$ (the simplicial pair-of-pants;
[`examples/cobordism/merge_cobordism.py`](https://github.com/akellehe/tessera/blob/main/examples/cobordism/merge_cobordism.py)).
We do **not** build the spacetime — it *emerges as the dual* $W^{*}$ once the
residual fixes the carrier and the action selects among carriers. The merge is
coordinate-free: its $t\to t{+}1$ arrow is the **sign of $\ell^2$** on the
primal (the input$\to$bulk edges timelike), and the dual inherits that causal
character canonically, so $S_{\mathrm{Regge}}(W^{*})$ is the complex
(Sorkin) Lorentzian action — no vertex coordinates, no CDT constant lengths.
The realizability residual reads the **Riemannian** (magnitude) register; the
Lorentzian structure lives only on the bulk geometry and its dual.

## The mediated objective

Each bulk candidate $W$ is scored by

$$ F_\beta(W) \;=\; r_U(W) \;+\; \beta\,\bigl|S_{\mathrm{Regge}}(W^{*})\bigr|, $$

the realizability residual $r_U$ on the **primal** Hodge Laplacian (the base layer's
$\lVert(I-\psi\psi^\dagger)L_1\psi\rVert^2$) plus $\beta$ times the **magnitude of the
dual Lorentzian Regge action** on the circumcentric dual $W^{*}$ — the gravitational
prior, evaluated with rapidity-carrying (boost) dihedral angles on timelike hinges
(Regge 1961; Sorkin, *Lorentzian angles*, arXiv:1908.10022; Asante–Dittrich–Padua-Argüelles,
arXiv:2104.00485). The mutual-information / Van Raamsdonk reading of the dual metric
is Cao–Carroll–Michalakis's (arXiv:1606.08444); the contribution here is the
fixed-boundary, dual-side mediation and the inverse direction. At $\beta=0$ the
objective is $r_U$ alone, so the search **reproduces the base layer exactly**.

## The mechanism: holes cost gravitational action

The carrier the base layer uses is a triangulated sphere with three holonomy holes
opened by surgery, so its harmonic space $\ker L_1$ (the register) is two-dimensional
and carries the 13 charge-conserving gates. Opening each hole drives the residual of a
carried gate to zero — but it **raises** the dual Regge action monotonically:

| holes open $k$ | $b_1$ | $\lvert S_{\mathrm{Regge}}\rvert$ | residual (Identity) |
|:---:|:---:|:---:|:---:|
| 0 | 0 | 9.07 | floored |
| 1 | 0 | 10.43 | floored |
| 2 | 1 | 11.79 | floored |
| **3** | **2** | **13.15** | **$1.1\times10^{-29}$ (realized)** |

So $F_\beta$ is a genuine trade-off: a gate realizes only once enough holes are open
(b_1 = 2), and each hole is a fixed gravitational cost. A large enough $\beta$ makes
the search stop short of the topology a gate needs — the gate then **floors out of the
realizable set**. This is the conformal-mode intuition made discrete: the action term
is regularized by the vertex/volume bound, and increasing $\beta$ prices out structure.

## Realizable gates vs $\beta$

![Realizable gates vs beta](https://github.com/akellehe/tessera/releases/download/issue-attachments/regge_mediated_realizable_vs_beta.png)

The headline result. At $\beta=0$ the search opens all three holes for every carried
gate and recovers the **13-gate base layer bit-for-bit**. As $\beta$ rises the
realizable set **contracts** — $13 \to 11 \to 0$ — because the gravitational cost of
the holes overtakes their residual benefit. Requiring low gravitational action
**does** shrink what is realizable, and the curve says where: the bulk of the set
survives to $\beta\approx 2.5$, then the geometry that the charge-conserving gates need
($b_1=2$, three holes) becomes too expensive and the set collapses past $\beta\approx 3$.

The contraction is genuine physics of the *combined* objective, not a post-hoc
tie-break: $\beta$ enters the score the search minimizes, so a strong gravitational
prior can prevent $r_U$ from reaching zero within the volume bound. (Were $\beta$ only
post-selecting among already-realized fillings, this curve would be flat.)

## H3 amplitude error vs $\beta$

![H3 amplitude error vs beta](https://github.com/akellehe/tessera/releases/download/issue-attachments/regge_mediated_h3_vs_beta.png)

For every gate that *does* realize, at every $\beta$, the explicit amplitude
deviation $\lvert Z - \langle\psi_A|U|\psi_B\rangle\rvert$ — the holonomy-charge leak
$\lvert\Sigma(U|\psi_B\rangle)\rvert$, computed directly from the gate matrix and
**not** inferred from $r_U\to 0$ — sits at machine precision ($\approx 4.5\times
10^{-16}$, below the $\sim 10^{-15}$ target). Mediation **changes which** gates
realize, but it does not degrade the fidelity of the ones that survive: the value
equation $Z = \langle\psi_A|U|\psi_B\rangle$ holds under the Regge prior. The points
thin out left-to-right exactly as the realizable set contracts.

## H1–H3 under mediation

- **H1 (realizable).** $\beta=0$ realizes the 13 charge-conserving gates
  (`3-cycle (0231)`, `3-cycle (0312)`, `CNOT`, `CSX`, `CSXdg`, `H(x)H`, `Identity`,
  `SWAP`, `rev-CSX`, `rev-CSXdg`, `reversed-CNOT`, `sqrt-SWAP`, `sqrt-SWAP-dg`) —
  the base-layer set. The count contracts to 11 for $1\le\beta\le 2.5$ and to 0 for
  $\beta\ge 3$.
- **H2 (boundary byte-fixed).** The input register-edge geometry and the input
  periods are byte-identical across every gate and every $\beta$; the synthesis only
  opens interior holes and never perturbs the input boundary data.
- **H3 (value equation).** Holds to $\approx 4.5\times10^{-16}$ for every realized
  gate at every $\beta$ (plot above) — within the base-layer tolerance.

## Finding

Introducing the dual Regge action as a mediator **selects geometry**: at zero coupling
it is invisible (the base layer is recovered exactly), and as the coupling grows it
**contracts the realizable set** by pricing out the topology charge-conserving gates
require, while leaving the amplitude fidelity of the survivors intact. The realizable
gate set is therefore a genuine function of the gravitational coupling — the central
claim of the milestone, now quantified.

Reproduce with:

```bash
python examples/cobordism/mediated_gate_battery.py --plots /tmp/cobordism/plots
```
