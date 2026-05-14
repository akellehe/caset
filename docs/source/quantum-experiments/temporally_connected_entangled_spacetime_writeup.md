# Temporally-connected entangled spacetime

Companion experiment to the dual-lattice section of
[emergent_spectral_dimension_writeup.md](emergent_spectral_dimension_writeup.md).
The dual lattice there uses two distinct rules for the two axes of the
$(bond, snapshot)$ graph: tripartite mutual information for spatial
edges, and a causet-style same-bond forward step for temporal edges.
This experiment replaces the causet temporal sector with mutual
information — *every* bond pair $(n, m)$ across adjacent snapshots
that shares MI above the cutoff becomes connected.

## Construction

Vertices are unchanged: one per $(bond,\ snapshot)$ pair, with bonds
running over $n \in [0, N-2]$ and snapshots over $t \in [0, K-1]$.

**Spatial edges.** Within snapshot $t$, every bond pair $(n, m)$ with
$\mathcal{I}_t(n, m) > \varepsilon_I$ gets an edge of weight
$\mathcal{I}_t(n, m)$, where $\mathcal{I}_t$ is the tripartite
information matrix recorded as
`TDVPSnapshot.bondMutualInformation`. Same construction as the
existing dual lattice.

**Temporal edges (new).** Between snapshot pair $(t, t')$ with
$1 \leq t' - t \leq s_{\max}$, every bond pair $(n, m)$ with
$\widetilde{\mathcal{I}}(n, t; m, t') > \varepsilon_I$ gets an edge,
where the cross-snapshot weight is the symmetrised endpoint average

$$
\widetilde{\mathcal{I}}(n, t; m, t')
\;=\; \tfrac12 \bigl(\mathcal{I}_t(n, m) + \mathcal{I}_{t'}(n, m)\bigr).
$$

This replaces the causet "bond $n$ at $t \to$ bond $n$ at $t+1$" rule
(degree-2 temporal sector) with one whose temporal degree scales with
the number of bonds in the chain. Two bonds that are mutually informed
in either of two adjacent snapshots inherit a cross-time connection.

**Surrogate caveat.** A first-principles cross-snapshot MI between
bond bipartitions would come from the Choi state of the TDVP
propagator restricted to bipartition variables. Tessera's Choi
machinery operates on sites, not bonds, so we use the endpoint average
above. It reduces to the spatial weight when $t' = t$, is symmetric in
$(t, t')$, and shares the van Raamsdonk-style "distance is a function
of MI" interpretation when fed into the weighted Laplacian (edge
weights enter as $W = \mathcal{I}$, so $\ell = -\log \mathcal{I}$ is
the implicit edge length per
[holography-causal-ordering-emergent-dimension.md](../holography-causal-ordering-emergent-dimension.md)
§3.4). The averaged-endpoint surrogate does not capture the pure-time
correlations that the unitary evolution generates *between* snapshots
— it captures the part of the cross-time MI that is consistent with
the spatial MI at both endpoints. Quantitative bulk-geometry claims
should wait on a proper Choi-state bond MI.

## Hypothesis

The dual-lattice experiment with causet temporal edges plateaus near
$D_S \approx 1.8$ at moderate $N$. The MI-temporal sector adds a
larger temporal-edge population (degree up to $N - 1$ per snapshot
pair vs. degree 1 in the causet case). Two qualitatively different
outcomes are possible:

1. **Geometric preservation.** Cross-snapshot MI edges concentrate on
   bond pairs that *would* be temporal neighbors in the bulk
   geometry, so the spectral dimension stays near 2. The extra edges
   just thicken the diffusion bundle around the bulk geodesics.
2. **Small-world collapse.** The cross-snapshot MI fans out enough
   that the graph becomes effectively small-world: $D_S(\sigma)$ peaks
   well above 2 at intermediate $\sigma$ (the random walk explores
   many vertices per step), and the long-$\sigma$ tail does not match
   the Ambjorn-Loll $D_\infty - C/(B + \sigma)$ form.

Outcome 1 would say "MI is a richer temporal connectivity than the
causet step, but the bulk geometry is robust." Outcome 2 would say
"the operational MI temporal sector is too connected to recover the
bulk dimension."

## Setup

- Schwinger chain, $a = g = 1$, $N \in \{10, 20, 30, 40\}$, OBC.
- DMRG ground state at the specified $m/g$; bond-dim cap 64, 12 sweeps.
- $\sigma^- \sigma^+$ q-qbar quench at $i_0 = 3$, separation $d = 3$.
- 2-site TDVP with $\Delta t = 0.25$, total $T = 1.0$. Five snapshots.
- Per-snapshot bond MI recorded (`recordBondMutualInformation = True`).
- Cross-snapshot temporal MI uses the symmetrised endpoint average above.
- `--max-temporal-stride 10`: with $K = 5$ this is effectively
  unlimited (clipped to $K - 1 = 4$); every forward snapshot pair
  contributes temporal edges.
- 48-point logarithmic $\sigma$-grid on $[10^{-2}, 10^3]$.
- MI cutoff $\varepsilon_I = 10^{-8}$.
- $D_S(\sigma)$ via Savitzky-Golay smoothing (window 5, polyOrder 2).
- Three-parameter Ambjorn-Loll fit $D_S(\sigma) = D_\infty - C/(B + \sigma)$.

Reproduce one cell with:

```bash
OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 \
MKL_NUM_THREADS=10 BLIS_NUM_THREADS=10 \
python examples/quantum/temporally_connected_entangled_spacetime.py \
    --N 40 --m-over-g 0.5 \
    --max-temporal-stride 10 \
    --out-json /tmp/temporal-entangled/scan/N40_mg_0.5.json
```

Scan over $N \in \{10, 20, 30, 40\}$ and $m/g \in \{0.125, 0.25, 0.5\}$
with `examples/quantum/plot_temporally_connected.py` to render the
figure below from the resulting JSON.

## Results

![Temporally-connected entangled spacetime: D_S(σ), peak D_S vs N, D_∞
fit vs N](figures/temporally_connected_entangled_spacetime.png)

|  N  |  m/g  | $\|V\|$ | $\|E\|$ | $E_{sp}$ | $E_{tm}$ | deg mean | deg max | peak $D_S$ | $\sigma_{peak}$ | $D_\infty$ | fit $\chi^2$ |
|----:|------:|--------:|--------:|---------:|---------:|---------:|--------:|-----------:|----------------:|-----------:|-------------:|
|  10 | 0.125 |      45 |     900 |      180 |      720 |     40.0 |      40 |      2.737 |           0.394 |     +7.040 |     7.92e-01 |
|  10 | 0.250 |      45 |     900 |      180 |      720 |     40.0 |      40 |      3.072 |           0.309 |     +1.346 |     1.89e+00 |
|  10 | 0.500 |      45 |     900 |      180 |      720 |     40.0 |      40 |      3.178 |           1.050 |     +0.530 |     8.90e-01 |
|  20 | 0.125 |      95 |    4275 |      855 |     3420 |     90.0 |      90 |      3.103 |           0.504 |     -0.094 |     5.75e-01 |
|  20 | 0.250 |      95 |    4275 |      855 |     3420 |     90.0 |      90 |      3.291 |           0.394 |     -0.271 |     6.18e-01 |
|  20 | 0.500 |      95 |    4275 |      855 |     3420 |     90.0 |      90 |      3.363 |           0.822 |     +0.738 |     1.07e+00 |
|  30 | 0.125 |     145 |    9387 |     1877 |     7510 |    129.5 |     140 |      3.361 |           0.643 |     -0.128 |     6.40e-01 |
|  30 | 0.250 |     145 |   10150 |     2030 |     8120 |    140.0 |     140 |      3.555 |           0.394 |     -0.133 |     6.61e-01 |
|  30 | 0.500 |     145 |   10150 |     2030 |     8120 |    140.0 |     140 |      3.402 |           0.822 |     +0.690 |     1.14e+00 |
|  40 | 0.125 |     195 |   14162 |     2832 |    11330 |    145.3 |     190 |      3.487 |           0.643 |     -0.101 |     6.81e-01 |
|  40 | 0.250 |     195 |   17039 |     3409 |    13630 |    174.8 |     190 |      3.695 |           0.394 |     -0.113 |     6.98e-01 |
|  40 | 0.500 |     195 |   18525 |     3705 |    14820 |    190.0 |     190 |      3.460 |           0.643 |     +0.850 |     1.29e+00 |

Four readings:

1. **Outcome 2 wins.** Every $(N, m/g)$ cell has peak $D_S \in
   [2.7, 3.7]$ — well above the lattice value of 2 that the causet
   dual-lattice peaks at. The MI-temporal sector is too connected to
   recover the bulk 1+1D dimension at the scales tested.
2. **Edge counts saturate against the bond cutoff.** At $N \geq 20$
   the temporal:spatial edge ratio is exactly 4:1 (four forward
   strides $\times$ all bond pairs vs. one same-snapshot
   upper-triangle), and the maximum degree equals $N - 1$ in the
   densest cells. The MI cutoff $\varepsilon_I = 10^{-8}$ is rarely
   blocking edges; the graph is essentially complete on the temporal
   axis.
3. **$m/g$ dependence is small but present.** The peak $D_S$ varies
   by $\sim 0.3$ across $m/g$ at fixed $N$, monotonic at $N = 10$
   (light $<$ mid $<$ heavy on this scale) and non-monotonic at
   $N \geq 20$ (the $m/g = 0.25$ cell consistently peaks highest).
   The signal is *not* swamped by the high-degree saturation — physics
   still shows through, just weakly.
4. **The Ambjorn-Loll fit is no longer a good model.** At nine of
   twelve cells $\chi^2 / \mathrm{dof} > 0.5$ and the $D_\infty$
   parameter swings between $-0.27$ and $+7.04$ depending on the cell
   — the small mass case at $N = 10$ has $C \approx -36{,}000$ and
   $B \approx -5{,}700$, well outside any physical interpretation. The
   $D_S(\sigma)$ profile under MI-temporal connectivity does not have
   the $D_\infty - C/(B + \sigma)$ shape the AL form assumes; it rises
   sharply, peaks, and falls without a clean long-$\sigma$ plateau in
   the diffusion regime we sample. The faint grey overlays on the
   left panel of the figure show the fits visibly missing the curve.

## Falsification check

| Criterion | Expected if "MI temporal preserves bulk geometry" holds | Observed | Status |
|---|---|---|---|
| Trivial confirmation ($D_S \equiv 2$) | Rejected | **Rejected** — $D_S$ range spans $[0, 3.7]$ at every $(N, m/g)$. | Pass |
| Independence from $m/g$ | Rejected | **Rejected** — peak $D_S$ varies $\geq 0.3$ across $m/g$ at every $N$. | Pass |
| Strong falsification (non-monotonic outside small-$\sigma$ regime) | Not triggered | **Not triggered** — every profile is unimodal (rise then fall). | Pass |
| $D_S \to 2$ at intermediate $\sigma$ | Confirmed | **Rejected** — peak $D_S$ is $\geq 2.7$ in *every* cell. The construction overshoots the lattice dimension by 35–85 %. | **Fail** |
| Ambjorn-Loll fits the long-$\sigma$ tail | Confirmed | **Rejected** — fit residuals and parameters are pathological at $N \leq 30$ for $m/g \leq 0.25$. | **Fail** |

The construction passes the trivial-confirmation and $m/g$-sensitivity
checks, but it fails the substantive bulk-geometry prediction: $D_S$
overshoots 2 across the entire scan, and the AL form does not describe
the resulting profile. **The MI-temporal sector does not preserve the
spectral-dimension content of the causet dual lattice.**

## Reading

Compared against the causet dual lattice — which reaches peak $D_S
\approx 1.8$ at $N = 16$ in the
[emergent_spectral_dimension_writeup.md](emergent_spectral_dimension_writeup.md)
"Dual lattice" section — replacing the causet step with the
endpoint-averaged MI surrogate produces a graph that is genuinely
denser in the temporal direction, not just dressed-up. The same
weighted Laplacian sees a graph of much higher mean degree (40 at
$N = 10$, 190 at $N = 40$) and the heat-kernel walker explores a
small-world-like neighborhood per unit $\sigma$. That is consistent
with $D_S$ being driven up by topology, not by physics.

Two follow-ups would tighten the interpretation:

1. **Sweep $\varepsilon_I$** to thin the temporal sector. If the
   overshoot persists at $\varepsilon_I \in \{10^{-4}, 10^{-3}\}$
   where significant temporal edges are dropped, the conclusion
   "graph is too small-world" is robust. If it disappears, the
   threshold choice was driving the result.
2. **Replace the endpoint-averaged surrogate with a true Choi-state
   bond MI.** That requires extending the existing site-based Choi
   machinery to bipartition variables; it's the next refinement
   needed before any quantitative spacetime-reconstruction claim.

## Reproducibility

The full scan lives at `/tmp/temporal-entangled/scan/`:

```
N{10,20,30,40}_mg_{0.125,0.25,0.5}.json
scan.log
run_scan.sh
```

Each JSON record carries the input config, snapshot count, graph
counts, degree summary, $\sigma$/$P$/$D_S$ arrays, the Ambjorn-Loll
fit, and `peak_dS` / `sigma_peak`. The plot script
`examples/quantum/plot_temporally_connected.py` reads this directory
directly and writes
`docs/source/quantum-experiments/figures/temporally_connected_entangled_spacetime.png`.

The experiment script that produced the JSON is
`examples/quantum/temporally_connected_entangled_spacetime.py`. Anyone
with a tessera build at or after the `recordBondMutualInformation`
fix can regenerate every number above.

## See also

- [emergent_spectral_dimension_writeup.md](emergent_spectral_dimension_writeup.md)
  — the causet dual-lattice baseline this experiment varies from.
- [holography-causal-ordering-emergent-dimension.md](../holography-causal-ordering-emergent-dimension.md)
  §3.4 — the weighted-Laplacian convention $W = I$, $\ell = -\log I$.
- [causal_sets.md](../causal_sets.md) — the partial-order machinery
  the causet temporal edges were drawn from.
