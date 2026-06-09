# Cobordism experiment — emergent-bulk realizability at k=1: surgery makes b₁ an output

> What happens when, at the Hodge-qubit degree $k=1$, you stop **pinning** the
> bulk topology and let it emerge? The earlier re-test assembled the bulk as a
> solid torus *before* asking which boundary harmonics it carries — so
> "longitude realizes / meridian floors" was just the chosen filling's $H_1$ read
> back out, a property of the **input**, not an output. This report corrects that:
> fix only the boundary, grow the interior with a **topology-changing surgery
> move** (a boundary-fixed interior-cell *remove*, #196), and read $b_1$ **off the
> witness**. The headline is sharp and now genuinely emergent — $b_1$ moves on its
> own, the matched two-boundary meridian realizes **iff** the grown filling has
> $b_1=1$, and a sign-flipped control floors on *every* filling. Companion to the
> runnable example `examples/cobordism/emergent_bulk_realizability.py` and the
> diagram generator `examples/cobordism/visualize_emergent_bulk.py`; grounds in the
> charter `cobordism.md` §4b, §5.0.

## Verdict

**At $k=1$ the realizability obstruction does NOT have to be installed by a
hand-picked filling — it falls out of the topology the search grows.** With the
boundary-fixed surgery move-set the bulk's first Betti number $b_1$ is a pure
**output**, and two distinct obstructions separate cleanly:

- **A topological one (does the handle exist?).** The matched two-boundary meridian
  **floors on the disk filling** ($b_1=0$: it bounds) and **realizes as a bulk
  harmonic on the annulus filling** ($b_1=1$: the cobordism carries it),
  $r\approx7\times10^{-8}$, eigenvalue $\to 0$. The surgery search, handed only the
  disk seed, **opens the handle on its own** ($b_1:0\to1$) and the floored meridian
  realizes.
- **A cohomological one (do the periods match?).** Flip the sign of *one* of the two
  boundary meridians (the cobordism conjugation $\mathrm{geo}(\psi_A)\sqcup
  \mathrm{conj}(\mathrm{geo}(\psi_B))$) and the target **floors on every filling,
  even after surgery opens the handle** — opposite periods are not the restriction
  of any closed form, so no bulk topology can carry them.

The example runs clean, deterministic at the seed verdict, and `exit 0`.

![emergent-bulk composite: the disk floors, the annulus realizes, surgery opens the handle](https://github.com/akellehe/tessera/releases/download/issue-attachments/emergent_bulk_composite.png)

## The methodological fix

The earlier $k=1$ re-test built the bulk as `SimplicialProduct(SolidSimplex(2),
S^1)` — the solid torus $W=D^2\times S^1$ — *before* asking which boundary
harmonics it carries. But $H_1(W)=\mathbb{Z}\langle\text{longitude}\rangle$ is
*built into that choice*: the longitude survives $H_1(\partial W)\to H_1(W)$, the
meridian bounds the disk and dies. "Longitude realizes, meridian floors" was the
chosen filling's homology read back out; the topology was an **input**.

The fix demotes the bulk topology from input to output. Nothing is hardcoded as a
torus / solid torus / disk / cone. The bulk is built **generically from a face
list** — an octahedron (a triangulated $S^2$) with faces deleted — and the
realizability engine is given a **topology-changing** move it never had before: a
boundary-fixed interior-cell **remove** (`EigenstateSynthesis::removeInteriorCell`,
exposed as `RealizabilityOracle.GrowthMode.SURGERY`). Removing an interior top cell
opens a hole/handle while the pinned boundary $\partial W$ stays **bit-exact**, so
$b_1$ **moves**. The realizability test is the physical one (`harmonic=True`): a
boundary class realizes iff it is *carried* as a bulk harmonic, i.e. it is the
restriction of a closed-and-coclosed 1-form on $W$ — equivalently it lies in
$\mathrm{image}\big(H^1(W)\to H^1(\partial W)\big)$.

## Setup: the octahedron surface family and the two-boundary meridian

The whole story runs at $k=1$ on a **surface** bulk $W$ (top cells = triangles),
the faithful low-dimensional analogue of the 3-manifold meridian/longitude setting.
The octahedron has eight triangular faces; deleting antipodal faces opens boundary
circles:

- **delete one face** $\Rightarrow$ a **disk** ($b_1=0$): the one circle bounds, and
  the antipodal triangle $\{3,4,5\}$ is still **filled**, so the second meridian
  bounds it too;
- **delete the two antipodal faces** $\Rightarrow$ an **annulus** ($b_1=1$): the two
  boundary circles are homologous, each surviving in $H_1$ — the cobordism between
  them.

Drawn as a Schlegel diagram (hole A $=\{0,1,2\}$ the outer triangle, hole B
$=\{3,4,5\}$ the inner triangle, six band cells between), the disk is a fully
**filled** triangle and the annulus is that triangle with the inner cell **punched
out** (a ring):

![the two boundary meridians geo(psi_A) || geo(psi_B)](https://github.com/akellehe/tessera/releases/download/issue-attachments/emergent_bulk_boundary.png)

The **target** is the meridian carried on *both* boundary circles, read off the
annulus's own $H_1$ generator (`HodgeLaplacian(annulus).harmonics(1)[0]`) restricted
to the two cycles, built two ways:

| target | construction | period $p_A$ | period $p_B$ | $p_A/p_B$ |
|---|---|---|---|---|
| **matched** | the harmonic's restriction (equal periods) | $-1.095$ | $-1.095$ | $+1$ |
| **flipped** | negate circle B's period (the conjugation) | $-1.095$ | $+1.095$ | $-1$ |

A closed 1-form on the annulus has **equal** periods on its two homologous boundary
circles (the flow is conserved through the tube), so the matched target is the
restriction of a genuine harmonic and the flipped one cannot be — the sign flip is
a deliberate negative control.

## E1 — the 2×2: realizable iff $b_1=1$ **and** the periods match

Both targets against both validity-only fillings, seed verdict (`max_cones=0`, so
the residual is exact and bit-exact reproducible across runs):

| target | filling | $b_1$ | realizable | residual $r$ | eigenvalue $\lambda$ |
|---|---|---|---|---|---|
| **matched** | disk | $0$ | no | $4.52\times10^{-1}$ | $4.81\times10^{-1}$ |
| **matched** | **annulus** | $1$ | **YES** | $6.90\times10^{-8}$ | $6.32\times10^{-9}$ |
| flipped | disk | $0$ | no | $9.92\times10^{-1}$ | $9.25\times10^{-1}$ |
| flipped | annulus | $1$ | no | $2.03\times10^{-1}$ | $4.51\times10^{-1}$ |

Read straight off the table:

1. **MATCHED realizes only on the annulus.** Same boundary, same target — the
   *disk* filling ($b_1=0$) floors it ($r\approx0.45$) and the *annulus* ($b_1=1$)
   realizes it to $r\sim10^{-8}$ with eigenvalue $\to0$ (carried as a harmonic).
   The verdict flips with the filling's $b_1$.
2. **FLIPPED floors on both.** Even on the annulus, where the topology is right, the
   sign-flipped meridian floors ($r\approx0.20$, $\lambda\approx0.45\neq0$): opposite
   periods are not the restriction of any closed form. The realizable set is exactly
   $\mathrm{image}\big(H^1(W)\to H^1(\partial W)\big)$ — for the disk that image is
   $\{p=0\}$ (any nonzero meridian floors), for the annulus it is the diagonal
   $\{p_A=p_B\}$ (matched realizes, flipped does not).

![disk filling (b1=0): the matched meridian floors](https://github.com/akellehe/tessera/releases/download/issue-attachments/emergent_bulk_disk.png)
![annulus filling (b1=1): the matched meridian realizes as a harmonic](https://github.com/akellehe/tessera/releases/download/issue-attachments/emergent_bulk_annulus.png)

The disk panel is a fully filled triangle (the inner cell $\{3,4,5\}$ is present, so
circle B bounds it); the annulus panel is the same triangle with the inner cell
punched out — a ring — and its edges are colored by the realized **carried harmonic
1-form** (the `Verdict.state`), the meridian the cobordism carries.

The sign-flip is a clean negative control: the **same** annulus ($b_1=1$), only
circle B's period negated, floors exactly where the matched meridian realizes — the
cohomological obstruction made visible side by side (matched left, flipped right):

![matched vs flipped on the same annulus — the period sign decides realizability](https://github.com/akellehe/tessera/releases/download/issue-attachments/emergent_bulk_flipped.png)

## E2 — surgery moves $b_1$ on its own (the obstruction is emergent)

From the **disk seed** ($b_1=0$, the meridian floored) the SURGERY search scores
every interior-top-cell removal by the harmonic residual it reaches and commits the
best improving one. There is exactly one interior top cell — $\{3,4,5\}$ — and
removing it opens the handle. Across seeds:

| target | seed | removals | $b_1$ | realizable | residual $r$ |
|---|---|---|---|---|---|
| **matched** | 0 | 1 | $0\to1$ | **YES** | $4.76\times10^{-5}$ |
| **matched** | 1 | 1 | $0\to1$ | **YES** | $1.06\times10^{-4}$ |
| **matched** | 2 | 1 | $0\to1$ | **YES** | $5.31\times10^{-5}$ |
| **matched** | 3 | 1 | $0\to1$ | **YES** | $3.48\times10^{-5}$ |
| flipped | 0 | 1 | $0\to1$ | no | $2.25\times10^{-1}$ |
| flipped | 1 | 1 | $0\to1$ | no | $2.20\times10^{-1}$ |
| flipped | 2 | 1 | $0\to1$ | no | $2.21\times10^{-1}$ |
| flipped | 3 | 1 | $0\to1$ | no | $2.19\times10^{-1}$ |

$b_1$ moves $0\to1$ **on its own** for *both* targets (the opened handle always
lowers the residual, so the removal is committed) — scored purely by the harmonic
residual, with $\partial W$ held bit-exact. The grown $b_1$ is a pure **output**.
But only the **matched** meridian then realizes; the **flipped** one still floors on
the opened handle. Surgery delivers the *topology*; the leftover period mismatch is
the *cohomological* obstruction no filling can fix. The before/after — the red
interior cell $\{3,4,5\}$ punched out to a white hole, $b_1:0\to1$:

![surgery: disk -> annulus, b1 moves 0 -> 1 on its own](https://github.com/akellehe/tessera/releases/download/issue-attachments/emergent_bulk_surgery.png)

## E3 — only removal moves $b_1$; additive growth is frozen

The contrast that makes "surgery is the load-bearing move" precise. Without the
remove move, $b_1$ cannot budge:

| growth | $b_1$ | matched realizable | note |
|---|---|---|---|
| none (seed only, `max_cones=0`) | $0\to0$ | no ($r=4.5\times10^{-1}$) | the bare disk floor |
| additive attach (`attachInteriorVertex`) | $0\to0$ | rejected | wiring a triangle in grows $\partial W$ |

The additive `attachInteriorVertex` (the `FREE_CONNECTIVITY` atom) and the stellar
Pachner subdivision (`growInterior`) are **topology-preserving** at $k\ge1$:
a dangling additive cell is dropped by the downward-closure `ChainComplex`, and an
additive top-cell attach introduces boundary edges the bit-exact $\partial W$ guard
rejects. So no boundary-fixed *additive* move can move $b_1$ — the boundary-fixed
**remove** is the one move that does.

## Headline finding: the obstruction is emergent, not installed

Putting the pieces together, the emergent-bulk methodology — fix the boundary, grow
with surgery, let the obstruction fall out — **works at $k=1$**, and it resolves the
earlier "obstruction is installed by the filling" diagnosis:

- The disk seed floors the matched meridian; the surgery search opens the handle
  ($b_1:0\to1$) **on its own**, scored only by the residual, and the meridian
  realizes (E2). The topology is selected by the spectral residual, not by the
  experimenter.
- Realizability is two conditions, and the experiment separates them: a
  **topological** one ($b_1=1$, the handle exists — E1's disk/annulus split, E2's
  surgery) and a **cohomological** one (matched periods — E1's matched/flipped
  split, E2's flipped-still-floors). The sign-flipped meridian floors on *every*
  filling, so the period obstruction is genuinely distinct from the topological one.
- The move that moves $b_1$ is **removal**; the additive moves are frozen at $k\ge1$
  (E3).

So the realizable set at $k=1$ is exactly $\mathrm{image}\big(H^1(W)\to
H^1(\partial W)\big)$ for the $W$ the **search grows** — the earlier re-test's
result, now produced as an *output* (with $b_1$ read off the witness) rather than
installed by a pinned solid torus, and refined by the period-matching control.

## Method

The example is pure orchestration of the merged, separately-tested classes — no new
math:

1. **Build the boundary and the meridian targets.** The octahedron-minus-faces disk
   and annulus are built from a face list (`_surface`); the matched / flipped
   meridians are the annulus's `HodgeLaplacian.harmonics(1)[0]` restricted to the
   two boundary cycles, with circle B's period optionally negated.
2. **E1 — the 2×2.** `RealizabilityOracle(W).decideHarmonic(growth_mode=SURGERY,
   max_cones=0, harmonic=True)` on each (target, filling); record realizable /
   residual / eigenvalue, and `ChainComplex.bettiNumbers` of each filling.
3. **E2 — surgery emergence.** Re-run from the disk seed with `max_cones=3` across
   seeds; read `Verdict.surgery_removals` and the witness $b_1$ before/after.
4. **E3 — frozen without surgery.** Seed-only verdict, plus a direct
   `EigenstateSynthesis.attachInteriorVertex` probe showing the additive move is
   refused.

The surgery primitive (remove / restore, $b_1$ moves, $\partial W$ bit-exact), the
2×2, the surgery emergence, and the additive lock are re-derived independently in
`tests/cobordism/test_emergent_bulk_python.py` (a separate code path from the
example), which also checks the committed example self-verifies.

## Key findings

1. **$b_1$ is an output.** The surgery search opens the handle ($b_1:0\to1$) on its
   own from the disk seed, scored only by the harmonic residual, with $\partial W$
   bit-exact. Topology is read off the witness, not pinned.
2. **The matched meridian realizes iff $b_1=1$.** It floors on the disk ($b_1=0$,
   $r\approx0.45$) and realizes on the annulus ($b_1=1$, $r\approx7\times10^{-8}$,
   $\lambda\to0$); the surgery-grown annulus realizes it too ($r\sim10^{-5}$).
3. **The sign-flip floors everywhere.** Opposite periods ($p_A=-p_B$) are not the
   restriction of any closed form, so the flipped meridian floors on the disk *and*
   the annulus *and* the surgery-grown handle — a cohomological obstruction distinct
   from the topological one.
4. **Removal is the load-bearing move.** No boundary-fixed *additive* move (attach /
   Pachner subdivision) moves $b_1$ at $k\ge1$; the boundary-fixed remove is the one
   that does.
5. **The realizable set is $\mathrm{image}(H^1(W)\to H^1(\partial W))$.** Read as a
   continuous spectral certificate: zero on the image (carried), a positive floor
   off it whose size tracks the distance from the image.

## Conventions

- The bulk family is built **generically from a face list** (no `SimplicialProduct`,
  no `Toroid`, no `SolidSimplex`): an octahedron (triangulated $S^2$); the disk is it
  minus one face ($b_1=0$), the annulus minus two antipodal faces ($b_1=1$). All
  edges pinned Hermitian (weight $1$, phase $0$).
- The $k=1$ Laplacian is $L_1=\partial_1^\top\partial_1+\partial_2\partial_2^\top$;
  with `harmonic=True` the residual is $r=\lVert L_1\psi\rVert^2$ (distance from
  $\ker L_1=H_1(W)$) and the eigenvalue is the Rayleigh quotient
  $\lambda=\psi^\dagger L_1\psi$.
- Realizable iff $r<\texttt{REALIZE}=10^{-3}$; floored when $r>\texttt{CERT\_FLOOR}=
  10^{-2}$ (the certified obstruction floor). The LM tolerance `DEEP_EPS`$=10^{-7}$
  polishes below the verdict line; the verdict is read off the realized residual.
- Topology is read **off the witness** as an output: `ChainComplex.bettiNumbers`.
  The $E1$ seed verdict (`max_cones=0`) is exact and bit-exact reproducible; the
  surgery removal count and $b_1:0\to1$ hold across seeds (the residuals vary in the
  last digits with the seeded restart draws).
- The diagrams use a fixed Schlegel layout so the only visible change between
  fillings is which 2-cell is filled; every panel carries a large title and an
  explicit legend. Seeded and reproducible; the 10-CPU cap is honored (thread env set
  at launch); matplotlib uses the headless `Agg` backend.

## Reproduce

```
pip install -e ".[dev]"        # default Release; the fast linker is auto-gated off Release (bfd links the LTO _tessera)
python examples/cobordism/emergent_bulk_realizability.py   # the 2x2 + surgery tables
python examples/cobordism/visualize_emergent_bulk.py       # the panels + composite
python -m pytest tests/cobordism
```

The raw table defaults to `/tmp/cobordism/emergent_bulk_realizability.json` and the
diagram PNGs to `/tmp/cobordism/`, both **not committed** — the example, the
visualizer, and this report are the committed artifacts. The diagrams are uploaded to
the
[`issue-attachments`](https://github.com/akellehe/tessera/releases/tag/issue-attachments)
release and referenced above by their release URLs; attach the raw table to the
issue/PR to pin a result.
