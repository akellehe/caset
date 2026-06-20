# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Compose bipartite MergeCobordism merges into a sequence (#382).

The compose mechanism is: the EMERGENT result of one merge becomes a boundary
INPUT of the next (result-state = next-merge boundary), never a hand-welded
interior. Each `MergeCobordism` on the `RegisterTopology` pins its neutral-pair
inputs over the EXACT period path (`residualForPeriods`, pin inputs only) and
reads the emergent result block over `cyclePeriods` -- so the result is read out
of the relaxed geometry after the fact, never inserted by fiat. Each step is a
genuine valid manifold (b1 = 2 shared color register, every triangle in <= 2
tets); the merges are *composed*, not welded.

The structural finding (reproducing #353's own documented result): a bipartite
sequence does NOT reach the color singlet [1, omega, omega^2] (|sigma| = 0). The
emergent result stays color-charged (|sigma| ~ 0.7) at every step -- two color
objects meeting pairwise do not make a singlet. The proton needs the TRIPARTITE
junction W_ABC (boundary = psi_A ⊔ psi_B ⊔ psi_C: three neutral-pair inputs into
one bulk), where all three colors meet at once. This sequence demonstrates the
compose machinery and pins the bipartite obstruction that motivates W_ABC.
"""

import cmath

import tessera

cob = tessera.cobordism
_W = cmath.exp(2j * cmath.pi / 3)

# Three color-neutral q-qbar pairs (Sigma = 0 each): the carriable inputs.
_A = [1, -1, 0]   # R - Gbar
_B = [1, 0, -1]   # R - Bbar
_C = [0, 1, -1]   # G - Bbar
_SINGLET = [1, _W, _W * _W]  # the proton target: |sigma| = 0


def _merge(inputs, max_iters=60):
    # The register pins INPUTS only and reads the EMERGENT result block, so no
    # output states are supplied (emergesResult()).
    return cob.MergeCobordism(inputs, [], max_iters=max_iters, seed=0,
                              topology=cob.RegisterTopology())


def _fmt(vec):
    return "[" + ", ".join(f"{z.real:+.3f}" for z in vec) + "]"


def main():
    print("=== #382 bipartite merge SEQUENCE (compose result -> next boundary) ===")
    print(f"target color singlet {_fmt(_SINGLET)}  |sigma| = {abs(sum(_SINGLET)):.2e}\n")

    # Step 1: merge two neutral pairs; the emergent result AB is read out.
    m1 = _merge([_A, _B])
    ab = list(m1.output_state)
    print(f"step 1  merge(A, B) -> AB = {_fmt(ab)}")
    print(f"        |sigma_AB| = {abs(sum(ab)):.4f}   r_U(inputs) = {m1.stats.state_residual:.3e}"
          f"   b1 = {list(m1.stats.betti_cobordism)[1]}")

    # Step 2: the EMERGENT AB is now an INPUT boundary of the next merge (compose,
    # not weld) alongside the third neutral pair C.
    m2 = _merge([ab, _C])
    abc = list(m2.output_state)
    print(f"step 2  merge(AB, C) -> ABC = {_fmt(abc)}   [AB fed as a boundary input]")
    print(f"        |sigma_ABC| = {abs(sum(abc)):.4f}   r_U(inputs) = {m2.stats.state_residual:.3e}"
          f"   b1 = {list(m2.stats.betti_cobordism)[1]}")

    print("\n=== finding ===")
    print(f"  bipartite sequence result |sigma_ABC| = {abs(sum(abc)):.4f}  (singlet target = 0)")
    print("  -> two color objects meeting pairwise do NOT make a singlet (#353's")
    print("     finding). The proton needs the tripartite junction W_ABC, where")
    print("     all three colors meet at once.")


if __name__ == "__main__":
    main()
