# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The master recursive construction, iterated on a real complex.

The whitepaper's boxed display, executed line by line and reported at every
scale::

    P_l        = PersistentPartition(R_l)
    E_v^{l+1}  = certified isolated subspace of C_v^l
    R_{l+1}(z) = Feshbach_{P_l}(R_l(z))
    h_{l+1}    = labeled sum of the E_v^{l+1},  G = J^dag W J
    H_{l+1}    = Fock(h_{l+1})

The point of the driver is that the recursion is *iterated on geometry*. A
single-level reduction, or a reduction of a hand-written matrix, exercises the
identities but never the hierarchy the construction is named after: each level
must discover its own components, reduce its own operator, assemble its own
certified fibers, and hand a well-formed level to the next step.

Two response modes are available and both are driven here:

``--mode static``
    ``R_{l+1} = Feshbach_{P_l}(R_l(0))``, the exact supported static Schur
    complement. Exact at zero frequency, and it does NOT preserve the nonzero
    spectrum -- no nonzero-spectrum claim is attached to it anywhere.

``--mode pencil``
    ``R_{l+1}(z) = Feshbach_{P_l}(R_l(z))`` at a declared ``--lambda`` over a
    declared window. This is the pencil the specification defines; the child
    carries the window, the solve and compatibility residuals, the resonance
    flag, and the producing certificate.

Every number printed is measured. Where a quantity was not measured it prints
as ``unknown`` -- never as zero.

Usage
-----
``python examples/cobordism/master_recursion.py --levels 2``
``python examples/cobordism/master_recursion.py --mode pencil --lambda 0.37``
"""

from __future__ import annotations

import argparse
import cmath
import math

import tessera as T

cob = T.cobordism
obs = T.observables

# The declared host seed, fixed so a run is reproducible from the command line
# alone. It labels an attempt; it does not certify one.
DECLARED_HOST_SEED = 20250401


# =====================================================================
# host
# =====================================================================

def build_host(n_refine, seed=DECLARED_HOST_SEED):
    """The refined closed-S4 host.

    The bare boundary of a 5-simplex -- the smallest closed 4-manifold
    triangulation -- refined by ``n_refine`` PreGeometric stellar Pachner
    adds, then given a mild deterministic non-uniform metric. Deliberately a
    standalone copy of the same construction the other drivers use, so an
    example never imports from the test tree and cannot drift when a fixture
    is edited.
    """
    st = T.Spacetime(T.Metric(True, T.Signature(4, T.Lorentzian)), T.CDT,
                     1.0, 1.0, T.PREFERRED, T.SimplexBoundarySphere(4))
    st.build()
    for edge in st.getEdgeList().toVector():
        edge.setLength(cmath.sqrt(complex(1.0)))
    applied = 0
    for step in range(seed, seed + n_refine * 4):
        move = T.AddMove(st, step, False, T.PachnerMode.PreGeometric, False)
        if move.propose() and move.apply():
            applied += 1
        if applied >= n_refine:
            break
    for index, edge in enumerate(st.getEdgeList().toVector()):
        edge.setLength(cmath.sqrt(complex(1.0 + 0.01 * (index % 6))))
    return st


# =====================================================================
# reporting helpers -- unknown is never zero
# =====================================================================

def show(value, digits=3):
    """A measured float, or ``unknown``. Never substitutes zero."""
    if value is None:
        return "unknown"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if math.isnan(number):
        return "unknown"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return f"{number:.{digits}g}"


def level_zero_supports(st, gamma):
    """P_0: the level-zero partition, discovered over the one-skeleton.

    Level zero is the only scale whose similarity graph is the geometry
    itself rather than a response operator, so it uses the spacetime entry
    point. Every scale above it partitions its own reduced operator.

    Modularity is a heuristic PROPOSAL generator: it proposes candidate
    supports and never vetoes an otherwise certified fiber. The resolution
    ``gamma`` is a declared knob of that proposal, not a physical parameter --
    at a low enough resolution the whole complex is one community, which
    leaves no interface cell and so nothing for the recursion to reduce.
    """
    modularity = obs.PersistentModularity.fromSpacetime(st)
    config = obs.PersistentModularityConfig()
    slice_ = modularity.discover(gamma, config)
    supports = [list(component.support) for component in slice_.components]
    return [support for support in supports if support]


# =====================================================================
# E_v: the certified isolated subspace of C_v
# =====================================================================

def certified_bands(st, level, supports, degree):
    """Assemble the boxed display's ``E_v`` from the fiber layer's bands.

    Each band is placed on the level's own fine coordinates by matching cells
    on their VERTEX SET -- never on an index or an imposed order -- and its
    isolation gaps and certificate travel with it onto the summand. An
    uncertified band is still summed and reported; it is never dropped, and
    it makes the sum's certificate fail to hold.
    """
    position_of = {frozenset(cell): index
                   for index, cell in enumerate(level.cellVertices)}
    tracker = obs.SpectralFiberTracker(st)
    bands = []
    for component, support in enumerate(supports):
        read = tracker.enumerateBands(list(support), degree)
        cells = list(read.cellVertices)
        for fiber in read.fibers:
            certificate = fiber.certificate()
            rank = int(certificate.rank)
            if rank == 0:
                continue
            frame = [0j] * (level.dimension * rank)
            placed = 0
            right = fiber.rightFrame()
            for row, cell in enumerate(cells):
                index = position_of.get(frozenset(cell))
                if index is None:
                    continue  # a cell this level does not carry
                placed += 1
                for column in range(rank):
                    frame[index * rank + column] = complex(right[row, column])
            if placed == 0:
                continue
            band = cob.RecursiveQuotient.CertifiedBand()
            band.component = component
            band.frame = frame
            band.rank = rank
            band.lowerGap = certificate.lowerGap
            band.upperGap = certificate.upperGap
            band.frequencyLower = certificate.frequencyLower
            band.frequencyUpper = certificate.frequencyUpper
            band.accepted = bool(certificate.accepted)
            band.certificate = certificate.certificate
            bands.append(band)
    return bands


# =====================================================================
# the recursion
# =====================================================================

def report_level(index, level):
    provenance = level.levelProvenance
    reduction = level.staticReduction()
    print(f"  level {index}: origin={provenance.origin.name} "
          f"dim={level.dimension} components={level.componentCount} "
          f"regime={level.regime.name}")
    print(f"    reduced to {len(reduction.coordinates)} coordinates; "
          f"certificate holds={reduction.certificate.holds()} "
          f"solve residual={show(reduction.solveResidual)}")
    if provenance.origin != cob.LevelOrigin.Base:
        print(f"    carried window=[{show(provenance.windowLower)}, "
              f"{show(provenance.windowUpper)}] "
              f"resonant={provenance.resonant} "
              f"producing certificate holds="
              f"{provenance.certificate.holds()}")


def report_fock(level, bands, max_terms):
    """The last two lines of the boxed display."""
    summary = level.certifiedFiberSum(bands)
    print(f"    labeled sum: {len(summary.summandCertificates)} certified "
          f"summands, nominal rank {summary.nominalRank}, effective "
          f"{summary.effectiveRank}")
    print(f"      worst isolation gap={show(summary.worstIsolationGap)} "
          f"gram defect={show(summary.gramDefect)} "
          f"all bands accepted={summary.allBandsAccepted} "
          f"kernel nullity={summary.quotientNullity}")
    stage = level.fockStage(summary, max_terms)
    dimension = ("inf" if math.isinf(stage.fockDimension)
                 else f"{stage.fockDimension:.6g}")
    print(f"      Fock stage: {stage.modes} modes, dim H = 2^{stage.modes} "
          f"= {dimension}, many-body spectrum "
          f"{'materialized' if stage.spectrumMaterialized else 'refused'}"
          f" ({len(stage.fockSpectrum)} values)")
    return summary, stage


def run(levels, mode, lam, window, refine, degree, gamma, max_terms):
    st = build_host(refine)
    supports = level_zero_supports(st, gamma)
    print(f"host: {len(st.getVertexList().toVector())} vertices, "
          f"{len(st.getEdgeList().toVector())} edges")
    print(f"P_0: {len(supports)} discovered components over the one-skeleton")

    level = cob.RecursiveQuotient.overVertexSupports(st, degree, supports)
    print("\nthe recursion:")
    report_level(0, level)

    bands = certified_bands(st, level, supports, degree)
    if bands:
        report_fock(level, bands, max_terms)
    else:
        print("    labeled sum: no band was enumerated on this host; the "
              "certified sum is REFUSED rather than filled with a "
              "retained-coordinate substitute")

    for index in range(1, levels + 1):
        if mode == "pencil":
            response = level.feshbach(lam, window[0], window[1])
            partition = cob.RecursiveQuotient.persistentPartition(
                response.response, len(response.coordinates))
            if len(partition) < 2:
                print(f"  level {index}: the response network has one "
                      f"component; the hierarchy has bottomed out here")
                break
            level = level.nextLevelAtLambda(partition, lam, window[0],
                                            window[1])
        else:
            partition = level.childPersistentPartition()
            if len(partition) < 2:
                print(f"  level {index}: the response network has one "
                      f"component; the hierarchy has bottomed out here")
                break
            level = level.nextLevel(partition)
        report_level(index, level)
        # Above level zero the coordinates are reduced coordinates, not cells,
        # so the fiber layer has nothing to enumerate on: the labeled sum of a
        # coarse level is its own retained-fiber sum, and it is reported as
        # exactly that rather than as a certified band sum.
        summary = level.labeledFiberSum()
        stage = level.fockStage(summary, max_terms)
        print(f"    retained-fiber sum: rank {summary.nominalRank}, "
              f"gram defect={show(summary.gramDefect)}, "
              f"certified bands={summary.fromCertifiedBands}")
        print(f"      Fock stage: {stage.modes} modes, many-body spectrum "
              f"{'materialized' if stage.spectrumMaterialized else 'refused'}")

    print("\nlineage of the deepest level:")
    for provenance in list(level.coordinateProvenance)[:6]:
        print(f"    {provenance}")
    if level.dimension > 6:
        print(f"    ... {level.dimension - 6} more")


def main():
    parser = argparse.ArgumentParser(
        description="Iterate the master recursive construction on a real "
                    "complex.")
    parser.add_argument("--levels", type=int, default=2,
                        help="levels to build above level zero (default 2)")
    parser.add_argument("--mode", choices=("static", "pencil"),
                        default="static",
                        help="the response step: the static lambda = 0 Schur "
                             "complement, or the energy-dependent pencil")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.37,
                        help="spectral parameter for --mode pencil")
    parser.add_argument("--window", type=float, nargs=2,
                        default=(0.0, 1.0), metavar=("LOWER", "UPPER"),
                        help="declared band window for --mode pencil")
    parser.add_argument("--refine", type=int, default=4,
                        help="PreGeometric stellar Pachner adds on the host")
    parser.add_argument("--degree", type=int, default=1,
                        help="form degree of the Hodge operator")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="modularity resolution for the level-zero "
                             "partition proposal")
    parser.add_argument("--max-terms", type=int, default=1 << 22,
                        help="term budget for the free many-body spectrum; "
                             "past it the enumeration refuses instead of "
                             "allocating 2^M values")
    args = parser.parse_args()
    run(args.levels, args.mode, complex(args.lam, 0.0), tuple(args.window),
        args.refine, args.degree, args.gamma, args.max_terms)


if __name__ == "__main__":
    main()
