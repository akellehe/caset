# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The battery's observables — the three readout surfaces migrated by
composition (#583).

Each adapter composes its source machinery (never re-derives it): the
migration contract is that an adapter's values EQUAL the source's on the same
fixtures, proven by the migration-equivalence tests. Records contain only
gauge- and relabel-invariant JSON-able channels (complex values as explicit
re/im pairs; covariant raw periods reported in the propagation-root-fixed
convention).

* ``SingletDiagnostic`` — the #574 surface: the relabeling-invariant singlet
  ``r_state`` of ``Proton.singlet()`` against the whole complex (DIAGNOSTIC
  only — it never steers anything), plus the hole/Betti census with the
  ``holes_vs_b3_divergent`` flag.
* ``BlockResiduals`` — the #574 per-output-block reads: each provenance block
  (vertex region + target, e.g. ``ProtonIngredients.output_blocks()`` or the
  campaign record) scored against its OWN sub-complex exactly as
  ``baryonResidual()``/``antibaryonResidual()`` score it (uniform-metric
  ``fromCells`` sub-complex + ``r_state``; the full leak ``‖target‖²`` when
  the region contains no full cell).
* ``MassRadius`` — the #575 battery: closed-fan interior census, intensive +
  extensive masses, dual-volume radius with the primal cross-check,
  participation/localization, and the r·m table with its definitional spread.
* ``PairLoopFlavor`` — the #576 read: dual-basis pair-loop charges, the 2:1
  multiplicity criterion, the duality residual — with the
  odd-one-out-vs-spectator criterion evaluated ONLY when provenance supplies
  the diquark pair (build history travels via the campaign record /
  geometry-dump metadata; it is never guessed).
"""
import tessera as _T

from tessera.observe import mass_radius as _mass_radius
from tessera.observe import pair_loop_flavor as _pair_loop
from tessera.observe.observable import Observable, split_complex

_cob = _T.cobordism


class SingletDiagnostic(Observable):
    """The whole-complex singlet diagnostic + hole/Betti census (#574)."""

    name = "singlet_diagnostic"
    requires = {}
    #: r_state is an eigensolve read re-run on a relabeled rebuild; direct
    #: values sit at ~1e-16 on the fixtures.
    gate_tol = 1e-9

    def measure(self, register, provenance=None):
        target = _cob.Proton.singlet()
        residual = float(_cob.MultiCobordism.r_state(
            register.st, register.degree, target))
        return {
            "singlet_residual": residual,
            "holes_used": len(register.holes),
            "holes_total": register.holes_total,
            "b3": register.b3,
            "betti": list(register.betti),
            "holes_vs_b3_divergent": register.holes_vs_b3_divergent,
            **split_complex("target", target),
        }


class BlockResiduals(Observable):
    """Per-output-block carry residuals from provenance blocks (#574).

    Provenance shape: ``provenance["blocks"]`` is a list of blocks, each with
    ``label`` (optional), ``vertices`` (the block's emergent vertex region)
    and its target — either ``target`` (a complex sequence, e.g. straight off
    ``ProtonIngredients.output_blocks()``) or the JSON-able ``target_re`` /
    ``target_im`` pair (a campaign record). The residual mirrors
    ``ProtonIngredients::outputBlockResidual`` exactly: the ambient top cells
    whose vertices ALL lie in the region form the block's own sub-complex
    (uniform-metric ``fromCells``, matching how the drive's ``rU`` scored the
    block), scored with ``MultiCobordism.r_state``; an empty region reports
    the full leak ``‖target‖²``.
    """

    name = "block_residuals"
    requires = {"needs_provenance": ("blocks",)}
    gate_tol = 1e-9

    @staticmethod
    def _block_target(block, index):
        if "target" in block:
            return [complex(t) for t in block["target"]]
        if "target_re" in block and "target_im" in block:
            return [complex(re, im)
                    for re, im in zip(block["target_re"], block["target_im"])]
        raise ValueError(
            f"provenance block {index} has neither 'target' nor "
            f"'target_re'/'target_im'")

    def measure(self, register, provenance=None):
        rows = []
        for index, block in enumerate(provenance["blocks"]):
            vertices = {int(v) for v in block["vertices"]}
            target = self._block_target(block, index)
            cells_inside = []
            for cell in register.cells:
                vids = [v.getId() for v in cell.getVertices()]
                if all(v in vertices for v in vids):
                    cells_inside.append(vids)
            target_norm2 = float(sum(abs(t) ** 2 for t in target))
            if not cells_inside:
                residual = target_norm2  # the full leak — nothing carries it
            else:
                sub = _T.Spacetime.fromCells(
                    register.dimensions, cells_inside, 1.0, 0.0)
                residual = float(_cob.MultiCobordism.r_state(
                    sub, register.degree, target))
            rows.append({
                "label": str(block.get("label", f"block{index}")),
                "n_region_vertices": len(vertices),
                "n_cells_in_region": len(cells_inside),
                "full_leak": not cells_inside,
                "residual": residual,
                "target_norm2": target_norm2,
                **split_complex("target", target),
            })
        return {"n_blocks": len(rows), "blocks": rows}

    def transform_provenance(self, provenance, perm):
        """Block regions are vertex-id sets — map them through the RELABEL
        permutation (targets and labels are id-free).

        An emergent block region can reference vertices that are no longer in
        any top cell (surgical moves orphan them); such ids are INERT in the
        residual (they never match a cell vertex — the C++ read ignores them
        the same way). The permutation maps the live vertex-id set onto
        itself, so an inert id kept as itself stays inert on the relabeled
        complex — ``perm.get(v, v)`` is exact, and the region's size (a
        reported channel) is preserved."""
        blocks = [dict(block,
                       vertices=[perm.get(int(v), int(v))
                                 for v in block["vertices"]])
                  for block in provenance["blocks"]]
        return dict(provenance, blocks=blocks)


class MassRadius(Observable):
    """The #575 mass/radius battery on the relaxed 4D interior."""

    name = "mass_radius"
    requires = {"dimensions": 4}
    #: Geometric aggregates (sums over hundreds of hinges, r·m products up to
    #: ~1e3) are re-summed in a different container order on the relabeled
    #: rebuild — order-ULP noise scales with the magnitudes, so this gate
    #: tolerance is absolute-loose while the raw residuals stay reported.
    gate_tol = 1e-6

    @staticmethod
    def _shell_key(shell):
        return "unshelled" if shell is None else str(int(shell))

    def measure(self, register, provenance=None):
        reading = _mass_radius.measure(register.st, register.holes)
        census = {k: v for k, v in reading["census"].items()
                  if k != "boundary_tets"}  # vertex-id tuples: label-bound
        mass = dict(reading["mass"])
        mass["shell_means"] = {self._shell_key(k): v
                               for k, v in mass["shell_means"].items()}
        localization = dict(reading["localization"])
        localization["shell_profile"] = {
            self._shell_key(k): v
            for k, v in localization["shell_profile"].items()}
        return {
            "census": census,
            "mass": mass,
            "radius": reading["radius"],
            "localization": localization,
            "rm": reading["rm"],
            "n_holes": reading["n_holes"],
        }


class PairLoopFlavor(Observable):
    """The #576 pair-loop dual-basis flavor read on the 3-hole register."""

    name = "pair_loop_flavor"
    requires = {"min_holes": 3, "dimensions": 4}
    #: Direct reads reproduce to ~1e-16; the derived clustering ratio rho
    #: divides two small charge differences and amplifies eigensolve roundoff
    #: to ~1e-13 (the source's RHO_GATE_TOL) — one tolerance covers the
    #: record's every leaf, raw residuals reported alongside.
    gate_tol = _pair_loop.RHO_GATE_TOL

    def measure(self, register, provenance=None):
        read = register.derived(
            ("pair_loop_read", register.target),
            lambda: _pair_loop.joint_read(
                register.st, register.holes, register.target,
                es=register.es,
                sigma=register.eps,
                weights=register.derived(
                    "hodge_weights",
                    lambda: _cob.HodgeLaplacian(register.st).weights(
                        register.degree)),
                cell_index=register.derived(
                    "cell_index",
                    lambda: {frozenset(t): i
                             for i, t in enumerate(register.es.cellSimplices())}),
            ))
        diquark_pair = None
        if provenance is not None and provenance.get("diquark_pair") is not None:
            diquark_pair = tuple(int(i) for i in provenance["diquark_pair"])
        verdict = _pair_loop.evaluate_criteria(read, diquark_pair=diquark_pair)

        # The oriented periods are U(1)-covariant (w -> e^{i theta} w under the
        # GAUGE knob) and flip with the endSignCovector propagation root under
        # RELABEL; dividing out w0's unit phase reports them in the one
        # propagation-root-fixed convention, so every record leaf is invariant.
        w = list(read["w"])
        phase0 = (w[0] / abs(w[0])) if abs(w[0]) > 0 else 1.0
        w_fixed = [wi / phase0 for wi in w]
        loop_w_fixed = [wi / phase0 for wi in read["loop_w"]]

        record = {
            "r_u": float(read["r_u"]),
            "q": [float(x) for x in read["q"]],
            "loop_q": [float(x) for x in read["loop_q"]],
            "dual_residual": [float(x) for x in read["dual_residual"]],
            "pair_loops": [list(p) for p in _pair_loop.PAIR_LOOPS],
            "odd_loop": list(verdict["odd_loop"]),
            "dual_hole": int(verdict["dual_hole"]),
            "rho": float(verdict["rho"]),
            "rho_max": float(_pair_loop.RHO_MAX),
            "multiplicity_2_1": bool(verdict["multiplicity_2_1"]),
            "odd_is_diquark_loop": verdict["odd_is_diquark_loop"],
            "odd_is_diquark_loop_status": (
                "evaluated" if diquark_pair is not None
                else "not_evaluable(no_provenance)"),
            **split_complex("w", w_fixed),
            **split_complex("loop_w", loop_w_fixed),
        }
        return record


#: The battery's default line-up, in measurement order.
DEFAULT_OBSERVABLES = (SingletDiagnostic, BlockResiduals, MassRadius,
                       PairLoopFlavor)
