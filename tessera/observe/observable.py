# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The thin gated ``Observable`` base (#583).

An Observable is a pure post-hoc reader over a ``Register``: it never shapes
the lattice, never loops, never steers — it measures and reports. The base is
deliberately thin:

* ``name`` — the record key.
* ``requires`` — declarative preconditions the battery evaluates BEFORE
  measuring, so an inapplicable observable is skipped WITH A REASON instead
  of crashing: ``min_holes`` (register holes needed), ``dimensions`` (top-cell
  dimension), ``needs_provenance`` (True, or the provenance keys required —
  e.g. the diquark pair or the output blocks travel via the campaign record /
  geometry-dump metadata, never guessed and never re-derived by re-running a
  build), ``needs_causal_content`` (the observable reads causal structure;
  all-spacelike specimens skip it).
* ``measure(register, provenance=None) -> dict`` — the pure read. Records are
  JSON-able by contract: float/int/bool/str/None and (nested) lists/dicts
  only. Complex values are stored as explicit re/im pairs via
  ``split_complex`` (the #580 propagation discipline at the reporting layer:
  a channel is real BY CONSTRUCTION or it carries both parts explicitly —
  nothing is silently ``.real``-ed).
* ``gated_measure`` — the GAUGE/RELABEL harness: re-measure on the
  transformed register and compare EVERY numeric leaf of the record
  (``report_delta``, adapted from the C_ij chamber readout's every-channel
  gate). Records must therefore contain only gauge- and relabel-invariant
  channels; covariant raw values (e.g. complex periods) are reported in a
  gauge-fixed convention by their adapter. The gates are post-hoc validation,
  never a loop condition.

The self-test pattern: ``report_delta`` flags a perturbed channel (any leaf,
however deep) with a nonzero delta, mismatched strings/None with ``inf``, and
raises on a record-shape mismatch — the framework tests prove all three, so a
silently-passing gate cannot be a comparison that never happened.
"""
import json
import math

from tessera.observe.register import GATE_SEED, GAUGE_THETA


def split_complex(name, value):
    """A complex scalar or sequence as the two explicit JSON-able leaves
    ``{name}_re`` / ``{name}_im`` — the one naming convention for complex
    record channels (#580 discipline: the imaginary part is real physics and
    is always carried, never silently dropped)."""
    try:
        values = list(value)
    except TypeError:
        z = complex(value)
        return {f"{name}_re": z.real, f"{name}_im": z.imag}
    zs = [complex(v) for v in values]
    return {f"{name}_re": [z.real for z in zs],
            f"{name}_im": [z.imag for z in zs]}


def ensure_jsonable(record):
    """Assert ``record`` serializes to JSON (NaN/inf allowed, Python-style)
    and return it. Raises ``TypeError`` on any non-JSON-able leaf — complex
    values must go through ``split_complex`` first."""
    json.dumps(record)
    return record


def report_delta(a, b):
    """The max absolute difference over every numeric leaf of two nested
    records (the C_ij chamber readout's every-channel gate metric, adapted):

    * dicts must have identical keys (a shape mismatch raises ``KeyError``);
    * lists/tuples must have identical lengths (else ``inf``);
    * strings and ``None`` must be equal (else ``inf`` — a changed status IS
      a flagged channel);
    * bools compare as 0/1; numbers as ``|a - b|``; two NaNs agree (delta 0 —
      NaN is a legitimate reported value, e.g. a not-applicable reading), a
      NaN against a number is ``inf`` (an appeared/vanished reading is a
      flagged channel, never a silent NaN poisoning the max).
    """
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            return float("inf")
        if set(a) != set(b):
            raise KeyError(
                f"record keys differ: {sorted(a)} vs {sorted(b)}")
        deltas = [report_delta(a[k], b[k]) for k in a]
        return max(deltas) if deltas else 0.0
    if isinstance(a, str) or isinstance(b, str):
        return 0.0 if a == b else float("inf")
    if a is None or b is None:
        return 0.0 if (a is None and b is None) else float("inf")
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))):
            return float("inf")
        if len(a) != len(b):
            return float("inf")
        deltas = [report_delta(x, y) for x, y in zip(a, b)]
        return max(deltas) if deltas else 0.0
    if isinstance(a, bool) or isinstance(b, bool):
        return float(abs(int(a) - int(b)))
    x, y = float(a), float(b)
    if math.isnan(x) or math.isnan(y):
        return 0.0 if (math.isnan(x) and math.isnan(y)) else float("inf")
    return abs(x - y)


class Observable:
    """Base class for gated post-hoc observables. Subclasses set ``name``,
    ``requires`` and ``gate_tol``, implement ``measure``, and (when their
    provenance references vertex ids) override ``transform_provenance``."""

    #: The record key; must be unique within a battery.
    name = None

    #: Declarative preconditions — see the module docstring for the keys.
    requires = {}

    #: Both gate residuals must stay below this for the ``*_ok`` verdicts.
    #: Direct period/charge reads sit at ~1e-16; derived ratios (rho) amplify
    #: eigensolve roundoff to ~1e-13; geometric aggregates re-summed in a
    #: relabeled container order carry order-ULP noise — adapters pick the
    #: tolerance their channels warrant and the raw residuals are always
    #: reported alongside.
    gate_tol = 1e-9

    def measure(self, register, provenance=None):
        """The pure read: a JSON-able record of invariant channels."""
        raise NotImplementedError

    def skip_reason(self, register, provenance=None):
        """The reason this observable cannot measure this register (a string
        the battery reports), or None when it can."""
        requires = self.requires
        min_holes = requires.get("min_holes", 0)
        if len(register.holes) < min_holes:
            return f"holes={len(register.holes)} < min_holes={min_holes}"
        dimensions = requires.get("dimensions")
        if dimensions is not None and register.dimensions != dimensions:
            return f"dimensions={register.dimensions} != {dimensions}"
        needs = requires.get("needs_provenance")
        if needs:
            if provenance is None:
                return "no_provenance"
            if needs is not True:
                keys = (needs,) if isinstance(needs, str) else tuple(needs)
                missing = sorted(k for k in keys if k not in provenance)
                if missing:
                    return f"provenance_missing:{','.join(missing)}"
        if requires.get("needs_causal_content") and not register.causal_content:
            return "no_causal_content"
        return None

    def transform_provenance(self, provenance, perm):
        """Map provenance through the RELABEL permutation so the gate compares
        like with like. The default is identity — correct for provenance that
        references hole INDICES or carries no vertex ids; provenance carrying
        vertex ids (e.g. block regions) must override."""
        return provenance

    def gated_measure(self, register, provenance=None, gauge_register=None,
                      relabel=None):
        """Measure, then run both gates and report their residuals:

        * GAUGE — re-measure with the register target rotated by the global
          U(1) phase (``Register.gauged``); every numeric leaf must agree.
        * RELABEL — re-measure on the vertex-relabeled rebuild
          (``Register.relabeled``), with provenance mapped through the
          permutation; every numeric leaf must agree.

        ``gauge_register`` / ``relabel`` (the ``(register, perm)`` pair) let a
        battery share one transformed register across its observables.
        Returns the full battery entry:
        ``{"status": "measured", "record": ..., "gates": ...}``.
        """
        record = ensure_jsonable(self.measure(register, provenance))
        if gauge_register is None:
            gauge_register = register.gauged(GAUGE_THETA)
        gauge_delta = report_delta(
            record, self.measure(gauge_register, provenance))
        if relabel is None:
            relabel = register.relabeled(GATE_SEED)
        relabeled_register, perm = relabel
        relabel_delta = report_delta(
            record, self.measure(relabeled_register,
                                 self.transform_provenance(provenance, perm)))
        gates = {
            "gauge_delta": float(gauge_delta),
            "relabel_delta": float(relabel_delta),
            "gauge_ok": bool(gauge_delta <= self.gate_tol),
            "relabel_ok": bool(relabel_delta <= self.gate_tol),
            "gate_tol": float(self.gate_tol),
        }
        return {"status": "measured", "record": record, "gates": gates}
