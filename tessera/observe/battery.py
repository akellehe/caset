# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The observable battery: a registry + ``measure_all`` (#583).

``Battery`` holds named observables and measures them all against one
``Register``, producing ONE flat JSON-able record: the register census block,
then per-observable ``status: measured`` entries (record + GAUGE/RELABEL gate
residuals) or ``status: skipped(reason)`` entries — an inapplicable
observable is a reported skip, never a crash and never a silent hole in the
record.

The default ``battery`` instance carries the three migrated readout surfaces
(``SingletDiagnostic``, ``BlockResiduals``, ``MassRadius``,
``PairLoopFlavor``); the campaign analyzer imports it directly::

    from tessera.observe import Register, battery
    record = battery.measure_all(Register(st, count=3), provenance=None)
"""
from tessera.observe.adapters import DEFAULT_OBSERVABLES
from tessera.observe.observable import ensure_jsonable
from tessera.observe.register import GATE_SEED, GAUGE_THETA


class Battery:
    """An ordered registry of ``Observable`` instances."""

    def __init__(self, observables=()):
        self._observables = []
        for observable in observables:
            self.register(observable)

    def register(self, observable):
        """Add an observable (unique non-empty ``name`` required). Returns it,
        so the call composes as a decorator on instances-by-construction."""
        if not observable.name:
            raise ValueError("observable has no name")
        if any(o.name == observable.name for o in self._observables):
            raise ValueError(f"duplicate observable name: {observable.name}")
        self._observables.append(observable)
        return observable

    @property
    def observables(self):
        return tuple(self._observables)

    def measure_all(self, register, provenance=None, gates=True,
                    gate_seed=GATE_SEED, gauge_theta=GAUGE_THETA):
        """Measure every registered observable against ``register``.

        Returns the flat battery record::

            {
              "register": {...census, b3, divergence flag...},
              "provenance_keys": [...],
              "observables": {
                name: {"status": "measured", "record": {...},
                       "gates": {"gauge_delta": .., "relabel_delta": ..,
                                 "gauge_ok": .., "relabel_ok": ..,
                                 "gate_tol": ..}}
                    | {"status": "skipped(<reason>)", "reason": "<reason>"},
              },
            }

        With ``gates=True`` the GAUGE and RELABEL transforms are built ONCE
        (lazily, on the first measured observable) and shared across the
        battery — one relabeled rebuild per record, not per observable.
        Provenance is passed through to each observable, which maps it
        through the relabel permutation itself (``transform_provenance``).
        """
        record = {
            "register": register.summary(),
            "provenance_keys": sorted(provenance) if provenance else [],
            "observables": {},
        }
        gauge_register = None
        relabel = None
        for observable in self._observables:
            reason = observable.skip_reason(register, provenance)
            if reason is not None:
                record["observables"][observable.name] = {
                    "status": f"skipped({reason})",
                    "reason": reason,
                }
                continue
            if not gates:
                record["observables"][observable.name] = {
                    "status": "measured",
                    "record": observable.measure(register, provenance),
                }
                continue
            if gauge_register is None:
                gauge_register = register.gauged(gauge_theta)
                relabel = register.relabeled(gate_seed)
            record["observables"][observable.name] = observable.gated_measure(
                register, provenance,
                gauge_register=gauge_register, relabel=relabel)
        return ensure_jsonable(record)


#: The default battery — the three migrated readout surfaces, in order.
battery = Battery(cls() for cls in DEFAULT_OBSERVABLES)


def measure_all(register, provenance=None, **kwargs):
    """``battery.measure_all`` on the default battery (the analyzer-facing
    one-call surface)."""
    return battery.measure_all(register, provenance=provenance, **kwargs)
