# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""tessera.observe — the observable measurement layer (#583, part of #559).

The emergent-proton readouts as first-class gated Observables: one shared
read context (``Register``), one gate harness (``Observable.gated_measure``
— GAUGE/RELABEL over every numeric leaf), one record shape (JSON-able,
complex values as explicit re/im pairs), and one battery call for the
campaign analyzer::

    from tessera.observe import Register, battery
    record = battery.measure_all(Register(st, count=3), provenance=None)

Observables are read after the fact — never loop conditions; nothing here
shapes the lattice. The faithful input path for campaign attempts is the
geometry dump (``load_geometry_dump`` / ``rebuild_spacetime``): the engine
build is not process-deterministic, so a base seed labels an attempt — it
does not reproduce it.
"""
from tessera.observe.adapters import (          # noqa: F401
    DEFAULT_OBSERVABLES,
    BlockResiduals,
    MassRadius,
    PairLoopFlavor,
    SingletDiagnostic,
)
from tessera.observe.battery import Battery, battery, measure_all  # noqa: F401
from tessera.observe.geometry_dump import (     # noqa: F401
    GEOMETRY_SCHEMA,
    load_geometry_dump,
    rebuild_spacetime,
    verify_rebuild,
    write_geometry_dump,
)
from tessera.observe.observable import (        # noqa: F401
    Observable,
    ensure_jsonable,
    report_delta,
    split_complex,
)
from tessera.observe.register import (          # noqa: F401
    GATE_SEED,
    GAUGE_THETA,
    OMEGA,
    SINGLET,
    Register,
    build_complex,
    induced_orientation_signs,
    register_holes,
)
