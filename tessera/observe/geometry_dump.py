# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Campaign geometry dumps: the attempt's ONLY faithful record (#578 finding).

The engine build is NOT process-deterministic — identical fresh processes
diverge on the same seed (measured, not hypothetical) — so a base seed labels
an attempt, it does not reproduce it. The faithful record of a converged
state is therefore its GEOMETRY DUMP: schema-1 JSON with

* ``schema`` — 1 (this format);
* ``dimensions`` — the top-cell dimension;
* ``cells`` — top cells in INTRINSIC vertex order (each cell's stored order
  carries the orientation and is never sorted; the cell LIST is sorted by
  vertex-set key so the same state always serializes to the same bytes);
* ``edges`` — every edge as ``[src, tgt, re(l2), im(l2)]`` rows, sorted by
  the (min, max) id pair;
* ``vertex_times`` — sorted ``[vertex_id, time]`` pairs;

plus any attempt metadata the writer recorded (base_seed, verdicts, betti,
holes, singlet, ...). This is exactly the #562 campaign worker's dump format
(the frozen, sha256-manifested scripts of #579 under
``examples/cobordism/proton_campaign/`` are the schema's provenance; this
module is its IMPORTABLE home, and the writer is byte-identical to the frozen
one on the same state — guarded by test). ``Spacetime.fromCells`` + the
recorded lengths/times rebuild the exact state without re-running anything.

``rebuild_spacetime`` builds the skeleton C++-side (the ``ReggeSolver`` ctor
via ``build_complex``) — never a Python-driven materialization (the #451
lesson; the campaign analyzer's own rebuild is a separate frozen script).
"""
import json
import os

from tessera.observe.register import build_complex

#: The dump format this module reads and writes; bump on any format change.
GEOMETRY_SCHEMA = 1


def write_geometry_dump(st, path, meta=None):
    """Write ``st``'s faithful record to ``path`` (atomic; canonically
    ordered so the same state always serializes to the same bytes). ``meta``
    entries (attempt metadata) are merged in first — the geometry keys win on
    any collision. Reads state only; returns ``path``."""
    top = st.getTopSimplices()
    cells = sorted(([int(v.getId()) for v in c.getVertices()] for c in top),
                   key=sorted)
    times = {}
    for c in top:
        for v in c.getVertices():
            times[int(v.getId())] = float(v.getTime())
    edges = sorted(([int(e.getSource().getId()), int(e.getTarget().getId()),
                     e.getSquaredLength().real, e.getSquaredLength().imag]
                    for e in st.getEdgeList().toVector()),
                   key=lambda r: (min(r[0], r[1]), max(r[0], r[1])))
    record = dict(meta or {})
    record.update({
        "schema": GEOMETRY_SCHEMA,
        "dimensions": (len(cells[0]) - 1) if cells else 0,
        "cells": cells,
        "edges": edges,
        "vertex_times": sorted(times.items()),
    })
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(record, fh)
    os.replace(tmp, path)
    return path


def load_geometry_dump(path):
    """Load and validate a schema-1 geometry dump. Raises ``ValueError`` on a
    missing/unknown schema or missing geometry keys — a record that is not a
    faithful geometry dump is never silently half-read."""
    with open(path) as fh:
        dump = json.load(fh)
    schema = dump.get("schema")
    if schema != GEOMETRY_SCHEMA:
        raise ValueError(
            f"{path}: geometry dump schema {schema!r} is not the supported "
            f"schema {GEOMETRY_SCHEMA}")
    missing = [k for k in ("dimensions", "cells", "edges", "vertex_times")
               if k not in dump]
    if missing:
        raise ValueError(f"{path}: geometry dump is missing {missing}")
    return dump


def rebuild_spacetime(dump):
    """A live ``Spacetime`` carrying the dumped final state: ``fromCells`` on
    the top cells (intrinsic vertex order preserved), then the recorded
    per-vertex times and per-edge complex squared lengths, with the skeleton
    materialized C++-side (``ReggeSolver`` ctor)."""
    edges = {}
    for u, v, re_l2, im_l2 in dump["edges"]:
        key = (min(int(u), int(v)), max(int(u), int(v)))
        edges[key] = complex(re_l2, im_l2)
    times = {int(vid): float(t) for vid, t in dump["vertex_times"]}
    return build_complex(dump["cells"], edges, vertex_times=times,
                         dimensions=int(dump["dimensions"]))


def verify_rebuild(st, dump):
    """Check the rebuilt state carries exactly the dumped complex: same top
    cells (as vertex sets), same edge squared lengths, and — when the dump's
    metadata recorded them — the same combinatorial reads. Returns a
    ``{key: (expected, actual)}`` mismatch dict (empty = verified)."""
    import tessera as T
    cob = T.cobordism

    mismatches = {}
    cells = sorted(sorted(int(v.getId()) for v in c.getVertices())
                   for c in st.getTopSimplices())
    dumped = sorted(sorted(int(v) for v in c) for c in dump["cells"])
    if cells != dumped:
        mismatches["cells"] = (len(dumped), len(cells))
    lengths = {}
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        l2 = e.getSquaredLength()
        lengths[(min(a, b), max(a, b))] = (l2.real, l2.imag)
    dumped_lengths = {(min(int(u), int(v)), max(int(u), int(v))): (re, im)
                      for u, v, re, im in dump["edges"]}
    if lengths != dumped_lengths:
        wrong = sum(1 for k, val in dumped_lengths.items()
                    if lengths.get(k) != val)
        mismatches["edge_lengths"] = (len(dumped_lengths), wrong)
    checks = {
        "betti": lambda: list(cob.MultiCobordism.betti(st)),
        "holes": lambda: len(cob.MultiCobordism.emergent_holes(st, 3)),
    }
    for key, compute in checks.items():
        if key in dump:
            value = compute()
            if dump[key] != value:
                mismatches[key] = (dump[key], value)
    return mismatches
