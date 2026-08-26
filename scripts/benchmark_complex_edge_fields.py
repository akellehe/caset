#!/usr/bin/env python3
"""Scaling benchmark for direct complex edge fields (#882).

Measures construction, one z update, one U update, an exact in-memory
snapshot/rollback, face-holonomy products, and schema-6-style checkpoint size
at three or more requested complex sizes. No length, additive phase, root, or
logarithm view participates in a timed path.
"""

import argparse
import json
import statistics
import time

import tessera


def _median(callable_, repeats):
    samples = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = callable_()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), result


def _cells(edge_target):
    # A connected strip of oriented triangles with O(edge_target) edges.
    triangle_count = max(1, edge_target // 2)
    return [[index, index + 1, index + 2]
            for index in range(triangle_count)]


def _build(cells):
    return tessera.Spacetime.fromCellsWithFields(
        2, cells, squaredLength=complex(-0.875, 1.625),
        canonicalLink=complex(0.8, -0.45))


def _snapshot(st):
    return [(edge, complex(edge.squaredLength()),
             complex(edge.canonicalLink()))
            for edge in st.getEdgeList().toVector()]


def _restore(snapshot):
    for edge, z, link in snapshot:
        edge.setSquaredLength(z)
        edge.setCanonicalLink(link)


def _record(st, cells):
    edges = []
    for edge in st.getEdgeList().toVector():
        a, b = sorted((int(edge.getSource().getId()),
                       int(edge.getTarget().getId())))
        z = complex(edge.squaredLength())
        link = complex(edge.canonicalLink())
        edges.append({
            "a": a,
            "b": b,
            "canonical_orientation": [a, b],
            "z": [z.real, z.imag],
            "U": [link.real, link.imag],
        })
    return json.dumps({"schema_version": 6,
                       "raw_complex": {"dimensions": 2,
                                       "cells": cells,
                                       "edges": sorted(
                                           edges,
                                           key=lambda item:
                                           (item["a"], item["b"])),
                                       "boundary_components": [],
                                       "cooriented_cuts": []}},
                      sort_keys=True, separators=(",", ":"))


def benchmark(size, repeats, holonomy_repeats):
    cells = _cells(size)
    construction_s, st = _median(lambda: _build(cells), repeats)
    edges = st.getEdgeList().toVector()
    edge = edges[len(edges) // 2]
    z0 = complex(edge.squaredLength())
    u0 = complex(edge.canonicalLink())

    def update_z():
        edge.setSquaredLength(z0 + complex(0.001, -0.002))
        edge.setSquaredLength(z0)

    def update_u():
        edge.setCanonicalLink(u0 * complex(1.0003, -0.0007))
        edge.setCanonicalLink(u0)

    z_update_s, _ = _median(update_z, max(repeats, 10))
    u_update_s, _ = _median(update_u, max(repeats, 10))

    def snapshot_rollback():
        saved = _snapshot(st)
        probe = saved[len(saved) // 2][0]
        probe.setSquaredLength(complex(3.25, -7.5))
        probe.setCanonicalLink(complex(-0.4, 1.2))
        _restore(saved)
        return saved

    rollback_s, saved = _median(snapshot_rollback, repeats)
    assert all(edge_.squaredLength() == z and edge_.canonicalLink() == link
               for edge_, z, link in saved)

    triangle = cells[len(cells) // 2]

    def holonomies():
        product = 1 + 0j
        for _ in range(holonomy_repeats):
            product = st.faceHolonomy(triangle)
        return product

    holonomy_s, product = _median(holonomies, repeats)
    record_s, record = _median(lambda: _record(st, cells), repeats)
    return {
        "requested_edges": size,
        "vertices": len(st.getVertexList().toVector()),
        "edges": len(edges),
        "top_cells": len(cells),
        "construction_median_s": construction_s,
        "z_roundtrip_median_s": z_update_s,
        "U_roundtrip_median_s": u_update_s,
        "snapshot_rollback_median_s": rollback_s,
        "face_holonomy_median_s": holonomy_s / holonomy_repeats,
        "face_holonomy_batch_product": [product.real, product.imag],
        "checkpoint_encode_median_s": record_s,
        "checkpoint_bytes": len(record.encode("utf-8")),
        "exact_field_roundtrip": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="64,512,4096",
                        help="comma-separated target edge counts")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--holonomy-repeats", type=int, default=1000)
    args = parser.parse_args(argv)
    sizes = [int(value) for value in args.sizes.split(",") if value.strip()]
    if len(sizes) < 3 or any(value < 1 for value in sizes):
        parser.error("--sizes needs at least three positive integers")
    report = {
        "benchmark": "direct_complex_edge_fields",
        "coordinate_contract": "z in C; U in C*",
        "sizes": [benchmark(size, max(1, args.repeats),
                            max(1, args.holonomy_repeats))
                  for size in sizes],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
