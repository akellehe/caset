# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The state record a build writes, and the orientation it can be read for.

A finished complex is `GeometryState`: top cells in their INTRINSIC vertex
order, every edge's complex squared length, and per-vertex times. That is
enough to rebuild the exact state — and the intrinsic order is what makes
orientation recoverable, so these cells are never sorted internally (only the
lists themselves are ordered canonically, so one state always serializes to one
sequence of bytes).

`Orientation` reads a coherent orientation off such a complex. The engine's
`ChainComplex.fundamentalClass` cannot do this here: it requires a CLOSED
oriented d-manifold (dim ker ∂_d = b_d = 1), and every complex a proton build
produces is a d-ball with boundary, where b_d = 0 and no absolute fundamental
class exists. What does exist is an orientation relative to the boundary, and
it is determined by propagation: each top cell's ordered vertices induce an
orientation on each of its facets, two cells sharing an interior facet are
coherently oriented exactly when their induced orientations there are opposite,
and that rule fixes every sign from one seed. A contradiction around a cycle
means the complex is non-orientable.
"""
import json
import os

import tessera

_observables = tessera.observables


class GeometryState:
    """Read and write the geometry record (schema 1)."""

    SCHEMA = 1

    @staticmethod
    def cells(spacetime):
        """Top cells as vertex-id lists in intrinsic order, canonically
        ordered as a LIST (by sorted vertex set) but never sorted within a
        cell — the internal order carries the orientation."""
        return sorted(([int(v.getId()) for v in cell.getVertices()]
                       for cell in spacetime.getTopSimplices()), key=sorted)

    @staticmethod
    def write(spacetime, path, meta=None):
        """Write the state to `path` atomically, returning the path. `meta` is
        merged in first, so the schema and geometry keys always win."""
        cells = GeometryState.cells(spacetime)
        times = {}
        for cell in spacetime.getTopSimplices():
            for vertex in cell.getVertices():
                times[int(vertex.getId())] = float(vertex.getTime())
        edges = sorted(([int(edge.getSource().getId()),
                         int(edge.getTarget().getId()),
                         (edge.getLength() ** 2).real,
                         (edge.getLength() ** 2).imag]
                        for edge in spacetime.getEdgeList().toVector()),
                       key=lambda row: (min(row[0], row[1]), max(row[0], row[1])))
        record = dict(meta or {})
        record.update({
            "schema": GeometryState.SCHEMA,
            "dimensions": (len(cells[0]) - 1) if cells else 0,
            "cells": cells,
            "edges": edges,
            "vertex_times": sorted(times.items()),
        })
        temporary = path + ".tmp"
        with open(temporary, "w") as handle:
            json.dump(record, handle)
        os.replace(temporary, path)
        return path

    @staticmethod
    def load(path):
        """Load and validate a state record; a record of another schema, or one
        missing geometry, raises rather than being half-read."""
        with open(path) as handle:
            record = json.load(handle)
        if record.get("schema") != GeometryState.SCHEMA:
            raise ValueError(
                f"{path}: state schema {record.get('schema')!r} is not the "
                f"supported schema {GeometryState.SCHEMA}")
        missing = [key for key in ("dimensions", "cells", "edges", "vertex_times")
                   if key not in record]
        if missing:
            raise ValueError(f"{path}: state record is missing {missing}")
        return record

    @staticmethod
    def rehydrate(record):
        """A live `Spacetime` carrying the recorded state. Never re-runs a
        build — the engine is not process-deterministic, so a record is the
        only faithful way back to a state."""
        edges = {}
        for source, target, real_part, imaginary_part in record["edges"]:
            key = (min(int(source), int(target)), max(int(source), int(target)))
            edges[key] = complex(real_part, imaginary_part)
        times = {int(vertex): float(time)
                 for vertex, time in record["vertex_times"]}
        return _observables.LiveComplex.load(
            [[int(v) for v in cell] for cell in record["cells"]],
            edges, times, int(record["dimensions"]))


class Orientation:
    """A coherent orientation of a complex that may have boundary.

    `signs` maps each top cell (as its intrinsic vertex tuple) to ε = ±1;
    `orientable` is False when propagation met a contradiction, in which case
    `signs` holds the partial assignment that led there. `boundary_facets`
    maps each boundary facet (sorted vertex tuple) to the orientation ∂W
    inherits from the single cell that carries it.
    """

    def __init__(self, cells):
        self.cells = [tuple(int(v) for v in cell) for cell in cells]
        self.signs = {}
        self.orientable = True
        self.boundary_facets = {}
        self._build()

    @staticmethod
    def _parity(sequence):
        """Parity of the permutation sorting `sequence`: +1 even, -1 odd.
        Counts inversions directly — these facets have a handful of entries."""
        inversions = 0
        for i in range(len(sequence)):
            for j in range(i + 1, len(sequence)):
                if sequence[i] > sequence[j]:
                    inversions += 1
        return 1 if inversions % 2 == 0 else -1

    @classmethod
    def _induced(cls, cell):
        """Every facet of `cell` with the orientation the cell induces on it,
        expressed against the sorted representative of that facet: dropping
        index i contributes (-1)^i, and re-sorting the remaining vertices
        contributes the parity of that permutation."""
        induced = []
        for i in range(len(cell)):
            facet = cell[:i] + cell[i + 1:]
            sign = (1 if i % 2 == 0 else -1) * cls._parity(facet)
            induced.append((tuple(sorted(facet)), sign))
        return induced

    def _build(self):
        # facet -> [(cell index, induced sign), ...]
        incidence = {}
        for index, cell in enumerate(self.cells):
            for facet, sign in self._induced(cell):
                incidence.setdefault(facet, []).append((index, sign))

        assigned = {}
        for seed in range(len(self.cells)):
            if seed in assigned:
                continue
            assigned[seed] = 1          # a seed per connected component
            frontier = [seed]
            while frontier:
                current = frontier.pop()
                for facet, sign in self._induced(self.cells[current]):
                    carriers = incidence[facet]
                    if len(carriers) != 2:
                        continue        # boundary (or a non-manifold facet)
                    (a, sign_a), (b, sign_b) = carriers
                    other, other_sign = (b, sign_b) if a == current else (a, sign_a)
                    # Coherent means the two induced orientations cancel on the
                    # shared facet: eps_current*sign + eps_other*other_sign = 0.
                    wanted = -assigned[current] * sign * other_sign
                    if other in assigned:
                        if assigned[other] != wanted:
                            self.orientable = False
                    else:
                        assigned[other] = wanted
                        frontier.append(other)
        self.signs = {self.cells[i]: s for i, s in assigned.items()}
        for facet, carriers in incidence.items():
            if len(carriers) == 1:
                index, sign = carriers[0]
                self.boundary_facets[facet] = assigned.get(index, 0) * sign

    @classmethod
    def fromSpacetime(cls, spacetime):
        return cls(GeometryState.cells(spacetime))

    def boundaryComponents(self):
        """The connected components of ∂W, as lists of boundary facets. Two
        facets are joined when they share a codimension-one face of the
        boundary (a (d-2)-face of the complex), which is the adjacency that
        makes each component a closed surface in its own right — the cobordism's
        separate ends."""
        facets = list(self.boundary_facets)
        ridges = {}
        for facet in facets:
            for i in range(len(facet)):
                ridge = facet[:i] + facet[i + 1:]
                ridges.setdefault(ridge, []).append(facet)
        seen, components = set(), []
        for facet in facets:
            if facet in seen:
                continue
            component, frontier = [], [facet]
            seen.add(facet)
            while frontier:
                current = frontier.pop()
                component.append(current)
                for i in range(len(current)):
                    for neighbour in ridges[current[:i] + current[i + 1:]]:
                        if neighbour not in seen:
                            seen.add(neighbour)
                            frontier.append(neighbour)
            components.append(component)
        return components
