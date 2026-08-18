# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Read a finished complex off a state file and say what it is.

    python examples/cobordism/inspect_state.py state_0196.json [--degree 3]

Answers, in order: is it orientable and how do the top cells split; what is the
boundary ∂W, how many ends does it have and which is the past one; what is the
topology; what causal character do the edge intervals carry; and where does the
degree-k spectrum sit relative to its kernel.

Every number here is read from the recorded state — nothing is re-run, and the
build is not process-deterministic, so re-running could not reproduce it anyway.
"""
import argparse
import collections

import numpy as np

import tessera
from geometry_state import GeometryState, Orientation

cobordism = tessera.cobordism


class StateReport:
    """The report, one section per question."""

    def __init__(self, path, degree=3, weights="squared"):
        self.record = GeometryState.load(path)
        self.spacetime = GeometryState.rehydrate(self.record)
        self.orientation = Orientation(self.record["cells"])
        self.degree = degree
        cobordism.HodgeLaplacian.setDefaultWeightConvention(
            cobordism.HodgeWeightConvention.SquaredContent if weights == "squared"
            else cobordism.HodgeWeightConvention.Content)
        self.times = {int(v): float(t) for v, t in self.record["vertex_times"]}

    def orientationSection(self):
        signs = self.orientation.signs
        counts = collections.Counter(signs.values())
        print(f"orientable        : {self.orientation.orientable}")
        print(f"top cells         : {len(self.record['cells'])}  "
              f"(eps=+1: {counts.get(1, 0)}, eps=-1: {counts.get(-1, 0)})")
        if not self.orientation.orientable:
            print("  NOTE: propagation met a contradiction — no coherent "
                  "orientation exists; the signs above are the partial "
                  "assignment up to that point.")

    def boundarySection(self):
        components = self.orientation.boundaryComponents()
        facets = self.orientation.boundary_facets
        print(f"\nboundary dW       : {len(facets)} facets in "
              f"{len(components)} component(s)")
        for index, component in enumerate(sorted(
                components, key=lambda c: min(min(self.times.get(v, 0.0)
                                                  for v in f) for f in c))):
            vertices = {v for facet in component for v in facet}
            times = [self.times.get(v, 0.0) for v in vertices]
            induced = collections.Counter(facets[f] for f in component)
            print(f"  component {index}: {len(component)} facets, "
                  f"{len(vertices)} vertices, t in "
                  f"[{min(times):.4f}, {max(times):.4f}]  "
                  f"induced orientation +1/-1: "
                  f"{induced.get(1, 0)}/{induced.get(-1, 0)}")
        if len(components) == 2:
            print("  two ends: the cobordism's M0 and M1 (earliest listed first)")
        elif len(components) == 1:
            print("  a single end: not yet split into two boundary components")

    def topologySection(self):
        betti = cobordism.MultiCobordism.betti(self.spacetime)
        holes = cobordism.MultiCobordism.emergent_holes(self.spacetime, self.degree)
        print(f"\nbetti             : {list(betti)}")
        print(f"emergent holes k={self.degree}: {len(holes)}")

    def causalSection(self):
        values = np.array([complex(re, im) for _u, _v, re, im in self.record["edges"]])
        magnitude = np.abs(values)
        scale = magnitude.max() if magnitude.size else 1.0
        tolerance = 1e-9 * scale
        real, imaginary = values.real, values.imag
        spacelike = int(np.sum((real > tolerance) & (np.abs(imaginary) <= tolerance)))
        timelike = int(np.sum((real < -tolerance) & (np.abs(imaginary) <= tolerance)))
        null = int(np.sum(magnitude <= tolerance))
        complex_valued = int(np.sum(np.abs(imaginary) > tolerance))
        print(f"\nedges             : {values.size}  spacelike {spacelike}  "
              f"timelike {timelike}  null {null}  complex {complex_valued}")
        if values.size:
            print(f"  |l^2| in [{magnitude.min():.4e}, {magnitude.max():.4e}], "
                  f"median {np.median(magnitude):.4e}")
            phase = np.angle(values)
            near_null = np.abs(np.abs(phase) - np.pi / 2) < 0.1
            print(f"  arg(l^2) within 0.1 rad of +/-pi/2 (the null locus): "
                  f"{int(near_null.sum())}/{phase.size}")

    def spectrumSection(self):
        flat = np.array(cobordism.HodgeLaplacian(self.spacetime)
                        .laplacian(self.degree, True))
        size = int(round(np.sqrt(flat.size)))
        if size == 0:
            print(f"\nL_{self.degree}: no cells at this degree")
            return
        singular = np.linalg.svd(flat.reshape(size, size), compute_uv=False)
        print(f"\nL_{self.degree} spectrum      : {size} cells, "
              f"sigma_max {singular[0]:.4e}, sigma_min {singular[-1]:.4e}, "
              f"ratio {singular[-1]/singular[0]:.3e}")
        print(f"  6 smallest: {np.array2string(singular[-6:][::-1], precision=4)}")

    def run(self):
        for section in (self.orientationSection, self.boundarySection,
                        self.topologySection, self.causalSection,
                        self.spectrumSection):
            section()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("state", help="a state file written by --checkpoint")
    parser.add_argument("--degree", type=int, default=3,
                        help="register degree k to report topology and spectrum at")
    parser.add_argument("--hodge-weights", choices=("content", "squared"),
                        default="squared", help="weight convention for L_k")
    arguments = parser.parse_args()
    print(f"state             : {arguments.state}")
    StateReport(arguments.state, arguments.degree, arguments.hodge_weights).run()


if __name__ == "__main__":
    main()
