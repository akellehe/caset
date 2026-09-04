# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Shared fixtures for the chain-level Whitney Hodge pencil tests: the
specification's 3x3 CDT-like torus (T6/T7) and the scaling-verification
mesh generators (flat jittered torus, flat cylinder, conformally flat torus),
each with the continuum harmonic edge integrals they are compared against."""
import numpy as np

from tessera import cobordism as cob


def edges(K):
    return [tuple(int(v) for v in e) for e in K.kSimplexVertices(1)]


def lengths(K, table):
    return [complex(table[e]) for e in edges(K)]


def torus_cells(N):
    def vid(i, j):
        return (i % N) * N + (j % N)

    cells = []
    for i in range(N):
        for j in range(N):
            cells.append([vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)])
            cells.append([vid(i, j), vid(i, j + 1), vid(i + 1, j + 1)])
    return cells, vid


def torus33(h=1.0, v=-0.5, dgl=0.5):
    """The oracle's 3x3 CDT-like torus: 9 vertices, 27 edges, 18 triangles;
    horizontal edges h, vertical edges v, diagonals dgl."""
    n = 3
    cells, _ = torus_cells(n)
    K = cob.ChainComplex.fromTopCells(cells)

    def stype(e):
        a, b = e
        ia, ja = divmod(a, n)
        ib, jb = divmod(b, n)
        di, dj = (ib - ia) % n, (jb - ja) % n
        if di in (1, 2) and dj == 0:
            return "v"
        if di == 0:
            return "h"
        return "d"

    table = {e: {"h": h, "v": v, "d": dgl}[stype(e)] for e in edges(K)}
    return K, lengths(K, table)


def random_allowable(K, rng, scale=0.05):
    """Unit Euclidean lengths with a small complex perturbation (allowable)."""
    n = K.numSimplices(1)
    return [complex(1.0 + scale * rng.normal(), scale * rng.normal()) for _ in range(n)]


def flat_torus(N, jitter=0.0, lorentz=False, seed=0, Lt=1.0, Lx=1.0):
    """N x N triangulated torus with vertices jittered on the flat torus
    [0,Lt) x [0,Lx); squared lengths dx^2 + dt^2 (Euclidean) or dx^2 - dt^2
    (Lorentzian). Continuum harmonic 1-forms dt, dx: edge integrals are the
    displacement components, returned as the n_1 x 2 array W."""
    rng = np.random.default_rng(seed)
    cells, vid = torus_cells(N)
    K = cob.ChainComplex.fromTopCells(cells)
    coords = {vid(i, j): np.array([(i + jitter * rng.uniform(-1, 1)) * Lt / N,
                                   (j + jitter * rng.uniform(-1, 1)) * Lx / N])
              for i in range(N) for j in range(N)}
    s, W = [], []
    for (a, b) in edges(K):
        d = coords[b] - coords[a]
        d[0] -= Lt * np.round(d[0] / Lt)
        d[1] -= Lx * np.round(d[1] / Lx)
        s.append(complex((d[1] ** 2 - d[0] ** 2) if lorentz else (d[0] ** 2 + d[1] ** 2)))
        W.append(d.copy())
    return K, s, np.array(W, dtype=complex)


def conformal_torus(N, amp=0.3, jitter=0.15, lorentz=False, seed=0):
    """Metric e^{2 phi}(dt^2 + dx^2) or e^{2 phi}(-dt^2 + dx^2) with
    phi = amp sin(2 pi t) cos(2 pi x) by the midpoint rule; `amp` may be
    complex. Harmonic 1-forms are conformally invariant in two dimensions,
    so the continuum answer is span(dt, dx)."""
    rng = np.random.default_rng(seed)
    cells, vid = torus_cells(N)
    K = cob.ChainComplex.fromTopCells(cells)
    coords = {vid(i, j): np.array([(i + jitter * rng.uniform(-1, 1)) / N,
                                   (j + jitter * rng.uniform(-1, 1)) / N])
              for i in range(N) for j in range(N)}

    def phi(p):
        return amp * np.sin(2 * np.pi * p[0]) * np.cos(2 * np.pi * p[1])

    s, W = [], []
    for (a, b) in edges(K):
        d = coords[b] - coords[a]
        d -= np.round(d)
        q = (d[1] ** 2 - d[0] ** 2) if lorentz else (d[0] ** 2 + d[1] ** 2)
        s.append(complex(np.exp(2 * phi(coords[a] + 0.5 * d)) * q))
        W.append(d.copy())
    return K, s, np.array(W, dtype=complex)


def flat_cylinder(N, L, jitter=0.25, lorentz=False, seed=0):
    """S^1 x [0, L/N] cobordism with two boundary circles; jitter only on the
    interior layers. The absolute (Neumann) harmonic 1-form is d(theta),
    tangential to the boundary; W is its n_1 x 1 edge-integral column."""
    rng = np.random.default_rng(seed)

    def vid(i, j):
        return i * N + (j % N)

    cells = []
    for i in range(L):
        for j in range(N):
            cells.append([vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)])
            cells.append([vid(i, j), vid(i, j + 1), vid(i + 1, j + 1)])
    K = cob.ChainComplex.fromTopCells(cells)
    coords = {vid(i, j): np.array([(i + (jitter * rng.uniform(-1, 1) if 0 < i < L else 0.0)) / N,
                                   (j + jitter * rng.uniform(-1, 1)) / N])
              for i in range(L + 1) for j in range(N)}
    s, W = [], []
    for (a, b) in edges(K):
        d = coords[b] - coords[a]
        d[1] -= np.round(d[1])
        s.append(complex((d[1] ** 2 - d[0] ** 2) if lorentz else (d[0] ** 2 + d[1] ** 2)))
        W.append([d[1]])
    return K, s, np.array(W, dtype=complex)


def conformal_torus_rotated(N, amp=0.3, jitter=0.15, epsilon=0.0, seed=0):
    """The Lorentzian conformally flat torus with its timelike direction rotated
    by e^{-2 i epsilon} at the coordinate level: q = dx^2 - e^{-2 i eps} dt^2.
    This is the generator's own rotation (it knows the split of every edge into
    spatial and temporal parts); the library's `LorentzianFamily.rotate` acts on
    declared per-edge causal types instead."""
    rng = np.random.default_rng(seed)
    cells, vid = torus_cells(N)
    K = cob.ChainComplex.fromTopCells(cells)
    coords = {vid(i, j): np.array([(i + jitter * rng.uniform(-1, 1)) / N,
                                   (j + jitter * rng.uniform(-1, 1)) / N])
              for i in range(N) for j in range(N)}

    def phi(p):
        return amp * np.sin(2 * np.pi * p[0]) * np.cos(2 * np.pi * p[1])

    rot = np.exp(-2j * epsilon)
    s, W = [], []
    for (a, b) in edges(K):
        d = coords[b] - coords[a]
        d -= np.round(d)
        q = d[1] ** 2 - rot * d[0] ** 2
        s.append(complex(np.exp(2 * phi(coords[a] + 0.5 * d)) * q))
        W.append(d.copy())
    return K, s, np.array(W, dtype=complex)


def torus33_causal_types(K):
    """Declared causal types for the specification's 3x3 torus: vertical edges
    timelike, horizontal and diagonal edges spacelike (the CDT-like reading)."""
    from tessera import chainhodge as ch
    n = 3
    out = []
    for (a, b) in edges(K):
        ia, ja = divmod(a, n)
        ib, jb = divmod(b, n)
        di, dj = (ib - ia) % n, (jb - ja) % n
        out.append(ch.CausalType.Timelike if (di in (1, 2) and dj == 0) else ch.CausalType.Spacelike)
    return out
