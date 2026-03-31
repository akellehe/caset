#!/usr/bin/env python3
"""Solve for the spacetime geometry around a point mass using the Regge solver,
then render the result as an animated GIF.

The point mass is placed at a central vertex. The solver adjusts edge lengths
so that the Regge equations (∂S/∂ℓ² = 0) are satisfied, where
S = S_grav + S_matter with S_matter = -M Σ √(-ℓ²) (proper-time action).

The resulting triangulation is exported as a rotating GIF showing how curvature
concentrates near the mass.

Usage:
    python examples/regge_point_mass.py
    python examples/regge_point_mass.py --mass 5.0 --n-simplices 200 --save schwarzschild.gif
"""
import argparse

import caset
from caset.utils.memory_monitor import MemoryMonitor
from caset.utils.progress import SingleTaskProgress


def main():
    monitor = MemoryMonitor()
    p = argparse.ArgumentParser(
        description="Regge solver: point mass → spacetime geometry → GIF")

    p.add_argument("--n-simplices", type=int, default=50,
                   help="Initial number of simplices (default: 50)")
    p.add_argument("--mass", type=float, default=1.0,
                   help="Point mass in geometrized units G=c=1 (default: 1.0)")
    p.add_argument("--learning-rate", type=float, default=0.01,
                   help="Gradient descent learning rate (default: 0.01)")
    p.add_argument("--max-iters", type=int, default=100,
                   help="Maximum solver iterations (default: 100)")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="Convergence tolerance (default: 1e-6)")
    p.add_argument("--save", type=str, default="point_mass.gif",
                   help="Output file (.gif or .png, default: point_mass.gif)")
    p.add_argument("--tilt", type=float, default=25.0,
                   help="GIF precession tilt in degrees (default: 25)")
    p.add_argument("--spin", type=int, default=1,
                   help="GIF Y-axis rotations per loop (default: 1)")
    p.add_argument("--precession", type=int, default=1,
                   help="GIF precession cycles per loop (default: 1)")
    p.add_argument("--no-curvature-gif", action="store_true",
                   help="Skip the per-slice curvature heat map GIF")

    args = p.parse_args()

    prog = SingleTaskProgress(memory_monitor=monitor)

    # Build the triangulation
    prog.phase("building", extra=f"{args.n_simplices} simplices")
    sig = caset.Signature(4, caset.Lorentzian)
    metric = caset.Metric(True, sig)
    st = caset.Spacetime(metric, caset.CDT, 1.0, 1.0, caset.PREFERRED,
                         caset.Toroid())
    st.build(args.n_simplices)
    print(f"  Vertices: {st.getVertexCount()}, "
          f"Top simplices: {st.getSimplexCount()}")

    # Thermalize to get a more physical starting configuration
    target = st.getN41()
    cdt = caset.CDTSimulation(st, 2.2, 0.5, 0.6, 1.0 / target, target)
    prog.phase("tuning", total=20)
    cdt.tune(progress=prog.on_tick)
    prog.phase("thermalizing", total=10)
    cdt.sweep(10, progress=prog.on_tick)

    # Pick the spatial center: vertex with the most spacelike neighbors.
    # Ties broken by BFS depth-5 sum-of-distances (avoids torus wrapping).
    from collections import defaultdict, deque

    verts = st.getVertexList().toVector()
    slices = defaultdict(list)
    for v in verts:
        slices[round(v.getTime())].append(v)

    # Pick the largest spatial slice (peak of volume profile)
    peak_time = max(slices, key=lambda t: len(slices[t]))
    slice_verts = slices[peak_time]

    # Build spacelike adjacency for this slice
    slice_ids = {v.getId() for v in slice_verts}
    adj = defaultdict(list)
    for v in slice_verts:
        for e in v.getEdges():
            other = e.getTarget() if e.getSource().getId() == v.getId() \
                else e.getSource()
            if other.getId() in slice_ids and e.getSquaredLength() > 0:
                adj[v.getId()].append(other.getId())

    # Primary: most spacelike neighbors
    max_deg = max(len(adj[v.getId()]) for v in slice_verts)
    candidates = [v for v in slice_verts if len(adj[v.getId()]) == max_deg]

    if len(candidates) == 1:
        center = candidates[0]
    else:
        # Tiebreaker: BFS depth <= 5, pick smallest total distance
        max_depth = 5
        best_v, best_total = candidates[0], float("inf")
        for v in candidates:
            dist = {v.getId(): 0}
            q = deque([v.getId()])
            total = 0
            while q:
                uid = q.popleft()
                if dist[uid] >= max_depth:
                    continue
                for nid in adj[uid]:
                    if nid not in dist:
                        dist[nid] = dist[uid] + 1
                        total += dist[nid]
                        q.append(nid)
            if total < best_total:
                best_total = total
                best_v = v
        center = best_v

    print(f"  Center vertex: id={center.getId()}, "
          f"slice t={peak_time}, degree={len(adj[center.getId()])}")

    # Configure matter: static point mass along worldline through all slices
    matter = caset.MatterConfiguration()
    worldline = caset.MatterConfiguration.buildWorldline(center, st)
    print(f"  Worldline: {len(worldline)} vertices across "
          f"{len(set(v.getTime() for v in worldline))} time slices")
    matter.setWorldlineMass(center, args.mass, st)

    # Create solver
    solver = caset.ReggeSolver(st, matter)
    S_grav = solver.reggeAction()
    S_matt = solver.matterAction()
    F0 = solver.actionGradientNorm()
    print(f"  S_grav = {S_grav:.4f},  S_matter = {S_matt:.4f}")
    print(f"  ||∇S||² = {F0:.6f}")

    # Solve with progress display
    prog.phase("solving", total=args.max_iters,
               extra=f"‖∇S‖²={F0:.6f}")

    def on_progress(iteration, F):
        prog.on_tick()
        prog._lock.acquire()
        prog._extra = f"‖∇S‖²={F:.6f}"
        prog._lock.release()

    converged, F_final, iters = solver.solve(
        tol=args.tol,
        max_iters=args.max_iters,
        learning_rate=args.learning_rate,
        progress=on_progress,
    )

    status = "Converged" if converged else "Did not converge"
    print(f"\n{status} after {iters} iterations")
    print(f"  ||∇S||²: {F0:.6f} → {F_final:.6f}")
    print(f"  S_grav:   {S_grav:.4f} → {solver.reggeAction():.4f}")
    print(f"  S_matter: {S_matt:.4f} → {solver.matterAction():.4f}")

    # Render simplicial embedding GIF
    prog.phase("rendering", extra=args.save)
    st.save(args.save, tilt=args.tilt, spin=args.spin,
            precession=args.precession)

    # Render per-slice curvature heat map GIF
    if not args.no_curvature_gif and args.save.endswith(".gif"):
        from curvature_slice_gif import render_curvature_gif
        curv_path = args.save.replace(".gif", "_curvature.gif")
        prog.phase("rendering", extra=f"curvature → {curv_path}")
        render_curvature_gif(st, solver, worldline, curv_path)

    prog.finish(f"saved {args.save}")


if __name__ == "__main__":
    main()
