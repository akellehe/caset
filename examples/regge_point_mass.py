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
from tqdm import tqdm


def main():
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

    args = p.parse_args()

    # Build the triangulation
    print(f"Building spacetime with {args.n_simplices} simplices...")
    sig = caset.Signature(4, caset.Lorentzian)
    metric = caset.Metric(True, sig)
    st = caset.Spacetime(metric, caset.CDT, 1.0, 1.0, caset.PREFERRED,
                         caset.Toroid())
    st.build(args.n_simplices)
    print(f"  Vertices: {st.getVertexCount()}, "
          f"Top simplices: {st.getSimplexCount()}")

    # Thermalize to get a more physical starting configuration
    target = st.getN41()
    cdt = caset.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, target)
    cdt.tune()
    print("Thermalizing...")
    cdt.sweep(10)

    # Pick the vertex with highest degree as the "center"
    verts = st.getVertexList().toVector()
    center = max(verts, key=lambda v: v.degree())
    print(f"  Center vertex: id={center.getId()}, degree={center.degree()}")

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

    # Solve with tqdm progress bar
    bar = tqdm(total=args.max_iters, desc="Solving", unit="iter",
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                          "[{elapsed}<{remaining}, {rate_fmt}] "
                          "F={postfix}")
    bar.set_postfix_str(f"{F0:.6f}")

    def on_progress(iteration, F):
        bar.update(1)
        bar.set_postfix_str(f"{F:.6f}")

    converged, F_final, iters = solver.solve(
        tol=args.tol,
        max_iters=args.max_iters,
        learning_rate=args.learning_rate,
        progress=on_progress,
    )
    bar.close()

    status = "Converged" if converged else "Did not converge"
    print(f"\n{status} after {iters} iterations")
    print(f"  ||∇S||²: {F0:.6f} → {F_final:.6f}")
    print(f"  S_grav:   {S_grav:.4f} → {solver.reggeAction():.4f}")
    print(f"  S_matter: {S_matt:.4f} → {solver.matterAction():.4f}")

    # Render
    print(f"Saving {args.save}...")
    st.save(args.save, tilt=args.tilt, spin=args.spin,
            precession=args.precession)
    print(f"Done. Output: {args.save}")


if __name__ == "__main__":
    main()
