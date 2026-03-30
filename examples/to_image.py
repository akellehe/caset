#!/usr/bin/env python3
import argparse
import caset
from caset.utils.memory_monitor import MemoryMonitor
from caset.utils.progress import SingleTaskProgress


def main():
    monitor = MemoryMonitor()
    p = argparse.ArgumentParser(description="Build, thermalize, and render a CDT spacetime.")

    # Spacetime
    p.add_argument("--n-simplices", type=int, default=2000,
                   help="Initial number of simplices (default: 2000)")

    # CDT couplings
    p.add_argument("--k0", type=float, default=2.2,
                   help="Bare inverse Newton's constant (default: 2.2)")
    p.add_argument("--k4", type=float, default=0.5,
                   help="Cosmological constant coupling (default: 0.5)")
    p.add_argument("--delta", type=float, default=0.6,
                   help="Asymmetry parameter (default: 0.6)")
    p.add_argument("--epsilon", type=float, default=None,
                   help="Volume-fixing strength (default: 1/target_N41)")
    p.add_argument("--targetN41", type=int, default=None,
                   help="Target (d,1)-volume (default: N41 after build)")
    p.add_argument("--quadraticVolume", action="store_true", default=True,
                   help="Use quadratic volume fixing (default)")
    p.add_argument("--no-quadraticVolume", action="store_false", dest="quadraticVolume",
                   help="Use linear volume fixing")

    # Thermalization
    p.add_argument("--n-sweeps", type=int, default=50,
                   help="Number of thermalization sweeps (default: 50)")

    # GIF rotation
    p.add_argument("--tilt", type=float, default=25.0,
                   help="Precession cone half-angle in degrees (default: 25)")
    p.add_argument("--spin", type=int, default=1,
                   help="Y-axis rotations per loop (default: 1)")
    p.add_argument("--precession", type=int, default=1,
                   help="Precession cycles per loop (default: 1)")

    # Output
    p.add_argument("--save", type=str, default="spacetime.gif",
                   help="Output filename, .gif or .png (default: spacetime.gif)")

    args = p.parse_args()

    prog = SingleTaskProgress(memory_monitor=monitor)
    prog.phase("building", extra=f"{args.n_simplices} simplices")

    sig = caset.Signature(4, caset.Lorentzian)
    metric = caset.Metric(True, sig)
    st = caset.Spacetime(metric, caset.CDT, 1.0, 1.0, caset.PREFERRED, caset.Toroid())
    st.build(args.n_simplices)

    target = args.targetN41 if args.targetN41 is not None else st.getN41()
    eps = args.epsilon if args.epsilon is not None else 1.0 / max(target, 1)
    cdt = caset.CDTSimulation(st, args.k0, args.k4, args.delta, eps,
                              target, args.quadraticVolume)

    prog.phase("tuning")
    cdt.tune()
    if args.n_sweeps > 0:
        prog.phase("thermalizing", total=args.n_sweeps)
        cdt.sweep(args.n_sweeps, progress=prog.on_tick)

    prog.phase("rendering", extra=args.save)
    st.save(args.save, tilt=args.tilt, spin=args.spin, precession=args.precession)
    prog.finish(f"saved {args.save}")


if __name__ == "__main__":
    main()