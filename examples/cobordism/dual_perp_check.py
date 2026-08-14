# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Check the "spatial and temporal curvature are perpendicular" claim on the
per-frame dumps written by `proton_animation.py --dump-dir`.

Reads each `frame_NNNN.json` and reports, per frame, three independent senses of
perpendicular between the two dual panels (Re ε = spatial, Im ε = temporal):

  * FIELD    — cosine between the two per-dual-node heat vectors (0 = the two
               fields are orthogonal as signals on the dual complex);
  * GRADIENT — angle between each field's direction of steepest increase, fit
               over the drawn dual-node positions (90 deg = the two heat maps
               run perpendicular, which is what the eye reads off tricontourf).
               Reported with the planar-fit R²: at low R² the field is lumpy
               rather than ramped and its "direction" means little;
  * AXES     — angle between the |heat|-weighted principal axes of the dual node
               cloud (90 deg = the two hot regions lie along perpendicular axes).

Frame numbers match the animation's window title, so `--frame 213` is the panel
titled "frame 213".

Usage (repo root):
    python examples/cobordism/dual_perp_check.py proton_dumps/run-x
    python examples/cobordism/dual_perp_check.py proton_dumps/run-x --frame 213
"""
import argparse
import glob
import json
import os

import numpy as np


def load(path):
    with open(path) as f:
        d = json.load(f)
    node = next((n for n in d["nodes"] if n.get("active")), d["nodes"][0])
    P = np.array([p for p in node["dual_positions"]], dtype=float)
    re = np.array(node["re_heat"], dtype=float)
    im = np.array(node["im_heat"], dtype=float)
    ok = np.all(np.isfinite(P), axis=1)
    return d, P[ok], re[ok], im[ok]


def planar_gradient(P, f):
    A = np.column_stack([np.ones(len(P)), P[:, 0], P[:, 1]])
    coef, *_ = np.linalg.lstsq(A, f, rcond=None)
    ss = float(np.sum((f - f.mean()) ** 2))
    r2 = float(1 - np.sum((f - A @ coef) ** 2) / ss) if ss > 0 else 0.0
    return coef[1:], r2


def principal_axis(P, w):
    if w.sum() <= 0:
        return None
    mu = (P * w[:, None]).sum(0) / w.sum()
    X = (P - mu) * np.sqrt(w)[:, None]
    return np.linalg.eigh(X.T @ X)[1][:, -1]


def angle(a, b, signed=False):
    if a is None or b is None:
        return float("nan")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    c = float(np.dot(a, b)) / (na * nb)
    if not signed:                       # undirected axes: fold to [0, 90]
        c = abs(c)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def analyze(path):
    d, P, re, im = load(path)
    if len(P) < 4:
        return None
    nre, nim = np.linalg.norm(re), np.linalg.norm(im)
    cos = float(re @ im / (nre * nim)) if nre > 0 and nim > 0 else float("nan")
    g_re, r2_re = planar_gradient(P, re)
    g_im, r2_im = planar_gradient(P, im)
    return {
        "frame": d["frame"], "n": len(P), "F": d.get("F"),
        "cos": cos,
        "grad": angle(g_re, g_im, signed=True), "r2": min(r2_re, r2_im),
        "axes": angle(principal_axis(P, np.abs(re)), principal_axis(P, np.abs(im))),
        "im_zero": bool(np.allclose(im, 0.0)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump_dir")
    ap.add_argument("--frame", type=int, help="report just this frame, in detail")
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dump_dir, "frame_*.json")))
    if not paths:
        raise SystemExit(f"no frame dumps in {args.dump_dir}")
    if args.frame is not None:
        one = os.path.join(args.dump_dir, f"frame_{args.frame:04d}.json")
        if not os.path.exists(one):
            raise SystemExit(f"no {one}")
        paths = [one]

    rows = [r for r in (analyze(p) for p in paths[::args.stride]) if r]
    if not rows:
        raise SystemExit("no frames with enough finite dual nodes to analyze")

    print(f"{'frame':>6} {'nodes':>5} {'cos(Re,Im)':>11} {'grad angle':>11} "
          f"{'R²':>5} {'axes angle':>11}")
    for r in rows:
        flag = "  <- Im channel is identically zero" if r["im_zero"] else ""
        print(f"{r['frame']:>6} {r['n']:>5} {r['cos']:>11.3f} "
              f"{r['grad']:>10.1f}° {r['r2']:>5.2f} {r['axes']:>10.1f}°{flag}")

    if len(rows) > 1:
        g = np.array([r["grad"] for r in rows])
        a = np.array([r["axes"] for r in rows])
        g = g[np.isfinite(g)]
        a = a[np.isfinite(a)]
        print(f"\nover {len(rows)} frames:")
        if g.size:
            near = float((np.abs(np.abs(g) - 90) <= 20).mean()) * 100
            print(f"  gradient angle: mean {g.mean():.1f}°  median "
                  f"{np.median(g):.1f}°  within 20° of perpendicular: {near:.0f}% "
                  f"(uniform null ≈ 22%)")
        if a.size:
            near = float((a >= 70).mean()) * 100
            print(f"  axes angle:     mean {a.mean():.1f}°  "
                  f"above 70°: {near:.0f}% (uniform null ≈ 22%)")
        r2 = np.array([r["r2"] for r in rows])
        if np.nanmean(r2) < 0.3:
            print(f"  NOTE: mean planar-fit R² = {np.nanmean(r2):.2f} — these "
                  f"fields are lumpy, not ramped, so the gradient angle is a "
                  f"weak descriptor; trust the axes and cosine columns more.")


if __name__ == "__main__":
    main()
