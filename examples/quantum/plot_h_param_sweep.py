"""Plot the issue #11 H-parameter sweep as a 2×3 grid of 3D surfaces.

Each subplot shows peak D_S as a function of two H_pair parameters
(the others pinned at the v0.2 finite-size reference center). The
surface is the seed-averaged peak D_S over the 7×7 grid; surface
color follows the same value (cool when D_S < 4, hot when D_S > 4,
near-white at D_S = 4 which is the H_DS4 target).

Input: /tmp/interaction-history/h_param_sweep.json (written by
/tmp/h_param_sweep_driver.py).

Output: docs/source/quantum-experiments/figures/v02_h_param_sweep.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers projection
import numpy as np

SCAN_JSON = Path("/tmp/interaction-history/h_param_sweep.json")
OUT_PNG = Path(__file__).resolve().parents[2] / (
    "docs/source/quantum-experiments/figures/v02_h_param_sweep.png"
)


PARAM_LABEL = {
    "jc":  r"$J_c$ (charge–charge)",
    "js":  r"$J_s$ (spin–spin)",
    "dm":  r"$\delta_m$ (mass shift)",
    "gcp": r"$\gamma_{CP}$ (CP-violation)",
}


def load_records():
    with open(SCAN_JSON) as f:
        meta = json.load(f)
    return meta


def gather_surface(records, pair_x, pair_y, grids):
    """Build (X, Y, Z, Z_std) arrays for the (pair_x, pair_y) surface.

    Z is the mean peak D_S over seeds at each (x, y) cell. Z_std is
    the per-cell std (used only for diagnostic printing; the surface
    plot uses the mean).
    """
    xs = grids[pair_x]
    ys = grids[pair_y]
    Z = np.full((len(xs), len(ys)), np.nan)
    Z_std = np.full((len(xs), len(ys)), np.nan)
    Z_cells = [[[] for _ in ys] for _ in xs]
    for r in records:
        if r.get("pair") != [pair_x, pair_y]:
            continue
        xv, yv = r["grid_xy"]
        i = min(range(len(xs)), key=lambda k: abs(xs[k] - xv))
        j = min(range(len(ys)), key=lambda k: abs(ys[k] - yv))
        ds = r["peak_dS"]
        if np.isfinite(ds):
            Z_cells[i][j].append(ds)
    for i in range(len(xs)):
        for j in range(len(ys)):
            if Z_cells[i][j]:
                Z[i, j] = float(np.mean(Z_cells[i][j]))
                Z_std[i, j] = float(np.std(Z_cells[i][j]))
    return np.array(xs), np.array(ys), Z, Z_std


def main():
    meta = load_records()
    records = meta["records"]
    grids = meta["grids"]
    pairs = [tuple(p) for p in meta["pairs"]]
    center = meta["center"]
    beta = meta["beta"]
    T = meta["T"]
    N = meta["N"]
    n_seeds = meta["n_seeds"]

    print(f"loaded {len(records)} records covering {len(pairs)} pairs "
          f"at β={beta}, T={T}, N={N}, n_seeds={n_seeds}")

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        rf"Charged Cartan v0.2  •  $H$-parameter sweeps "
        rf"(N={N}, T={T}, $\beta$={beta:.0e}, {n_seeds} seeds/cell)" + "\n"
        rf"Center: $J_c$={center['jc']}, $J_s$={center['js']}, "
        rf"$\delta_m$={center['dm']}, $\gamma_{{CP}}$={center['gcp']}  "
        rf"•  $z$ = mean peak $D_S$;  color centered on $D_S=4$",
        fontsize=12, y=0.995,
    )

    # Find a global color range centered on 4
    all_Z = []
    surfaces = {}
    for (px, py) in pairs:
        xs, ys, Z, Zstd = gather_surface(records, px, py, grids)
        surfaces[(px, py)] = (xs, ys, Z, Zstd)
        all_Z.append(Z[np.isfinite(Z)])
    all_Z = np.concatenate(all_Z) if all_Z else np.array([4.0])
    # Symmetric color window around 4, clipped to the actual data range
    z_lo = max(np.nanmin(all_Z) - 0.1, 4.0 - max(4.0 - np.nanmin(all_Z), 0.1))
    z_hi = min(np.nanmax(all_Z) + 0.1, 4.0 + max(np.nanmax(all_Z) - 4.0, 0.1))
    # Force symmetric half-width if practical
    half = max(abs(z_lo - 4.0), abs(z_hi - 4.0))
    z_lo, z_hi = 4.0 - half, 4.0 + half

    cmap = plt.get_cmap("RdBu_r")  # blue < 4 < red
    norm = plt.Normalize(vmin=z_lo, vmax=z_hi)

    for idx, (px, py) in enumerate(pairs):
        ax = fig.add_subplot(2, 3, idx + 1, projection="3d")
        xs, ys, Z, Zstd = surfaces[(px, py)]
        # meshgrid: X varies along axis-0, Y along axis-1
        Xg, Yg = np.meshgrid(xs, ys, indexing="ij")

        # Plot the mean-D surface with the shared norm
        surf = ax.plot_surface(
            Xg, Yg, Z,
            cmap=cmap, norm=norm,
            edgecolor="black", linewidth=0.2, antialiased=True,
            rstride=1, cstride=1, alpha=0.95,
        )

        # Overlay the D_S = 4 contour at z=4 as a thick black ring (a
        # "what H pulls us to 4" visual cue).
        try:
            ax.contour(Xg, Yg, Z, levels=[4.0], colors="k",
                       linewidths=2.0, zorder=5, offset=z_lo)
        except Exception:
            pass

        # Plane at D_S=4 (translucent reference)
        Xf, Yf = np.meshgrid(
            np.linspace(xs.min(), xs.max(), 2),
            np.linspace(ys.min(), ys.max(), 2),
            indexing="ij",
        )
        Zf = np.full_like(Xf, 4.0)
        ax.plot_surface(Xf, Yf, Zf, color="gray", alpha=0.18,
                        linewidth=0, zorder=1)

        ax.set_xlabel(PARAM_LABEL[px], fontsize=10)
        ax.set_ylabel(PARAM_LABEL[py], fontsize=10)
        ax.set_zlabel(r"peak $D_S$", fontsize=10)
        ax.set_title(f"({px}, {py})", fontsize=11)
        # Lock z-range so all subplots share the same vertical scale
        ax.set_zlim(z_lo, z_hi)
        ax.view_init(elev=22, azim=-58)
        ax.tick_params(labelsize=8)

    # One shared colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.18, 0.012, 0.64])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label(r"mean peak $D_S$", fontsize=10)
    cb.ax.axhline(4.0, color="black", linewidth=1.0)

    fig.subplots_adjust(left=0.04, right=0.90, top=0.92, bottom=0.05,
                        wspace=0.20, hspace=0.20)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    print(f"wrote {OUT_PNG}")

    # Diagnostic: print the (pair, cell) closest to D_S = 4 globally
    best_cell = None
    best_dist = float("inf")
    for (px, py), (xs, ys, Z, Zstd) in surfaces.items():
        for i in range(len(xs)):
            for j in range(len(ys)):
                if not np.isfinite(Z[i, j]):
                    continue
                d = abs(Z[i, j] - 4.0)
                if d < best_dist:
                    best_dist = d
                    best_cell = (px, py, xs[i], ys[j], Z[i, j], Zstd[i, j])
    if best_cell is not None:
        px, py, xv, yv, dz, std = best_cell
        print(f"  closest to D_S=4: pair ({px}, {py}) at ({px}={xv:.3f}, "
              f"{py}={yv:.3f}) → D_S = {dz:.4f} ± {std:.4f} "
              f"(|D-4| = {abs(dz-4.0):.4f})")


if __name__ == "__main__":
    main()
