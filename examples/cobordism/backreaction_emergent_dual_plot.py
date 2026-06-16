# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Render of the backreaction selection (companion to
``backreaction_emergent_dual.py``): the matter-energy landscape over the
carrier family, with the emergent dual W*(kappa) tracking from the conformal
runaway corner toward the matter's interior minimum as the coupling grows."""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402


def render(scan, traj, grid, outdir):
    g = np.asarray(grid)
    E = np.array([[scan[(round(float(sw), 3), round(float(sb), 3))]["E"]
                   for sw in g] for sb in g])      # rows = s_bulk, cols = s_wt
    ReS = np.array([[scan[(round(float(sw), 3), round(float(sb), 3))]["ReS"]
                     for sw in g] for sb in g])

    fig, (axE, axS) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    ext = (g[0], g[-1], g[0], g[-1])

    # -- matter energy + the emergent-dual trajectory ----------------------- #
    im = axE.imshow(E, origin="lower", extent=ext, aspect="auto",
                    cmap="viridis")
    fig.colorbar(im, ax=axE, label="matter energy E")
    iEmin = np.unravel_index(np.argmin(E), E.shape)
    axE.plot(g[iEmin[1]], g[iEmin[0]], "*", ms=18, mfc="#FFD700",
             mec="black", mew=1.2, label="matter minimum")
    ks = [k for k, _ in traj]
    xs = [w[0] for _, w in traj]
    ys = [w[1] for _, w in traj]
    axE.plot(xs, ys, "-o", color="#D55E00", lw=2, ms=6, mec="white",
             label=r"emergent dual $W^*(\kappa)$")
    for (kp, w) in (traj[0], traj[-1]):
        axE.annotate(rf"$\kappa={kp:.0f}$", (w[0], w[1]),
                     textcoords="offset points", xytext=(6, 6),
                     fontsize=8.5, color="#D55E00")
    axE.set_xlabel("worldtube timelike scale  $s_{wt}$")
    axE.set_ylabel("bulk timelike scale  $s_{bulk}$")
    axE.set_title("matter regulates & sources the emergent dual\n"
                  r"($\kappa=0$ at the conformal-runaway corner → pinned to the "
                  r"matter minimum)", fontsize=10.5)
    axE.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

    # -- the action (the runaway it regulates) ------------------------------ #
    im2 = axS.imshow(ReS, origin="lower", extent=ext, aspect="auto",
                     cmap="magma")
    fig.colorbar(im2, ax=axS, label=r"Re $S_{\mathrm{Regge}}$")
    axS.plot(xs, ys, "-o", color="#56B4E9", lw=2, ms=6, mec="white")
    axS.set_xlabel("worldtube timelike scale  $s_{wt}$")
    axS.set_ylabel("bulk timelike scale  $s_{bulk}$")
    axS.set_title("Re $S$ falls monotonically toward the corner —\nthe "
                  "conformal mode the matter must regulate", fontsize=10.5)

    fig.suptitle("Backreaction on the merge substrate — the stress-energy "
                 "sources the emergent dual", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "backreaction_emergent_dual.png")
    fig.savefig(path, dpi=130, facecolor="white")
    return path
