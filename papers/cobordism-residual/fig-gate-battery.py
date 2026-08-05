"""Regenerate fig-gate-battery.pdf from a spectral_gate_realizability run log.

Usage: python fig-gate-battery.py <run.log>

Parses the 52-gate battery table (gate name, family, residual, dim ker L1,
|Sigma| leak, verdict) and renders the log-residual scatter used in the paper.
Fonts are embedded as TrueType (fonttype 42) so the PDF carries no Type 3
fonts, per arXiv/journal requirements.
"""
import re
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
})
import matplotlib.pyplot as plt

ROW = re.compile(
    r"\s+(.+?)\s{2,}(\S[\w/() -]*?)\s+([\d.]+e[+-]\d+)\s+(\d+)\s+([\d.]+)\s+(YES|floor)\s*"
)

rows = []
with open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/sgr-300.log") as fh:
    for line in fh:
        m = ROW.fullmatch(line.rstrip("\n"))
        if m:
            name, family, r, dim, leak, verdict = m.groups()
            rows.append((name.strip(), float(r), verdict == "YES"))

assert len(rows) == 52, f"expected 52 gates, parsed {len(rows)}"

fig, ax = plt.subplots(figsize=(6.6, 3.0))
xs = range(len(rows))
for x, (name, r, realized) in zip(xs, rows):
    if realized:
        ax.scatter(x, r, marker="o", facecolors="none", edgecolors="#0072B2",
                   s=28, linewidths=1.1, zorder=3)
    else:
        ax.scatter(x, r, marker="x", color="#D55E00", s=24, linewidths=1.1,
                   zorder=3)
ax.axhline(1e-9, color="0.3", linestyle="--", linewidth=0.8)
ax.text(0.5, 2.5e-9, r"$\epsilon = 10^{-9}$", fontsize=7, color="0.3")
ax.set_yscale("log")
ax.set_ylim(1e-31, 1e2)
ax.set_xlim(-1, len(rows))
ax.set_xlabel("gate index (battery order)")
ax.set_ylabel(r"harmonic residual $r(U)$")
ax.scatter([], [], marker="o", facecolors="none", edgecolors="#0072B2",
           s=28, linewidths=1.1, label="realized (13)")
ax.scatter([], [], marker="x", color="#D55E00", s=24, linewidths=1.1,
           label="floored (39)")
ax.legend(loc="center right", frameon=False, fontsize=7)
ax.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig("fig-gate-battery.pdf")
print(f"wrote fig-gate-battery.pdf ({len(rows)} gates, "
      f"{sum(1 for _, _, k in rows if k)} realized)")
