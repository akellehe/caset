"""Held-fixed-geometry leak experiment — the kill test for tessera#602.

Protocol (the tessera pattern, made literal): the year-t register is the
**held-fixed initial geometry** — its edge set, filled triangles, metric,
and harmonic basis are frozen. The later structure (year t+1 flows, an
IPF null, a planted break, or the MRIO-implied national structure) is
then expressed ON that fixed geometry and tested against it. The **period
leak** is the R-norm of the harmonic (gauge-invariant) component of the
change: the part that no re-potentialing of vertices and no netting on
filled faces can absorb. Flow mass on vertex pairs absent from the fixed
edge set is reported separately as **off-complex mass** — demand for
topology change that the fixed geometry cannot carry at all.

Subcommands:
  history  — all consecutive year pairs vs IPF null, Frobenius, Leontief;
             writes out/leak_history.csv and out/decision.png
  plant    — Gate 2: rewire one sector's supply relationships in year t,
             verify the leak fires and attribution localizes to it
  mrio     — consistency control: MRIO-2018-implied national structure
             tested on the held-fixed national 2018 register (expected:
             no certificate — the MRIO layer regionalizes the same A)
  scan     — netting-threshold x metric-map stability scan
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

from econ_register import (
    CLOSURE,
    Register,
    build_money_flows,
    build_register,
    load_year,
    periods,
)

RECESSION_PAIRS = {(2000, 2001), (2007, 2008), (2008, 2009), (2019, 2020)}


# ----------------------------------------------------------------------
# Held-fixed protocol primitives
# ----------------------------------------------------------------------

def edge_vector(reg: Register, flows: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Express a flow matrix on the fixed geometry.

    Returns (f, off_mass): f[k] is the net flow on fixed edge k signed by
    the edge's stored orientation; off_mass is the total |net flow| on
    vertex pairs with no edge in the fixed complex.
    """
    A = flows.reindex(index=reg.vertices, columns=reg.vertices,
                      fill_value=0.0).values
    net = A - A.T
    f = np.array([net[i, j] for (i, j) in reg.edges])
    on = np.zeros_like(net, dtype=bool)
    for (i, j) in reg.edges:
        on[i, j] = on[j, i] = True
    off_mass = float(np.abs(np.triu(np.where(on, 0.0, net))).sum())
    return f, off_mass


def held_fixed_leak(reg: Register, flows_next: pd.DataFrame) -> dict:
    """Test a later structure against the frozen year-t geometry."""
    f_next, off_mass = edge_vector(reg, flows_next)
    delta = f_next - reg.net
    p = periods(reg, delta)
    rnorm = lambda v: float(np.sqrt(np.sum(reg.metric * v * v)))
    base = rnorm(reg.net)
    return {
        "leak": float(np.linalg.norm(p)) / max(base, 1e-300),
        "delta_norm": rnorm(delta) / max(base, 1e-300),
        "off_mass": off_mass / max(float(np.abs(reg.net).sum()), 1e-300),
        "periods": p,
    }


def attribute(reg: Register, p: np.ndarray, top: int = 10) -> pd.Series:
    """Per-sector attribution of a period-leak vector.

    Reconstructs the harmonic component H @ p on edges and credits each
    edge's magnitude to its endpoints. Returns the top sectors by share.
    """
    h_edge = reg.harmonics @ p
    score = np.zeros(len(reg.vertices))
    for k, (i, j) in enumerate(reg.edges):
        score[i] += abs(h_edge[k]) / 2.0
        score[j] += abs(h_edge[k]) / 2.0
    s = pd.Series(score, index=reg.vertices).sort_values(ascending=False)
    return (s / s.sum()).head(top)


# ----------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------

def ipf_null(prior: pd.DataFrame, target: pd.DataFrame,
             iters: int = 500, tol: float = 1e-12) -> pd.DataFrame:
    """2-D RAS/Sinkhorn: prior structure, later-year margins.

    Under the gauge reading this is the minimum-energy relaxation with
    pinned Gauss-law data: the network expected if sector sizes changed
    but relationships did not. Zeros of the prior are structural.
    """
    P = prior.clip(lower=0.0).values.copy()
    row_t = target.clip(lower=0.0).sum(axis=1).values  # payments out
    col_t = target.clip(lower=0.0).sum(axis=0).values  # receipts in
    for _ in range(iters):
        rs = P.sum(axis=1)
        P *= np.where(rs > 0, row_t / np.maximum(rs, 1e-300), 0.0)[:, None]
        cs = P.sum(axis=0)
        P *= np.where(cs > 0, col_t / np.maximum(cs, 1e-300), 0.0)[None, :]
        if (np.abs(P.sum(axis=1) - row_t).sum()
                / max(row_t.sum(), 1.0) < tol):
            break
    return pd.DataFrame(P, index=prior.index, columns=prior.columns)


def _industry_block(flows: pd.DataFrame) -> pd.DataFrame:
    inds = [v for v in flows.index if v not in CLOSURE]
    return flows.loc[inds, inds]


def leontief_distance(flows_a: pd.DataFrame, flows_b: pd.DataFrame) -> float:
    """Relative Frobenius distance between Leontief inverses.

    A[s, j] = intermediate purchases of j from s per dollar of j's total
    outlay; flows[j, s] is j paying s, so A is the transpose of the
    industry block scaled by column outlays.
    """
    def leontief(flows: pd.DataFrame) -> np.ndarray:
        blk = _industry_block(flows)
        outlay = flows.loc[blk.index].sum(axis=1)  # total payments out
        A = (blk.T / outlay.replace(0.0, np.nan)).fillna(0.0).values
        return np.linalg.inv(np.eye(len(A)) - A)
    La, Lb = leontief(flows_a), leontief(flows_b)
    return float(np.linalg.norm(Lb - La) / np.linalg.norm(La))


def frobenius_distance(flows_a: pd.DataFrame, flows_b: pd.DataFrame) -> float:
    return float(np.linalg.norm(flows_b.values - flows_a.values)
                 / np.linalg.norm(flows_a.values))


# ----------------------------------------------------------------------
# Subcommands
# ----------------------------------------------------------------------

def years_available(data: str) -> list[int]:
    use = pd.read_csv(pathlib.Path(data) / "use_summary.csv", usecols=["year"])
    return sorted(use["year"].unique())


def cmd_history(args: argparse.Namespace) -> None:
    years = years_available(args.data)
    rows = []
    flows_cache: dict[int, pd.DataFrame] = {}

    def flows_for(y: int) -> pd.DataFrame:
        if y not in flows_cache:
            flows_cache[y], _ = build_money_flows(load_year(args.data, y))
        return flows_cache[y]

    for t0, t1 in zip(years[:-1], years[1:]):
        f0, f1 = flows_for(t0), flows_for(t1)
        reg = build_register(f0, args.tau, args.metric)  # frozen geometry
        obs = held_fixed_leak(reg, f1)
        null = held_fixed_leak(reg, ipf_null(f0, f1))
        rows.append({
            "pair": f"{t0}-{t1}",
            "recession": (t0, t1) in RECESSION_PAIRS,
            "b1": reg.b1,
            "leak_observed": obs["leak"],
            "leak_ipf_null": null["leak"],
            "leak_excess": obs["leak"] - null["leak"],
            "delta_norm": obs["delta_norm"],
            "off_mass": obs["off_mass"],
            "frobenius": frobenius_distance(f0, f1),
            "leontief": leontief_distance(f0, f1),
        })
        top = attribute(reg, obs["periods"], top=5)
        print(f"{t0}->{t1}  leak {obs['leak']:.4f}  null {null['leak']:.4f}  "
              f"excess {obs['leak']-null['leak']:+.4f}  "
              f"frob {rows[-1]['frobenius']:.4f}  "
              f"top: {', '.join(f'{k} {v:.0%}' for k, v in top.items())}")

    df = pd.DataFrame(rows)
    out = pathlib.Path(args.out); out.mkdir(exist_ok=True)
    df.to_csv(out / "leak_history.csv", index=False)
    _decision_plot(df, out / "decision.png")
    print(f"\nwrote {out/'leak_history.csv'} and {out/'decision.png'}")


def _decision_plot(df: pd.DataFrame, path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    x = np.arange(len(df))
    for ax, cols, title in (
        (axes[0], [("leak_observed", "period leak (held-fixed geometry)"),
                   ("leak_ipf_null", "IPF null leak")],
         "Topological obstruction vs size-recomposition null"),
        (axes[1], [("frobenius", "Frobenius distance"),
                   ("leontief", "Leontief-inverse distance")],
         "Magnitude baselines"),
    ):
        for col, label in cols:
            ax.plot(x, df[col], marker="o", label=label)
        for i, rec in enumerate(df["recession"]):
            if rec:
                ax.axvspan(i - 0.4, i + 0.4, alpha=0.15, color="red")
        ax.set_title(title)
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["pair"], rotation=60, fontsize=7)
    fig.suptitle("Kill experiment: shaded = recession year-pairs")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def _rewire(flows: pd.DataFrame, sector: str, delta: float) -> pd.DataFrame:
    """Re-source a fraction of every buyer's purchases away from `sector`
    into their existing supplier mix, then rebalance to the ORIGINAL
    margins. This is a re-weighting the fixed geometry CAN carry, so the
    certificate must stay silent on it."""
    planted = flows.copy()
    inds = [v for v in planted.index if v not in CLOSURE and v != sector]
    for j in list(planted.index):
        if j == sector:
            continue
        moved = delta * planted.loc[j, sector]
        if moved <= 0:
            continue
        mix = planted.loc[j, inds].clip(lower=0.0)
        if mix.sum() > 0:
            planted.loc[j, inds] += (moved * mix / mix.sum()).values
            planted.loc[j, sector] -= moved
    return ipf_null(planted, flows)


def _inject(flows: pd.DataFrame, reg: Register, eps_musd: float) -> tuple[
        pd.DataFrame, int]:
    """Intensify the register mode carrying the largest observed period by
    eps (millions USD). A harmonic cochain is divergence-free (margins are
    untouched — no null can explain it) and orthogonal to every netting
    move: THE canonical irreducible structural change. The certificate
    must fire and measure it."""
    p_obs = periods(reg, reg.net)
    m = int(np.argmax(np.abs(p_obs)))
    h = reg.harmonics[:, m]
    planted = flows.copy()
    for k, (i, j) in enumerate(reg.edges):
        vi, vj = reg.vertices[i], reg.vertices[j]
        planted.loc[vi, vj] += eps_musd * h[k]  # signed: reverse flow ok
    return planted, m


def cmd_plant(args: argparse.Namespace) -> None:
    """Gate 2 — the control pair on the frozen year-t geometry.

    Negative control: a margin-preserving re-sourcing rewire is carryable
    by re-weighting; leak must be ~0. Positive control: an injected
    irreducible circulation of size eps must produce leak = eps/base to
    numerical precision, survive the IPF null untouched, and localize to
    the injected mode's sectors.
    """
    t = load_year(args.data, args.year)
    flows, _ = build_money_flows(t)
    reg = build_register(flows, args.tau, args.metric)  # frozen geometry
    rnorm_base = float(np.sqrt(np.sum(reg.metric * reg.net * reg.net)))

    neg = held_fixed_leak(reg, _rewire(flows, args.sector, args.delta))

    eps = args.eps * rnorm_base  # eps as a fraction of the base R-norm
    injected, mode = _inject(flows, reg, eps)
    pos = held_fixed_leak(reg, injected)
    pos_null = held_fixed_leak(reg, ipf_null(flows, injected))
    top = attribute(reg, pos["periods"], top=8)
    mode_sectors = attribute(
        reg, np.eye(reg.b1)[mode] if reg.b1 else np.zeros(0), top=8)
    overlap = len(set(top.index[:5]) & set(mode_sectors.index[:5]))

    print(f"Gate 2 on frozen {args.year} geometry "
          f"(b1={reg.b1}, metric={args.metric}, tau={args.tau:g}):")
    print(f"  negative control (rewire {args.sector}, delta={args.delta}): "
          f"leak {neg['leak']:.3e}  delta_norm {neg['delta_norm']:.3e}")
    print(f"  positive control (mode {mode}, eps={args.eps:g} of base): "
          f"leak {pos['leak']:.3e}  expected {args.eps:g}  "
          f"ipf-null leak {pos_null['leak']:.3e}  "
          f"excess {pos['leak'] - pos_null['leak']:.3e}")
    print(f"  attribution:  {', '.join(f'{k} {v:.0%}' for k, v in top.items())}")
    print(f"  mode sectors: "
          f"{', '.join(f'{k} {v:.0%}' for k, v in mode_sectors.items())}")
    print(f"  top-5 overlap: {overlap}/5")
    ok = (
        neg["leak"] < args.eps / 10.0
        and abs(pos["leak"] - args.eps) < args.eps / 10.0
        # size recomposition explains none of the injection: excess ~ eps
        and pos["leak"] - pos_null["leak"] > args.eps * 0.8
        and overlap >= 3
    )
    print(f"  GATE 2 {'PASS' if ok else 'FAIL'}")


def cmd_mrio(args: argparse.Namespace) -> None:
    """Consistency control: MRIO-2018-implied structure on the frozen
    national 2018 register. The MRIO layer regionalizes the same national
    accounts, so the certificate must NOT fire; a large leak here would
    mean the machinery manufactures false structure."""
    base = pathlib.Path(args.data)
    agg = pd.read_csv(base / "mrio_a_2018_industry.csv")
    nreg = pd.read_csv(base / "mrio_regions_2018.csv") \
             .set_index("consumer_industry_code")["n_regions"]
    conc = pd.read_csv(base / "bea_detail_summary_concordance.csv")
    d2s = dict(zip(conc["bea_detail_code"], conc["bea_summary_code"]))

    agg["a_national"] = agg["total_coefficient"] / \
        agg["consumer_industry_code"].map(nreg).values
    agg["s_sum"] = agg["supplier_industry_code"].map(d2s)
    agg["c_sum"] = agg["consumer_industry_code"].map(d2s)
    unmapped = agg[agg["s_sum"].isna() | agg["c_sum"].isna()]
    if len(unmapped):
        print(f"  note: {len(unmapped)} detail cells unmapped "
              f"({unmapped['total_coefficient'].abs().sum():.3g} coeff mass) "
              f"- dropped")
        agg = agg.dropna(subset=["s_sum", "c_sum"])
    # summary A: sum over supplier details, unweighted mean over consumer
    # details within the summary group (no detail output weights at spike
    # level - documented approximation)
    per_c = agg.groupby(["s_sum", "c_sum", "consumer_industry_code"])[
        "a_national"].sum().reset_index()
    A_mrio = per_c.groupby(["s_sum", "c_sum"])["a_national"].mean().unstack(
        fill_value=0.0)

    t = load_year(args.data, 2018)
    flows, _ = build_money_flows(t)
    reg = build_register(flows, args.tau, args.metric)  # frozen geometry

    blk = _industry_block(flows)
    outlay = flows.loc[blk.index].sum(axis=1)
    A_nat = (blk.T / outlay.replace(0.0, np.nan)).fillna(0.0)  # supplier x consumer

    common_s = [c for c in A_nat.index if c in A_mrio.index]
    common_c = [c for c in A_nat.columns if c in A_mrio.columns]
    Am = A_mrio.reindex(index=common_s, columns=common_c, fill_value=0.0)
    An = A_nat.loc[common_s, common_c]
    rel = float(np.linalg.norm(Am.values - An.values) /
                np.linalg.norm(An.values))
    print(f"MRIO consistency control (2018, county grain -> national):")
    print(f"  industries compared: {len(common_s)} x {len(common_c)}")
    print(f"  ||A_mrio - A_national||_F / ||A_national||_F = {rel:.4f}")

    # held-fixed leak of the MRIO-implied flow structure: replace the
    # industry block with A_mrio-scaled outlays, keep closure edges
    mrio_flows = flows.copy()
    scaled = (Am * outlay.reindex(common_c).values).T  # consumer pays supplier
    mrio_flows.loc[common_c, common_s] = scaled.values
    obs = held_fixed_leak(reg, mrio_flows)
    top = attribute(reg, obs["periods"], top=5)
    print(f"  held-fixed leak of MRIO-implied structure: {obs['leak']:.4f} "
          f"(off-complex mass {obs['off_mass']:.4f})")
    print(f"  attribution: {', '.join(f'{k} {v:.0%}' for k, v in top.items())}")


def cmd_scan(args: argparse.Namespace) -> None:
    t = load_year(args.data, args.year)
    flows, _ = build_money_flows(t)
    net = np.abs(flows.values - flows.values.T)[np.triu_indices(len(flows), 1)]
    taus = [0.0] + [float(np.quantile(net[net > 0], q))
                    for q in (0.5, 0.75, 0.9, 0.95)]
    print(f"threshold x metric scan, {args.year}:")
    for metric in ("conductance", "length", "unit"):
        for tau in taus:
            reg = build_register(flows, tau, metric)
            print(f"  metric={metric:12s} tau={tau:12.1f}  "
                  f"edges {len(reg.edges):5d}  tris {len(reg.triangles):6d}  "
                  f"b1 = {reg.b1}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--tau", default=0.0, type=float)
    ap.add_argument("--metric", default="conductance",
                    choices=("conductance", "length", "unit"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("history")
    p = sub.add_parser("plant")
    p.add_argument("--year", default=2005, type=int)
    p.add_argument("--sector", default="521CI",
                   help="sector to rewire away from (negative control)")
    p.add_argument("--delta", default=0.5, type=float,
                   help="fraction of purchases re-sourced")
    p.add_argument("--eps", default=0.05, type=float,
                   help="injected circulation, fraction of base R-norm")
    sub.add_parser("mrio")
    s = sub.add_parser("scan")
    s.add_argument("--year", default=2017, type=int)
    args = ap.parse_args()
    {"history": cmd_history, "plant": cmd_plant,
     "mrio": cmd_mrio, "scan": cmd_scan}[args.cmd](args)


if __name__ == "__main__":
    main()
