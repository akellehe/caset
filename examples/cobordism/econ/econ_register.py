"""Economic register — construction + harmonic register for one BEA year.

Exploratory spike for tessera#602 (see README.md). Builds the money-flow
network from the BEA summary Supply (TableID 262) and Use (TableID 259)
tables, runs the Gate-1 accounting checks, assembles the oriented weighted
complex, and computes the harmonic register: b1, harmonic circuits, period
coordinates, Gram matrix, and divergence residuals.

Money-flow orientation convention: an edge points in the direction the
dollars move — buyer pays seller. Divergence at a vertex is inflow minus
outflow; the flow-of-funds Gauss law says it vanishes at every vertex once
the accounts are closed (industries by accounting identity; households,
government, and rest-of-world by explicit net-lending edges into the
capital account; the capital account itself up to the statistical
discrepancy, which is reported, not hidden).

Hodge conventions (all in the docstring of :func:`harmonic_register`):
conservation and netting are metric-free statements about raw dollar
flows; the metric (the value→metric knob) enters only through the energy
inner product that decides orthogonality and projection lengths.

Run (see README.md for the environment)::

    PYTHONPATH=<worktree> python econ_register.py --data data --year 2017
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import pathlib

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Code classification (BEA summary Supply-Use framework)
# ----------------------------------------------------------------------

# Use-table value-added rows -> closure counterparty. Values are used as
# stored: positive means the industry pays the counterparty, negative
# means the counterparty pays the industry (subsidies). Product taxes
# (T00TOP/T00SUB) are deliberately absent: the product-tax wedge reaches
# GOV once, per commodity, through the supply-table T015 rate — the
# empirical identity is VABAS = V001 + V003 + T00OTOP (+ T00OSUB, 2018+)
# and T018 = T005 + VABAS.
VA_ROWS = {
    "V001": "HH",      # compensation of employees
    "V003": "CAP",     # gross operating surplus
    "T00OTOP": "GOV",  # other taxes on production
    "T00OSUB": "GOV",  # less: other subsidies (2018+ only)
}
USE_TOTAL_ROWS = {"T005", "T018", "VABAS", "VAPRO", "T00TOP", "T00SUB"}

# Use-table final-demand columns -> closure buyer.
FINAL_DEMAND_BUYERS = {
    "F010": "HH",
    "F02E": "CAP", "F02N": "CAP", "F02R": "CAP", "F02S": "CAP", "F030": "CAP",
    "F040": "ROW",
    "F06C": "GOV", "F06E": "GOV", "F06N": "GOV", "F06S": "GOV",
    "F07C": "GOV", "F07E": "GOV", "F07N": "GOV", "F07S": "GOV",
    "F10C": "GOV", "F10E": "GOV", "F10N": "GOV", "F10S": "GOV",
}
USE_TOTAL_COLS = {"T001", "T019"}

# Supply-table composition columns. Exact identities verified on 2017:
#   T013 (basic supply)      = T007 (domestic) + MCIF + MADJ
#   T014 (margins)           = Trade + Trans
#   T015 (taxes on products) = MDTY + TOP + SUB   (SUB stored negative)
#   T016 (purchaser supply)  = T013 + T014 + T015 = T019 (total use)
SUPPLY_IMPORT_COLS = ("MCIF", "MADJ")  # imports (+ CIF/FOB adj) -> ROW sells
SUPPLY_MARGIN_COLS = ("Trade", "Trans")  # rebooked as margin-commodity use
SUPPLY_TAX_COLS = ("MDTY", "TOP", "SUB")  # product-tax wedge -> GOV
SUPPLY_TOTAL_COLS = {"T007", "T013", "T014", "T015", "T016"}
SUPPLY_TOTAL_ROWS = {"T017"}

# Margin rebooking needs no hardcoded commodity lists: each margin column
# (Trade, Trans) sums to zero across commodities — positive entries are the
# margins buyers pay on goods, and the offsetting negative entries sit on
# the margin-providing commodities' own rows (verified exactly on 2017:
# Trade +3,264,931 / -3,264,931 across {42, 441, 445, 452, 4A0}). The
# negative entries ARE the redistribution weights.

CLOSURE = ("HH", "CAP", "GOV", "ROW")


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def _wide(df: pd.DataFrame) -> pd.DataFrame:
    """Long -> wide (row_code x col_code), NaN -> 0."""
    return df.pivot_table(
        index="row_code", columns="col_code", values="value", aggfunc="sum"
    ).fillna(0.0)


@functools.lru_cache(maxsize=4)
def load_tables(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pathlib.Path(data_dir)
    use = pd.read_csv(base / "use_summary.csv")
    supply = pd.read_csv(base / "make_summary.csv")  # TableID 262 is Supply
    return use, supply


@dataclasses.dataclass
class YearTables:
    year: int
    use: pd.DataFrame     # commodities+VA rows x industries+F+totals
    supply: pd.DataFrame  # commodities+T017 x industries+composition+totals
    commodities: list[str]
    industries: list[str]


def load_year(data_dir: str, year: int) -> YearTables:
    use_l, sup_l = load_tables(data_dir)
    use = _wide(use_l[use_l.year == year])
    supply = _wide(sup_l[sup_l.year == year])
    commodities = sorted(
        set(supply.index) - SUPPLY_TOTAL_ROWS
    )
    industries = sorted(
        set(use.columns) - set(FINAL_DEMAND_BUYERS) - USE_TOTAL_COLS
    )
    return YearTables(year, use, supply, commodities, industries)


# ----------------------------------------------------------------------
# Gate 1 — accounting identities
# ----------------------------------------------------------------------

def gate1_report(t: YearTables) -> dict:
    """Test the candidate accounting identities numerically.

    Returns relative residuals (max over codes, |lhs-rhs| / scale). The
    point is to let the data declare its valuation basis rather than
    assuming one.
    """
    C, I = t.commodities, t.industries
    U = t.use.reindex(index=C, columns=I, fill_value=0.0)
    F = t.use.reindex(index=C, columns=list(FINAL_DEMAND_BUYERS), fill_value=0.0)
    S = t.supply

    scale = float(S.get("T016", S.get("T013")).reindex(C).abs().sum())

    def rel(a: pd.Series, b: pd.Series) -> float:
        return float((a - b).abs().max()) / max(scale / len(C), 1.0)

    def _cols(cols: tuple[str, ...]) -> pd.Series:
        return S.reindex(columns=list(cols), fill_value=0.0).reindex(C) \
                .fillna(0.0).sum(axis=1)

    total_use = U.sum(axis=1) + F.sum(axis=1)
    checks = {
        # commodity: use-side total against the published T019
        "use_rowsum_vs_T019": rel(total_use, t.use["T019"].reindex(C)),
        # commodity: purchaser-price supply == total use (the SUT identity)
        "T016_vs_T019": rel(S["T016"].reindex(C), t.use["T019"].reindex(C)),
        # supply composition: domestic + imports = basic supply
        "basic_supply": rel(
            S["T007"].reindex(C) + _cols(SUPPLY_IMPORT_COLS),
            S["T013"].reindex(C),
        ),
        "margins": rel(_cols(SUPPLY_MARGIN_COLS), S["T014"].reindex(C)),
        "product_taxes": rel(_cols(SUPPLY_TAX_COLS), S["T015"].reindex(C)),
        "purchaser_supply": rel(
            S["T013"].reindex(C) + S["T014"].reindex(C) + S["T015"].reindex(C),
            S["T016"].reindex(C),
        ),
    }
    # industry: intermediate + VA = output (T018 = T005 + VABAS)
    va = pd.Series(0.0, index=I)
    for code in VA_ROWS:
        if code in t.use.index:
            va = va + t.use.loc[code, I].fillna(0.0)
    checks["vabas_identity"] = rel(va, t.use.loc["VABAS", I])
    checks["industry_output"] = rel(U.sum(axis=0) + va, t.use.loc["T018", I])
    # industry: supply column total T017 vs domestic make colsum
    checks["industry_supply_total"] = rel(
        S.reindex(index=C, columns=I, fill_value=0.0).sum(axis=0), S.loc["T017", I]
    )
    return checks


# ----------------------------------------------------------------------
# Money-flow network
# ----------------------------------------------------------------------

def build_money_flows(t: YearTables) -> tuple[pd.DataFrame, dict]:
    """Assemble the vertex x vertex money-flow matrix (buyer pays seller).

    Every purchaser-price use cell is split into a basic-price part
    (attributed to suppliers via the supply-table market shares, with
    imports sold by ROW and duties collected by GOV), a margin part
    (rebooked as purchases of the trade/transport commodities), and a
    product-tax part (paid to GOV, net of subsidies). Value added flows
    from industries to HH/CAP/GOV; final demand flows from the closure
    buyers; net-lending edges close HH, GOV, and ROW into CAP.

    Returns (flows, diagnostics). flows[i, j] = dollars paid by i to j.
    """
    C, I = t.commodities, t.industries
    U = t.use.reindex(index=C, columns=I, fill_value=0.0)
    Fd = t.use.reindex(index=C, columns=list(FINAL_DEMAND_BUYERS), fill_value=0.0)
    S = t.supply

    vertices = list(I) + list(CLOSURE)
    flows = pd.DataFrame(0.0, index=vertices, columns=vertices)

    # --- per-commodity purchaser-price decomposition rates -------------
    def _cols(cols: tuple[str, ...]) -> pd.Series:
        return S.reindex(columns=list(cols), fill_value=0.0).reindex(C) \
                .fillna(0.0).sum(axis=1)

    t016 = S["T016"].reindex(C).replace(0.0, np.nan)
    margin = S.reindex(columns=list(SUPPLY_MARGIN_COLS), fill_value=0.0) \
              .reindex(C).fillna(0.0)
    margin_pos = margin.clip(lower=0.0)   # margins buyers pay on each good
    margin_neg = (-margin).clip(lower=0.0)  # margin commodities' provision
    margin_rate = (margin_pos.sum(axis=1) / t016).fillna(0.0)
    tax_rate = (_cols(SUPPLY_TAX_COLS) / t016).fillna(0.0)  # SUB negative
    basic_rate = 1.0 - margin_rate - tax_rate
    # per-column redistribution weights (each margin column sums to zero)
    margin_supply_w = margin_neg / margin_neg.sum(axis=0).replace(0.0, np.nan)
    margin_supply_w = margin_supply_w.fillna(0.0)

    # --- supplier shares of each commodity at basic prices -------------
    M = S.reindex(index=C, columns=I, fill_value=0.0)  # domestic make
    imports = _cols(SUPPLY_IMPORT_COLS)
    denom = (M.sum(axis=1) + imports).replace(0.0, np.nan)
    share_dom = M.div(denom, axis=0).fillna(0.0)            # C x I
    share_row = (imports / denom).fillna(0.0)               # C

    def _buy(buyer: str, demand: pd.Series) -> None:
        """Route a commodity demand vector (purchaser prices) to sellers."""
        demand = demand.reindex(C).fillna(0.0)
        basic = demand * basic_rate
        # margins: rebooked as basic-price demand for the margin
        # commodities, split by each column's redistribution weights
        mdemand = pd.Series(0.0, index=C)
        for col in SUPPLY_MARGIN_COLS:
            pool = float((demand * (margin_pos[col] / t016).fillna(0.0)).sum())
            mdemand += pool * margin_supply_w[col]
        routed = basic + mdemand
        flows.loc[buyer, I] += routed @ share_dom
        flows.loc[buyer, "ROW"] += float(routed @ share_row)
        flows.loc[buyer, "GOV"] += float(demand @ tax_rate)

    for j in I:  # intermediate demand
        _buy(j, U[j])
    for f_col, buyer in FINAL_DEMAND_BUYERS.items():  # final demand
        if f_col in Fd.columns:
            _buy(buyer, Fd[f_col])

    # --- value added: industry pays HH / CAP / GOV ---------------------
    # Values as stored: positive -> industry pays; negative (subsidies)
    # -> counterparty pays the industry.
    for code, target in VA_ROWS.items():
        if code not in t.use.index:
            continue
        vals = t.use.loc[code, I].fillna(0.0)
        flows.loc[I, target] += vals.clip(lower=0.0).values
        flows.loc[target, I] += (-vals).clip(lower=0.0).values

    # --- net-lending closure into CAP ----------------------------------
    discrepancy = {}
    for sector in ("HH", "GOV", "ROW"):
        net = float(flows[sector].sum() - flows.loc[sector].sum())  # in - out
        if net >= 0:
            flows.loc[sector, "CAP"] += net
        else:
            flows.loc["CAP", sector] += -net
        discrepancy[sector + "_net_lending"] = net
    cap_residual = float(flows["CAP"].sum() - flows.loc["CAP"].sum())
    discrepancy["CAP_statistical_discrepancy"] = cap_residual

    vals = flows.to_numpy(copy=True)
    np.fill_diagonal(vals, 0.0)  # self-loops carry no register information
    flows = pd.DataFrame(vals, index=flows.index, columns=flows.columns)

    div = flows.sum(axis=0) - flows.sum(axis=1)  # inflow - outflow
    diagnostics = {
        "divergence": div,
        "total_flow": float(flows.values.sum()),
        **discrepancy,
    }
    return flows, diagnostics


# ----------------------------------------------------------------------
# Complex + harmonic register
# ----------------------------------------------------------------------

@dataclasses.dataclass
class Register:
    vertices: list[str]
    edges: list[tuple[int, int]]      # oriented tail->head = money direction
    net: np.ndarray                   # net flow per edge (>0 by orientation)
    gross: np.ndarray                 # gross bilateral flow per edge
    triangles: list[tuple[int, int, int]]
    b1: int
    harmonics: np.ndarray             # E x b1, R-orthonormal
    gram: np.ndarray                  # b1 x b1 Gram of fundamental cycles
    d1: np.ndarray                    # V x E boundary
    d2: np.ndarray                    # E x T boundary
    metric: np.ndarray                # R diagonal (energy weights)
    div_residual: float


def build_register(
    flows: pd.DataFrame,
    netting_threshold: float = 0.0,
    metric_mode: str = "conductance",
    edge_floor: float = 0.0,
) -> Register:
    """Oriented complex + weighted harmonic register from a flow matrix.

    Pairwise netting first (2-cycles are bookkeeping); an edge exists where
    the net bilateral flow exceeds ``edge_floor``. A triangle is filled
    when its three net edges form a directed 3-cycle whose minimum flow is
    at least ``netting_threshold`` (that minimum is the circulation a
    bookkeeper could cancel around the triad).

    Hodge conventions: conservation (Gauss law) is metric-free — a flow f
    is conserved iff d1 @ f = 0 in raw dollars. Netting subtracts raw
    circulations around filled faces (im d2). The metric R enters only as
    the energy inner product <f, g> = f^T R g deciding orthogonality:
    harmonic space H = ker d1 ∩ (im d2)^⊥_R, dim H = b1. metric_mode:
    'conductance' (R = 1/gross — thick relationships are easy directions),
    'length' (R = gross — the length ∝ value convention), 'unit' (R = I).
    """
    V = list(flows.index)
    A = flows.values
    net = A - A.T
    gross = A + A.T

    edges, net_w, gross_w = [], [], []
    n = len(V)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(net[i, j]) > edge_floor:
                if net[i, j] > 0:
                    edges.append((i, j))
                else:
                    edges.append((j, i))
                net_w.append(abs(net[i, j]))
                gross_w.append(gross[i, j])
    net_w = np.asarray(net_w)
    gross_w = np.asarray(gross_w)
    eidx = {e: k for k, e in enumerate(edges)}

    def _edge(i: int, j: int) -> tuple[int, float]:
        """(index, sign) of ordered pair i->j against stored orientation."""
        if (i, j) in eidx:
            return eidx[(i, j)], +1.0
        return eidx[(j, i)], -1.0

    # directed 3-cycles with min net flow >= threshold get filled
    triangles = []
    adj = {e: w for e, w in zip(edges, net_w)}
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                cyc = None
                for tri in (((i, j), (j, k), (k, i)), ((j, i), (i, k), (k, j))):
                    if all(e in adj for e in tri):
                        cyc = min(adj[e] for e in tri)
                        break
                if cyc is not None and cyc >= netting_threshold:
                    triangles.append((i, j, k))

    E, T = len(edges), len(triangles)
    d1 = np.zeros((n, E))
    for k, (i, j) in enumerate(edges):
        d1[i, k] = -1.0  # tail
        d1[j, k] = +1.0  # head
    d2 = np.zeros((E, T))
    for tcol, (i, j, k) in enumerate(triangles):
        for a, b in ((i, j), (j, k), (k, i)):
            idx, sign = _edge(a, b)
            d2[idx, tcol] = sign

    if metric_mode == "conductance":
        R = 1.0 / np.maximum(gross_w, 1e-12)
    elif metric_mode == "length":
        R = np.maximum(gross_w, 1e-12)
    elif metric_mode == "unit":
        R = np.ones(E)
    else:
        raise ValueError(f"unknown metric_mode {metric_mode!r}")
    # normalize the metric scale (units drop out of projections); median
    # rather than mean — near-zero-gross edges give R a huge dynamic
    # range and the mean is outlier-dominated
    R = R / np.median(R)

    # harmonic space: ker d1 ∩ ker (d2^T R). Row-normalize the stacked
    # constraint matrix before the SVD — row scaling preserves the null
    # space exactly and keeps the rank threshold meaningful when R spans
    # many orders of magnitude.
    K = np.vstack([d1, d2.T @ np.diag(R)])
    norms = np.linalg.norm(K, axis=1)
    K = K[norms > 0] / norms[norms > 0, None]
    _, s, vt = np.linalg.svd(K, full_matrices=True)
    rank = int((s > s.max() * max(K.shape) * np.finfo(float).eps).sum()) if len(s) else 0
    H = vt[rank:].T  # E x b1, l2-orthonormal
    b1 = H.shape[1]

    # R-orthonormalize (Gram of the raw null basis is the reported G)
    gram = H.T @ (R[:, None] * H)
    if b1:
        chol = np.linalg.cholesky(gram)
        H = H @ np.linalg.inv(chol).T

    # residual of the observed net flow against conservation (Gate-1 echo)
    div_residual = float(np.linalg.norm(d1 @ net_w) / max(net_w.sum(), 1.0))

    return Register(V, edges, net_w, gross_w, triangles, b1, H, gram, d1, d2, R,
                    div_residual)


def periods(reg: Register, f: np.ndarray) -> np.ndarray:
    """Period coordinates of an edge cochain f: R-projection onto harmonics."""
    return reg.harmonics.T @ (reg.metric * f)


def circuits(reg: Register, top: int = 6) -> list[list[tuple[str, str, float]]]:
    """Top participating edges of each harmonic mode, for interpretation."""
    out = []
    for m in range(reg.b1):
        h = reg.harmonics[:, m]
        order = np.argsort(-np.abs(h))[:top]
        out.append(
            [(reg.vertices[reg.edges[k][0]], reg.vertices[reg.edges[k][1]],
              float(h[k])) for k in order]
        )
    return out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data", type=str)
    ap.add_argument("--year", default=2017, type=int)
    ap.add_argument("--tau", default=0.0, type=float,
                    help="netting threshold (millions USD)")
    ap.add_argument("--edge-floor", default=0.0, type=float)
    ap.add_argument("--metric", default="conductance",
                    choices=("conductance", "length", "unit"))
    args = ap.parse_args()

    t = load_year(args.data, args.year)
    print(f"== Gate 1: accounting identities, {args.year} ==")
    for name, resid in gate1_report(t).items():
        print(f"  {name:28s} rel. residual {resid:.3e}")

    flows, diag = build_money_flows(t)
    div = diag["divergence"]
    print(f"\n== money-flow network: {len(flows)} vertices, "
          f"total flow {diag['total_flow']:.3e} musd ==")
    print(f"  max |divergence| / total flow: "
          f"{div.abs().max() / diag['total_flow']:.3e}")
    for k in ("HH_net_lending", "GOV_net_lending", "ROW_net_lending",
              "CAP_statistical_discrepancy"):
        print(f"  {k:28s} {diag[k]:.4g} musd")

    reg = build_register(flows, args.tau, args.metric, args.edge_floor)
    print(f"\n== register (tau={args.tau:g}, metric={args.metric}) ==")
    print(f"  edges {len(reg.edges)}, filled triangles {len(reg.triangles)}, "
          f"b1 = {reg.b1}")
    print(f"  divergence residual of observed net flow: {reg.div_residual:.3e}")
    if reg.b1:
        p = periods(reg, reg.net)
        print(f"  periods of the observed flow (top 10 |p|):")
        for m in np.argsort(-np.abs(p))[:10]:
            print(f"    mode {m:4d}  p = {p[m]:+.4e}")
        print("  top circuits (mode: edge participation):")
        for m, circ in enumerate(circuits(reg)[:5]):
            desc = ", ".join(f"{a}->{b} ({w:+.2f})" for a, b, w in circ)
            print(f"    mode {m}: {desc}")


if __name__ == "__main__":
    main()
