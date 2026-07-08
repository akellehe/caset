"""One-off export of BEA summary-level Make + Use tables from the TVL platform.

Exploratory spike scaffolding for tessera#602 — see README.md in this
directory. Reads the ``bea`` app's database through its Django ORM and
writes long-form CSVs plus a manifest. Read-only; no TVL repo changes.

Run from a TVL checkout with the ``bea`` virtualenv and DB credentials
in the environment (the documented invocation)::

    cd ~/tvl && eval "$(poetry run poe dbcreds 2>/dev/null)"
    PYTHONPATH=~/tvl/bea/src:~/tvl/tvl/src \
    DJANGO_SETTINGS_MODULE=bea.dal.settings \
    ~/tvl/bea/.venv/bin/python \
      ~/tessera-econ-register-spike/examples/cobordism/econ/fetch_bea_io.py \
      --out ~/tessera-econ-register-spike/examples/cobordism/econ/data

Outputs (long form, ``IODataPoint.RAW_DATAFRAME_COLUMNS``):

* ``use_summary.csv``  — TableID 259 (Use, summary; rows=commodities, cols=industries)
* ``make_summary.csv`` — TableID 262 (Make, summary; rows=industries, cols=commodities)
* ``manifest.json``    — export timestamp, year coverage, per-year row/code
  counts, and a cross-year code-set consistency check (the cheap vintage
  red-flag: a consistent vintage presents one industry code set for every
  year; a changing set means mixed revisions or reclassification).
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib

import django

django.setup()

from django.db import connections  # noqa: E402

from bea.dal.models import IODataPoint  # noqa: E402

USE_SUMMARY_TABLE_ID = 259
MAKE_SUMMARY_TABLE_ID = 262
DEFAULT_SINCE = 1997
DEFAULT_UNTIL = datetime.date.today().year


def export_table(table_id: int, since: int, until: int, out_path: pathlib.Path) -> dict:
    """Query one API IO table into CSV; return manifest facts about it."""
    df = IODataPoint.query_table(table_id, since, until)
    if df.empty:
        raise SystemExit(f"TableID {table_id}: no rows in [{since}, {until}] — aborting.")

    df = df.sort_values(["year", "row_code", "col_code"], kind="stable")
    df.to_csv(out_path, index=False)

    per_year = {
        int(year): {
            "rows": int(len(g)),
            "row_codes": int(g["row_code"].nunique()),
            "col_codes": int(g["col_code"].nunique()),
            "null_values": int(g["value"].isna().sum()),
        }
        for year, g in df.groupby("year")
    }
    # Vintage red-flag: the code sets should be identical in every year.
    row_sets = {year: frozenset(g["row_code"]) for year, g in df.groupby("year")}
    col_sets = {year: frozenset(g["col_code"]) for year, g in df.groupby("year")}
    return {
        "table_id": table_id,
        "file": out_path.name,
        "years": sorted(int(y) for y in df["year"].unique()),
        "per_year": per_year,
        "row_code_set_consistent": len(set(row_sets.values())) == 1,
        "col_code_set_consistent": len(set(col_sets.values())) == 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--since", type=int, default=DEFAULT_SINCE)
    parser.add_argument("--until", type=int, default=DEFAULT_UNTIL)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_database": connections["default"].settings_dict.get("NAME"),
        "vintage_caveat": (
            "Data as ingested in the TVL bea database at export time; the BEA "
            "API serves its current revision of the whole series, so a "
            "single-session ingest is one vintage, but incremental gap-fill "
            "ingests can mix revisions. Code-set consistency below is the "
            "cheap red-flag check, not proof."
        ),
        "tables": [
            export_table(
                USE_SUMMARY_TABLE_ID, args.since, args.until, args.out / "use_summary.csv"
            ),
            export_table(
                MAKE_SUMMARY_TABLE_ID, args.since, args.until, args.out / "make_summary.csv"
            ),
        ],
    }

    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    for table in manifest["tables"]:
        years = table["years"]
        print(
            f"TableID {table['table_id']} -> {table['file']}: "
            f"years {years[0]}–{years[-1]} ({len(years)}), "
            f"row-set consistent={table['row_code_set_consistent']}, "
            f"col-set consistent={table['col_code_set_consistent']}"
        )
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
