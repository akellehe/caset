#!/usr/bin/env python3
"""
fetch_arxiv_pdfs.py
-------------------
Download arXiv PDFs for every entry in a BibTeX file that has an
`eprint = {...}` field, then emit a new BibTeX file with
`file = {:<abs-path>:application/pdf}` fields attached so Zotero
auto-links the PDFs on import.

Usage
-----
    python3 fetch_arxiv_pdfs.py quark-majorization-emergent-spacetime.bib

Options
-------
    --out-dir DIR   where to save PDFs (default: ./pdfs)
    --out-bib FILE  updated .bib path (default: <input>_with_files.bib)
    --dry-run       list what would be fetched, download nothing
    --retries N     per-file retry count (default: 2)

Workflow
--------
    1. Run this script.
    2. In Zotero: File -> Import... -> pick the *_with_files.bib file.
       Zotero creates a collection named after the file and attaches
       the PDFs by absolute path.
    3. Paywalled / pre-arXiv entries (Kogut-Susskind 1975, Lieb-Robinson
       1972, Regge 1961, the two books) need manual attachment; the
       script lists them at the end.

Zero external dependencies; stdlib only. Python 3.8+.

Notes on politeness
-------------------
arXiv asks automated clients for >= 3s between requests. This script
sleeps 3.5s. Do not parallelize.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ARXIV_PDF_URL = "https://arxiv.org/pdf/{eprint}.pdf"
USER_AGENT = "bib-fetcher/1.0 (academic; contact: local user)"
SLEEP_SECONDS = 3.5
MIN_PDF_BYTES = 4096  # below this, treat as a redirect/error page


# -------- BibTeX parsing (brace-aware, stdlib only) --------

def parse_entries(text: str) -> list[dict]:
    """Return a list of entries: {type, key, fields, raw, span}."""
    entries = []
    i = 0
    n = len(text)
    while i < n:
        at = text.find("@", i)
        if at < 0:
            break
        m = re.match(r"@(\w+)\s*\{", text[at:])
        if not m:
            i = at + 1
            continue
        if m.group(1).lower() in {"comment", "preamble", "string"}:
            i = at + 1
            continue
        # balanced-brace scan for the entry body
        start = at + m.end() - 1  # index of the '{' after @type
        depth = 0
        j = start
        while j < n:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            break
        body = text[start + 1 : j]
        comma = body.find(",")
        if comma < 0:
            i = j + 1
            continue
        key = body[:comma].strip()
        fields = _parse_fields(body[comma + 1 :])
        entries.append({
            "type": m.group(1),
            "key": key,
            "fields": fields,
            "raw": text[at : j + 1],
            "span": (at, j + 1),
        })
        i = j + 1
    return entries


def _parse_fields(blob: str) -> dict[str, str]:
    """Parse `name = {value}` or `name = "value"` pairs, brace-aware."""
    fields: dict[str, str] = {}
    i, n = 0, len(blob)
    while i < n:
        while i < n and blob[i] in " \t\n\r,":
            i += 1
        if i >= n:
            break
        m = re.match(r"(\w+)\s*=\s*", blob[i:])
        if not m:
            break
        name = m.group(1).lower()
        i += m.end()
        if i >= n:
            break
        if blob[i] == "{":
            depth, j = 1, i + 1
            while j < n and depth > 0:
                if blob[j] == "{":
                    depth += 1
                elif blob[j] == "}":
                    depth -= 1
                j += 1
            fields[name] = blob[i + 1 : j - 1].strip()
            i = j
        elif blob[i] == '"':
            j = blob.find('"', i + 1)
            if j < 0:
                break
            fields[name] = blob[i + 1 : j].strip()
            i = j + 1
        else:
            j = i
            while j < n and blob[j] not in ",\n":
                j += 1
            fields[name] = blob[i:j].strip()
            i = j
    return fields


# -------- Downloading --------

def download(eprint: str, dest: Path, retries: int = 2) -> None:
    url = ARXIV_PDF_URL.format(eprint=eprint)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if len(data) < MIN_PDF_BYTES:
                raise RuntimeError(f"response too small: {len(data)} bytes")
            if not data[:5].startswith(b"%PDF-"):
                raise RuntimeError("response is not a PDF (got HTML redirect?)")
            dest.write_bytes(data)
            return
        except (urllib.error.URLError, RuntimeError) as ex:
            last_err = ex
            if attempt < retries:
                time.sleep(SLEEP_SECONDS * 2)
    assert last_err is not None
    raise last_err


# -------- Rewriting the bib --------

def splice_file_field(raw_entry: str, abs_pdf_path: Path) -> str:
    """Insert or replace a `file = {:<path>:application/pdf}` field."""
    # Zotero-compatible triple: description:path:mimetype
    field = f"  file    = {{:{abs_pdf_path}:application/pdf}}"
    if re.search(r"\n\s*file\s*=", raw_entry):
        return re.sub(
            r"\n\s*file\s*=\s*\{[^}]*\}\s*,?",
            "\n" + field + ",",
            raw_entry,
            count=1,
        )
    # Insert before the final closing brace, preserving trailing comma rules.
    return re.sub(r"\n\}\s*$", f",\n{field}\n}}", raw_entry, count=1)


# -------- Main --------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bib", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("pdfs"))
    ap.add_argument("--out-bib", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    bib_path: Path = args.bib.resolve()
    if not bib_path.is_file():
        print(f"error: {bib_path} not found", file=sys.stderr)
        return 1

    out_dir: Path = args.out_dir.resolve()
    out_bib: Path = (args.out_bib or bib_path.with_name(bib_path.stem + "_with_files.bib")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    text = bib_path.read_text(encoding="utf-8")
    entries = parse_entries(text)
    print(f"parsed {len(entries)} entries from {bib_path.name}")
    print(f"pdfs -> {out_dir}")
    print(f"new bib -> {out_bib}")
    print()

    fetched: list[str] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []
    no_arxiv: list[str] = []

    updated_text = text

    for e in entries:
        key = e["key"]
        eprint = e["fields"].get("eprint")
        if not eprint:
            no_arxiv.append(key)
            continue

        safe = eprint.replace("/", "_")
        pdf_path = out_dir / f"{key}__{safe}.pdf"

        if pdf_path.exists() and pdf_path.stat().st_size >= MIN_PDF_BYTES:
            skipped.append(key)
            print(f"[=] {key}  cached")
        elif args.dry_run:
            print(f"[?] {key}  would fetch arxiv:{eprint}")
            continue
        else:
            print(f"[.] {key}  arxiv:{eprint} ...", end=" ", flush=True)
            try:
                download(eprint, pdf_path, retries=args.retries)
                kb = pdf_path.stat().st_size // 1024
                print(f"ok ({kb} KB)")
                fetched.append(key)
                time.sleep(SLEEP_SECONDS)
            except Exception as ex:
                print(f"FAILED ({ex})")
                failed.append((key, str(ex)))
                if pdf_path.exists():
                    pdf_path.unlink()
                continue

        # Splice the file field into the raw entry.
        new_raw = splice_file_field(e["raw"], pdf_path)
        updated_text = updated_text.replace(e["raw"], new_raw, 1)

    if not args.dry_run:
        out_bib.write_text(updated_text, encoding="utf-8")

    print()
    print("=" * 60)
    print(f"fetched  {len(fetched):>3}")
    print(f"cached   {len(skipped):>3}")
    print(f"failed   {len(failed):>3}  {[k for k,_ in failed] or ''}")
    print(f"no-arxiv {len(no_arxiv):>3}  {no_arxiv or ''}")
    if not args.dry_run:
        print()
        print(f"new bib written: {out_bib}")
        print("next step: Zotero -> File -> Import... -> pick that file")
        print("entries without arXiv IDs (books, pre-1991 papers) need manual")
        print("attachment; the .bib still imports them as metadata-only.")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
