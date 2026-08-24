#!/usr/bin/env python3
"""Regenerate the Markdown edition of the Recursive Spectral Fibers whitepaper.

The LaTeX source is the single source of truth for the paper, including its
bibliography: the source carries an inline ``thebibliography`` environment
whose ``\\bibitem`` keys are the ones its ``\\cite`` commands resolve against,
and the compiled PDF renders those entries under numeric labels.

A direct pandoc conversion loses that.  Pandoc's GitHub-Flavored Markdown
writer has no bibliography engine of its own, so it renders every ``\\cite``
command as empty text and every bibliography entry without its label -- the
sentence "deficit angles \\cite{regge1961}, discrete" converts to "deficit
angles , discrete" and the reader is left with an unlabelled list of works no
sentence points at.  Running a citation processor instead is not an option
here, because that requires an external bibliography database whose keys match
the source, and the inline environment is deliberately the only bibliography
this paper has.

This module therefore preprocesses the source before handing it to pandoc.
Every ``\\cite`` becomes the bracketed numbers of the works it names, and every
bibliography entry is prefixed with its own number, reproducing the numeric
scheme the PDF already uses.  The conversion is otherwise pandoc's.

Usage:
    python3 scripts/whitepaper_markdown.py           # rewrite the Markdown
    python3 scripts/whitepaper_markdown.py --check   # fail if it would change
"""
import argparse
import difflib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DESIGN_DIR = REPO_ROOT / "docs" / "design"
TEX_PATH = DESIGN_DIR / "recursive_spectral_fibers_whitepaper.tex"
MD_PATH = DESIGN_DIR / "recursive_spectral_fibers_whitepaper.md"

# The Markdown edition carries the vector figures as captions only; the note
# says so once, immediately before the first section.
COMPANION_NOTE = (
    "> The rendered vector diagrams are preserved in the LaTeX/PDF edition; "
    "this Markdown edition is the searchable text companion."
)

PANDOC_ARGS = ["-f", "latex", "-t", "gfm", "--wrap=none"]


def bibliography_order(tex):
    """The ``\\bibitem`` keys of the inline bibliography, in citation order.

    Returns a dict mapping each key to its one-based label, which is the label
    the compiled PDF prints for that entry.
    """
    keys = re.findall(r"\\bibitem\{([^}]*)\}", tex)
    if not keys:
        raise SystemExit(
            "no \\bibitem entries found: the source no longer carries an "
            "inline bibliography, so this script's numbering scheme no longer "
            "applies"
        )
    duplicates = {key for key in keys if keys.count(key) > 1}
    if duplicates:
        raise SystemExit(f"duplicate \\bibitem keys: {sorted(duplicates)}")
    return {key: number for number, key in enumerate(keys, start=1)}


def split_cite_keys(argument):
    """The keys of one ``\\cite`` argument.

    LaTeX permits the argument to run across lines and to use a trailing ``%``
    to swallow the newline, so neither whitespace nor comment markers are part
    of a key.
    """
    argument = re.sub(r"%.*?(?:\n|$)", "", argument)
    return [key.strip() for key in argument.split(",") if key.strip()]


def resolve_citations(tex, labels):
    """Replace every ``\\cite`` with the bracketed labels of the works it names.

    Unknown keys are a hard error: silently dropping one would reproduce
    exactly the failure this script exists to prevent.
    """
    unknown = []

    def replace(match):
        keys = split_cite_keys(match.group(1))
        for key in keys:
            if key not in labels:
                unknown.append(key)
        numbers = [str(labels[key]) for key in keys if key in labels]
        return "[" + ", ".join(numbers) + "]"

    resolved = re.sub(r"\\cite\{([^}]*)\}", replace, tex, flags=re.DOTALL)
    if unknown:
        raise SystemExit(
            f"\\cite keys with no \\bibitem entry: {sorted(set(unknown))}"
        )
    return resolved


def label_bibliography(tex, labels):
    """Prefix each bibliography entry with the label the PDF prints for it."""

    def replace(match):
        key = match.group(1)
        return f"\\bibitem{{{key}}} [{labels[key]}]"

    return re.sub(r"\\bibitem\{([^}]*)\}", replace, tex)


def run_pandoc(tex):
    """Convert preprocessed LaTeX to Markdown."""
    if shutil.which("pandoc") is None:
        raise SystemExit(
            "pandoc is required to generate the Markdown edition.\n"
            "  Debian/Ubuntu: sudo apt-get install pandoc"
        )
    # Pandoc resolves \input and friends relative to the working directory, so
    # run it from the directory the source lives in.
    with tempfile.TemporaryDirectory() as scratch:
        staged = Path(scratch) / TEX_PATH.name
        staged.write_text(tex, encoding="utf-8")
        result = subprocess.run(
            ["pandoc", *PANDOC_ARGS, str(staged)],
            capture_output=True,
            text=True,
            cwd=DESIGN_DIR,
        )
    if result.returncode != 0:
        raise SystemExit(f"pandoc failed:\n{result.stderr}")
    return result.stdout


def insert_companion_note(markdown):
    """Place the companion note immediately before the first section."""
    if COMPANION_NOTE in markdown:
        return markdown
    match = re.search(r"^# ", markdown, flags=re.MULTILINE)
    if match is None:
        raise SystemExit("no section heading found: cannot place the note")
    head, tail = markdown[: match.start()], markdown[match.start() :]
    return f"{head.rstrip()}\n\n{COMPANION_NOTE}\n\n{tail}"


def generate():
    """The Markdown edition the current LaTeX source implies."""
    tex = TEX_PATH.read_text(encoding="utf-8")
    labels = bibliography_order(tex)
    prepared = label_bibliography(resolve_citations(tex, labels), labels)
    markdown = insert_companion_note(run_pandoc(prepared))
    return markdown.rstrip("\n") + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero if the committed Markdown differs from the source",
    )
    args = parser.parse_args()

    generated = generate()

    if not args.check:
        MD_PATH.write_text(generated, encoding="utf-8")
        print(f"wrote {MD_PATH.relative_to(REPO_ROOT)}")
        return 0

    if not MD_PATH.exists():
        print(f"{MD_PATH.relative_to(REPO_ROOT)} does not exist", file=sys.stderr)
        return 1

    committed = MD_PATH.read_text(encoding="utf-8")
    if committed == generated:
        print(f"{MD_PATH.relative_to(REPO_ROOT)} is current")
        return 0

    diff = difflib.unified_diff(
        committed.splitlines(keepends=True),
        generated.splitlines(keepends=True),
        fromfile="committed",
        tofile="generated",
    )
    sys.stderr.writelines(diff)
    print(
        f"\n{MD_PATH.relative_to(REPO_ROOT)} has drifted from "
        f"{TEX_PATH.relative_to(REPO_ROOT)}.\n"
        "Regenerate it with: python3 scripts/whitepaper_markdown.py",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
