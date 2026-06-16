# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Durability guard for the C++ API documentation page.

``docs/source/cpp_api.md`` lists every public header under ``include/`` via a
Breathe ``{doxygenfile} <header>`` directive. This test keeps that list honest:
it fails CI if a header is added under ``include/`` without a matching entry on
the page (so the docs can't silently rot), and equally if the page references a
header that no longer exists (catches typos / removed files).

Pure filesystem/text check — it does not import ``tessera`` and needs no C++
build, so it runs in the docs-only environment too.
"""

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INCLUDE_DIR = REPO_ROOT / "include"
CPP_API_PAGE = REPO_ROOT / "docs" / "source" / "cpp_api.md"

# ``{doxygenfile} Foo.h`` / ``{doxygenfile} Foo.hpp`` — capture the basename.
_DOXYGENFILE_RE = re.compile(r"\{doxygenfile\}\s+(\S+\.(?:h|hpp))\s*$", re.MULTILINE)


class TestCppApiDocsCoverage(unittest.TestCase):
    """Every public C++ header must be reachable from the C++ API page."""

    @classmethod
    def setUpClass(cls):
        cls.headers = {
            p.name
            for p in INCLUDE_DIR.rglob("*")
            if p.suffix in (".h", ".hpp")
        }
        text = CPP_API_PAGE.read_text(encoding="utf-8")
        cls.documented = set(_DOXYGENFILE_RE.findall(text))

    def test_include_dir_is_populated(self):
        """Sanity: we actually found headers and directives to compare."""
        self.assertTrue(self.headers, f"no headers found under {INCLUDE_DIR}")
        self.assertTrue(
            self.documented,
            f"no {{doxygenfile}} directives found in {CPP_API_PAGE}",
        )

    def test_every_header_is_documented(self):
        """No header under include/ may be missing from cpp_api.md."""
        missing = sorted(self.headers - self.documented)
        self.assertFalse(
            missing,
            "Public C++ headers missing a {doxygenfile} entry in "
            f"docs/source/cpp_api.md: {missing}. Add each to the matching "
            "per-module section so it is reachable from the C++ API docs.",
        )

    def test_no_stale_documented_headers(self):
        """No {doxygenfile} entry may point at a header that no longer exists."""
        stale = sorted(self.documented - self.headers)
        self.assertFalse(
            stale,
            "docs/source/cpp_api.md references headers that do not exist under "
            f"include/: {stale}. Remove or rename the stale {{doxygenfile}} "
            "entries.",
        )


if __name__ == "__main__":
    unittest.main()
