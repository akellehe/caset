# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

import unittest
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PKG_DIR = REPO_ROOT / "tessera"

class TestBuilding(unittest.TestCase):
    def test_building(self):
        """Verify that tessera is importable — from cmake-build-debug, pip install -e ., or pip install ."""
        # The tessera package is a Python wrapper; check the C extension underneath.
        spec = importlib.util.find_spec("tessera._tessera")
        self.assertIsNotNone(spec, "tessera._tessera C extension not found")
        self.assertIsNotNone(spec.origin, "tessera._tessera has no origin")

        print("  origin:", spec.origin)

        origin = Path(spec.origin).resolve()

        if "site-packages" in str(origin):
            # Loaded from a pip install.  Accept both editable (`pip install -e .`)
            # and regular (`pip install .`) local installs — both produce a
            # direct_url.json with a file:// URL pointing at the source tree.
            # Only reject if there's no direct_url at all (stale system wheel).
            from importlib.metadata import distribution
            dist = distribution("tessera")
            direct_url = dist.read_text("direct_url.json") or ""
            self.assertTrue(
                direct_url,
                f"tessera._tessera loaded from site-packages without a direct_url.json — "
                f"stale install? origin: {spec.origin}")
        elif origin.parent == SOURCE_PKG_DIR:
            # Loaded from the source-tree tessera/ package. conftest.py copies
            # the freshly-built .so here from cmake-build-debug so that the
            # source-tree package (which shadows site-packages when running
            # from the repo root) has a usable _tessera submodule. Verify a
            # matching build actually exists so we don't accept a stale copy.
            build_root = REPO_ROOT / "cmake-build-debug"
            matching = list(build_root.glob(f"*/{origin.name}")) if build_root.exists() else []
            self.assertTrue(
                matching,
                f"tessera._tessera loaded from source tree ({origin}) but no matching "
                f"build found under {build_root} — stale copy?")
        else:
            # Loaded from cmake-build-debug (conftest.py injected it)
            found = any("cmake-build-debug" in p for p in sys.path)
            self.assertTrue(
                found,
                "cmake-build-debug not on sys.path and tessera is not a pip install")
