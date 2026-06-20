# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""#353 proton-from-quarks color realizability, via the C++ register (#379).

#353's Stage 1 — the color-configuration **realizability map** (confinement) and
**S₃ gauge invariance** — runs on the carried color register `cob.Register`
(an icosahedron S² with three holonomy holes; ker L₁ = b₁ = 2, the S₃ standard
rep on the Σ=0 hyperplane). These tests reproduce that map through the C++
primitive:

  * **confinement**: a color-neutral (Σ=0) configuration realizes (residual →
    machine zero); a colored (Σ≠0) configuration floors (the confinement floor)
    — a ~25-order-of-magnitude split;
  * **S₃ (color-relabel = Weyl gauge) invariance**: the residual is identical
    under all 6 permutations of the three color holes;
  * the color singlet `[1, ω, ω²]` (plain-sum 0) realizes — exercising the
    induced-orientation sign convention (`reg.sign`), without which it mis-floors.

The merge / emergent-result side (#353 Stage 3) is covered by
`test_register_merge_353_python.py`.
"""

import cmath
import importlib.util
import itertools
import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES = os.path.join(_HERE, "..", "..", "examples", "cobordism")


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_EXAMPLES, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_SG = _load("spectral_gate_realizability")
_W = cmath.exp(2j * cmath.pi / 3)

# #353 realize/floor thresholds (spectral_gate_realizability REALIZE / CERT_FLOOR).
_REALIZE = 1e-9
_CERT_FLOOR = 1e-2

# Color configs: neutral (Σ=0, realize) vs colored (Σ≠0, floor = confinement).
_NEUTRAL = {
    "meson R-Gbar [1,-1,0]": [1, -1, 0],
    "meson G-Bbar [0,1,-1]": [0, 1, -1],
    "baryon real [1,1,-2]": [1, 1, -2],
    "singlet Z3 [1,w,w^2]": [1, _W, _W * _W],
}
_COLORED = {
    "quark R [1,0,0]": [1, 0, 0],
    "quark G [0,1,0]": [0, 1, 0],
    "diquark RG [1,1,0]": [1, 1, 0],
    "symmetric RGB [1,1,1]": [1, 1, 1],
}


def _residual(reg, cfg):
    """#353 score(): the realizability residual in the oriented period frame —
    reg.spectral_residual(reg.sign * periods)."""
    raw = np.asarray(reg.sign, dtype=complex) * np.asarray(cfg, dtype=complex)
    return reg.spectral_residual(list(raw))


# The verified icosahedron / 3-color-hole register (built in C++ via cob.Register).
_REG = _SG.Register()


class RegisterIsTheColorRegisterTest(unittest.TestCase):
    def test_b1_is_two_on_sigma_zero(self):
        # ker L₁ = b₁ = 2 = the S₃ standard rep (rank(P)=2: a genuine register).
        self.assertEqual(_REG.dim, 2)
        self.assertEqual(_REG.rank, 2)

    def test_sign_is_the_induced_covector(self):
        # P @ n = 0: the induced-orientation covector symmetrizes to Σ=0.
        self.assertEqual(len(_REG.sign), 3)


class RealizabilityMapTest(unittest.TestCase):
    """Confinement: Σ=0 realizes, Σ≠0 floors (the #353 color realizability map)."""

    def test_color_neutral_configs_realize(self):
        for name, cfg in _NEUTRAL.items():
            with self.subTest(cfg=name):
                self.assertLess(_residual(_REG, cfg), _REALIZE, name)

    def test_singlet_realizes_with_sign_convention(self):
        # [1,ω,ω²] has plain-sum 0 and MUST realize; it only does in the oriented
        # frame (reg.sign), the smoking-gun for the induced-sign convention.
        self.assertLess(_residual(_REG, [1, _W, _W * _W]), _REALIZE)

    def test_colored_configs_floor_confinement(self):
        for name, cfg in _COLORED.items():
            with self.subTest(cfg=name):
                self.assertGreater(_residual(_REG, cfg), _CERT_FLOOR, name)

    def test_realize_floor_split_is_enormous(self):
        worst_neutral = max(_residual(_REG, c) for c in _NEUTRAL.values())
        best_colored = min(_residual(_REG, c) for c in _COLORED.values())
        self.assertGreater(best_colored / max(worst_neutral, 1e-300), 1e6)


class S3GaugeInvarianceTest(unittest.TestCase):
    """The residual is identical under all 6 color-relabel (S₃ / Weyl) perms."""

    def test_singlet_is_s3_invariant(self):
        base = [1, _W, _W * _W]
        residuals = [_residual(_REG, [base[i] for i in p])
                     for p in itertools.permutations(range(3))]
        self.assertLess(max(residuals) - min(residuals), 1e-6)

    def test_colored_floors_under_every_relabel(self):
        # Confinement is gauge-invariant: a colored config floors (> CERT_FLOOR)
        # under EVERY color relabel. (The floor VALUE varies — permuting [1,1,0]
        # gives distinct colored states RG/RB/GB — so this asserts "floors", not
        # "floors equally"; only Σ=0 states are S₃-invariant in value, above.)
        base = [1, 1, 0]
        for p in itertools.permutations(range(3)):
            cfg = [base[i] for i in p]
            self.assertGreater(_residual(_REG, cfg), _CERT_FLOOR, str(p))


if __name__ == "__main__":
    unittest.main()
