# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The FP32 cuBLAS GPU r_U gradient returns the SAME gradient as the FP64 CPU
oracle (#348).

`EigenstateSynthesis.residualForPeriodsGradientGpu` (FP32 cuBLAS) must agree with
`EigenstateSynthesis.residualForPeriodsGradient` (FP64 CPU, the correctness
oracle) to single precision — element-wise, in L2, and in direction — across
substrate levels, perturbation scales/patterns, and register targets. "Same"
means: matches to FP32 tolerance (the gradients are not bit-identical by design,
since one is FP32 and the other FP64), with an identical descent direction.

GPU-gated: skipped without a CUDA build + a GPU. The slow level-2 scale check is
opt-in via RU_GPU_LEVEL2=1 (a single CPU-oracle call there is ~90s)."""
import importlib.util
import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EX = os.path.join(_HERE, "..", "..", "examples", "cobordism")
sys.path.insert(0, os.path.join(_EX, "deep_merge_baseline"))
sys.path.insert(0, _EX)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_EX, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MC = _load("merge_cobordism")
tessera = MC.tessera
cob = tessera.cobordism
import probe_deep as PD  # noqa: E402  (build_merge for the subdivided substrates)


def _gpu_available():
    """The GPU path is live iff the method exists and a trivial call succeeds
    (a CPU-only TESSERA_CUDA=OFF build has no GPU path)."""
    if not hasattr(cob.EigenstateSynthesis, "residualForPeriodsGradientGpu"):
        return False
    try:
        m = MC.MergeCobordism()
        m.st.materializeFacets()
        holes = [list(t) for t in m.hole_circles]
        tgt = [complex(z) for z in
               np.asarray(m.es.cyclePeriods(holes), complex).reshape(m.dim, 9)[0]]
        m.es.residualForPeriodsGradientGpu(holes, tgt)
        return True
    except Exception:
        return False


_GPU = _gpu_available()


def _level0():
    """Level-0 merge (MergeCobordism, n1=174): st, es, holes, period matrix."""
    m = MC.MergeCobordism()
    m.st.materializeFacets()
    holes = [list(t) for t in m.hole_circles]
    P = np.asarray(m.es.cyclePeriods(holes), complex).reshape(m.dim, len(holes))
    return m.st, m.es, holes, P


def _level(n):
    """Subdivided merge (build_merge(n)): st, es, holes, period matrix."""
    st, nreg, holes3, _hv, _c = PD.build_merge(n)
    st.materializeFacets()
    es = cob.EigenstateSynthesis(st, 1)
    holes = [sorted(v + off for v in h) for off in (0, nreg, 2 * nreg) for h in holes3]
    P = np.asarray(es.cyclePeriods(holes), complex)
    m = len(holes)
    return st, es, holes, P.reshape(len(P) // m, m)


def _emap(st):
    out = {}
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        out[(min(a, b), max(a, b))] = e
    return out


def _perturb(st, es, factor):
    """Scale each cell's edge squared length by factor(i) (1.0 = unchanged)."""
    cells = [tuple(int(v) for v in c) for c in es.cellSimplices()]
    em = _emap(st)
    for i, c in enumerate(cells):
        f = factor(i)
        if f != 1.0:
            ek = (min(c), max(c))
            em[ek].setSquaredLength(em[ek].getSquaredLength().real * f)
    st.materializeFacets()


def _both(es, holes, target):
    """The FP64 CPU oracle gradient and the FP32 GPU gradient at the current geometry."""
    tc = [complex(z) for z in target]
    g_cpu = np.asarray(es.residualForPeriodsGradient(holes, tc), float)
    g_gpu = np.asarray(es.residualForPeriodsGradientGpu(holes, tc), float)
    return g_gpu, g_cpu


@unittest.skipUnless(_GPU, "CUDA build / GPU not available")
class RuGradientGpuEqualsCpuTest(unittest.TestCase):
    """FP32-GPU == FP64-CPU oracle to single precision, everywhere we can check."""

    def _assert_same(self, g_gpu, g_cpu, ctx):
        self.assertEqual(len(g_gpu), len(g_cpu), f"{ctx}: length mismatch")
        self.assertTrue(np.all(np.isfinite(g_gpu)), f"{ctx}: GPU produced non-finite values")
        max_d = float(np.max(np.abs(g_gpu - g_cpu)))
        # element-wise FP32 agreement: |gpu - cpu| <= atol + rtol*|cpu|
        self.assertTrue(
            np.allclose(g_gpu, g_cpu, rtol=1e-3, atol=1e-5),
            f"{ctx}: not allclose (max|Δ|={max_d:.2e}, max|cpu|={np.max(np.abs(g_cpu)):.2e})")
        den = float(np.linalg.norm(g_cpu))
        if den > 1e-9:  # non-trivial gradient: also check L2-relative + direction
            rel = float(np.linalg.norm(g_gpu - g_cpu) / den)
            self.assertLess(rel, 1e-3, f"{ctx}: L2 relative error {rel:.2e}")
            cos = float(np.dot(g_gpu, g_cpu) /
                        (np.linalg.norm(g_gpu) * np.linalg.norm(g_cpu)))
            self.assertGreater(cos, 0.9999, f"{ctx}: descent-direction cosine {cos:.6f}")

    def test_realizable_base(self):
        """At the realizable base (r_U≈0) both gradients are ~0 and agree."""
        st, es, holes, P = _level0()
        g_gpu, g_cpu = _both(es, holes, P[0])
        self._assert_same(g_gpu, g_cpu, "level0 base")

    def test_uniform_perturbation_scales(self):
        """A range of perturbation strengths (so r_U sweeps from small to large)."""
        for s in (1.1, 1.3, 1.6, 2.0, 3.0):
            with self.subTest(scale=s):
                st, es, holes, P = _level0()
                _perturb(st, es, lambda i, s=s: s if i % 2 == 0 else 1.0)
                g_gpu, g_cpu = _both(es, holes, P[0])
                self._assert_same(g_gpu, g_cpu, f"level0 scale={s}")

    def test_perturbation_patterns(self):
        """Different spatial patterns of the perturbation across the edges."""
        patterns = {
            "all*1.3": lambda i: 1.3,
            "every2*1.4": lambda i: 1.4 if i % 2 == 0 else 1.0,
            "every3*1.5": lambda i: 1.5 if i % 3 == 0 else 1.0,
            "ramp": lambda i: 1.0 + 0.4 * ((i % 5) / 4.0),
            "alternating": lambda i: (1.6 if i % 2 == 0 else 0.7),
        }
        for name, fn in patterns.items():
            with self.subTest(pattern=name):
                st, es, holes, P = _level0()
                _perturb(st, es, fn)
                g_gpu, g_cpu = _both(es, holes, P[0])
                self._assert_same(g_gpu, g_cpu, f"level0 pattern={name}")

    def test_register_targets(self):
        """Different register targets (period rows / combinations / scalings)."""
        st, es, holes, P = _level0()
        _perturb(st, es, lambda i: 1.4 if i % 2 == 0 else 1.0)
        targets = {
            "P0": P[0],
            "P1": P[1] if P.shape[0] > 1 else P[0],
            "2*P0": 2.0 * P[0],
            "P0+P1": P[0] + (P[1] if P.shape[0] > 1 else P[0]),
        }
        for name, tgt in targets.items():
            with self.subTest(target=name):
                g_gpu, g_cpu = _both(es, holes, tgt)
                self._assert_same(g_gpu, g_cpu, f"level0 target={name}")

    def test_level1_substrate(self):
        """A larger (once-subdivided) substrate, a couple of perturbations."""
        for s in (1.3, 1.8):
            with self.subTest(scale=s):
                st, es, holes, P = _level(1)
                _perturb(st, es, lambda i, s=s: s if i % 3 == 0 else 1.0)
                g_gpu, g_cpu = _both(es, holes, P[0])
                self._assert_same(g_gpu, g_cpu, f"level1 scale={s}")

    @unittest.skipUnless(os.environ.get("RU_GPU_LEVEL2"),
                         "slow level-2 scale check; set RU_GPU_LEVEL2=1 to run")
    def test_level2_scale(self):
        """The level-2 deep merge (n1=2724) — the worst-conditioned, most
        stringent FP32 case. Slow (a CPU-oracle call is ~90s)."""
        st, es, holes, P = _level(2)
        _perturb(st, es, lambda i: 1.3 if i % 7 == 0 else 1.0)
        g_gpu, g_cpu = _both(es, holes, P[0])
        self._assert_same(g_gpu, g_cpu, "level2 scale")

    def test_gpu_determinism(self):
        """The GPU path is deterministic — same geometry, bit-identical output."""
        st, es, holes, P = _level0()
        _perturb(st, es, lambda i: 1.5 if i % 2 == 0 else 1.0)
        tc = [complex(z) for z in P[0]]
        g1 = np.asarray(es.residualForPeriodsGradientGpu(holes, tc), float)
        g2 = np.asarray(es.residualForPeriodsGradientGpu(holes, tc), float)
        self.assertTrue(np.array_equal(g1, g2), "GPU gradient is not deterministic")


if __name__ == "__main__":
    unittest.main()
