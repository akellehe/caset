"""GPU Euclidean dihedral kernel parity with the CPU path (#581 scope item 7).

``dihedral_from_cm`` (regge_cuda.cu) lacked the CPU path's ``(-1)^d``
diagonal-sign fix: the Cayley-Menger diagonal cofactors are negative in odd
dimension (e.g. tetrahedra), so on a 3D host the GPU computed the SUPPLEMENT
of every dihedral angle before clamping.  The kernel now applies the sign fix
before the clamp, exactly like ``Simplex::dihedralAngle``.

GPU-gated: skipped without a CUDA build + a GPU (the same detection the
r_U-gradient GPU suite uses — the whole build shares one TESSERA_CUDA flag).
"""

import math
import os
import sys
import unittest

import pytest

try:
    import tessera
    _IMPORT_OK = True
except Exception:  # pragma: no cover
    _IMPORT_OK = False

pytestmark = pytest.mark.skipif(not _IMPORT_OK, reason="tessera not built")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "cobordism"))


def _gpu_available():
    """The GPU lane is live iff the CUDA-only EigenstateSynthesis method
    exists and a trivial call succeeds (the CPU-only build throws)."""
    if not _IMPORT_OK:
        return False
    cob = tessera.cobordism
    if not hasattr(cob.EigenstateSynthesis, "residualForPeriodsGradientGpu"):
        return False
    try:
        from _holed_surface import holed_surface
        _st, es, holes, P = holed_surface(degree=1)
        es.residualForPeriodsGradientGpu(holes, [complex(z) for z in P[0]])
        return True
    except Exception:
        return False


_GPU = _IMPORT_OK and _gpu_available()


def _toroid4d(n=60):
    """The 4D CDT toroid — the host the GPU step lane is wired for (hinges =
    triangles with nonzero area, so the action and its gradient are nonzero;
    on d < 4 hosts hingeArea() of a sub-triangle hinge is 0 and the whole
    lane is vacuously zero)."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    st.setSeed(1234)
    st.build(n)
    return st


def _gpu_step(solver):
    """step(0.0) on the GPU lane; a CUDA resource failure (another tenant
    holding the VRAM) is an environment condition, not a correctness result —
    skip, mirroring the suite's skipped-without-GPU convention."""
    try:
        return solver.step(0.0)
    except RuntimeError as exc:  # pragma: no cover - depends on GPU tenancy
        if "out of memory" in str(exc):
            pytest.skip(f"GPU present but VRAM exhausted: {exc}")
        raise


@pytest.mark.skipif(not _GPU, reason="no CUDA build / GPU")
class TestGpuDihedralParity(unittest.TestCase):
    def test_gpu_step_gradient_norm_matches_cpu_on_cdt_toroid(self):
        # On a CUDA build ReggeSolver.step() computes the base action
        # gradient on the GPU through dihedral_from_cm (now sign-fixed and
        # clamped exactly like the CPU Simplex::dihedralAngle); a zero
        # learning rate makes it a pure measurement.  actionGradientNorm()
        # is the CPU finite-difference oracle over the same W-space
        # functional (central vs the kernel's forward difference, hence the
        # loose-ish tolerance).  In the wired d=4 lane the (-1)^d fix is a
        # no-op by parity — this guards the kernel's CPU parity end to end;
        # the odd-d case where the fix is load-bearing has no GPU consumer
        # (sub-triangle hinges carry hingeArea() = 0), so it is pinned by
        # construction against Simplex::dihedralAngle, not a GPU run.
        st = _toroid4d()
        solver = tessera.ReggeSolver(st, tessera.MatterConfiguration())
        f_cpu = solver.actionGradientNorm()
        f_gpu = _gpu_step(solver)
        self.assertGreater(f_cpu, 0.0)
        self.assertTrue(
            math.isclose(f_gpu, f_cpu, rel_tol=5e-3),
            f"GPU step gradient norm {f_gpu} != CPU {f_cpu}")


if __name__ == "__main__":
    unittest.main()
