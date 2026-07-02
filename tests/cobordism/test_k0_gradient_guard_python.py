"""The analytic r_U / r_psi gradient cores refuse degree k = 0 (#581 item 7).

The cores project the metric Hodge operator with ``laplacian(k).real()`` —
lossless for k >= 1 (the metric L_k is real symmetric there) but silently the
WRONG operator at k = 0, where L_0 consumes the full complex l^2 (#580).
Until a complex k = 0 core exists they fail loudly; k = 1 keeps working.
"""

import os
import sys

import numpy as np
import pytest

import tessera

cob = tessera.cobordism

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from _holed_surface import holed_surface  # noqa: E402


def test_k0_synthesis_gradient_cores_throw():
    st, _es1, holes, P = holed_surface(degree=1)
    es0 = cob.EigenstateSynthesis(st)  # default degree k = 0
    target = [complex(z) for z in P[0]]
    with pytest.raises(RuntimeError, match="580"):
        es0.residualForPeriodsGradient(holes, target)
    with pytest.raises(RuntimeError, match="k = 0"):
        es0.periodGapForPeriodsGradient(holes, target)


def test_k1_gradients_still_work():
    st, es, holes, P = holed_surface(degree=1)
    target = [complex(z) for z in P[0]]
    g = np.asarray(es.residualForPeriodsGradient(holes, target))
    assert g.shape[0] > 0 and np.all(np.isfinite(g))
    g2 = np.asarray(es.periodGapForPeriodsGradient(holes, target))
    assert g2.shape[0] > 0 and np.all(np.isfinite(g2))
