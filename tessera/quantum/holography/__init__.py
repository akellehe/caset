"""tessera.quantum.holography — emergent spectral dimension from a Schwinger TDVP state.

See ``docs/source/holography-causal-ordering-emergent-dimension.md`` for the
scientific charter, falsification criteria, and the relationship to the
causal-order comparison in ``tessera.quantum``.

The entire compute pipeline lives in the C++ extension; this module is a
thin re-export shim that surfaces the bindings under their natural import
path.

The four user-facing classes are:

* :class:`HolographyConfig`        — extends :class:`tessera.quantum.TDVPConfig`
                                     with the σ-grid and the mutual-information
                                     cutoff
* :class:`MutualInformationProfile`— symmetric MI matrix on the (site, snap)
                                     label set
* :class:`EmergentGraph`           — weighted Laplacian + heat-kernel trace +
                                     Graphviz export
* :class:`EmergentSpectralDimension` — workflow class: bind a HolographyConfig,
                                       call :meth:`compute` for the full result

Plus the data class :class:`SpectralDimensionResult` and the
:class:`AmbjornLollFit` static utility.

Quickstart
----------

>>> from tessera.quantum import TDVPConfig
>>> from tessera.quantum.holography import HolographyConfig, EmergentSpectralDimension
>>> cfg = HolographyConfig()
>>> cfg.tdvp = TDVPConfig()
>>> cfg.tdvp.N = 10; cfg.tdvp.a = 1.0; cfg.tdvp.g = 1.0
>>> cfg.tdvp.m = 0.5; cfg.tdvp.L0 = 0.0
>>> cfg.tdvp.dmrgMaxBondDim = 32; cfg.tdvp.dmrgNSweeps = 10
>>> cfg.tdvp.i0 = 3; cfg.tdvp.d = 3
>>> cfg.tdvp.dt = 0.2; cfg.tdvp.T = 0.4
>>> cfg.tdvp.snapshotEvery = 1
>>> cfg.tdvp.maxBondDim = 60
>>> result = EmergentSpectralDimension(cfg).compute()
>>> result.dInfinity > 0
True
"""

try:
    from tessera import _tessera
    _holo = _tessera.quantum.holography

    HolographyConfig            = _holo.HolographyConfig
    MutualInformationProfile    = _holo.MutualInformationProfile
    EmergentGraph               = _holo.EmergentGraph
    AmbjornLollFit              = _holo.AmbjornLollFit
    AmbjornLollFitResult        = _holo.AmbjornLollFitResult
    SpectralDimensionResult     = _holo.SpectralDimensionResult
    EmergentSpectralDimension   = _holo.EmergentSpectralDimension
    ChoiPropagator              = _holo.ChoiPropagator
    ChoiTDVPSettings            = _holo.ChoiTDVPSettings
    SchwingerParams             = _holo.SchwingerParams
except (ImportError, AttributeError) as exc:
    raise ImportError(
        "tessera.quantum.holography is unavailable: this tessera build does "
        "not include the quantum subsystem. Rebuild with TESSERA_QUANTUM=1 "
        "(e.g. `TESSERA_QUANTUM=1 pip install -e .`) to enable it."
    ) from exc

__all__ = [
    "HolographyConfig",
    "MutualInformationProfile",
    "EmergentGraph",
    "AmbjornLollFit",
    "AmbjornLollFitResult",
    "SpectralDimensionResult",
    "EmergentSpectralDimension",
    "ChoiPropagator",
    "ChoiTDVPSettings",
    "SchwingerParams",
]
