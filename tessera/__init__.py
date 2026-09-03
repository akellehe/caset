"""tessera -- Causal Set and CDT simulation library.

The heavy lifting lives in the C++ extension ``_tessera``. Classes are
organised into submodules whose names match their C++ namespaces:

* ``tessera.mesh``         — Vertex, Edge, Simplex, SimplexFilter, IDs
* ``tessera.spacetime``    — Spacetime, Metric, Signature, topologies, Pachner moves
* ``tessera.observables``  — SparseGraph, ModularityOptimizer, WilsonLoop, ...
* ``tessera.simulations``  — CDT, ReggeSolver, Simulation base
* ``tessera.quantum``      — Schwinger model, DMRG, TDVP, holography, InteractionSimulation

For backward compatibility every public class is also re-exported at the
top level, so ``from tessera import Spacetime`` continues to work alongside
the canonical ``from tessera.spacetime import Spacetime``.
"""

# Root-namespace classes / free functions (Poset, OrderAgreement,
# MatterConfiguration, HingeType, renderSpacetime, ForceLayout, ...).
from tessera._tessera import *                              # noqa: F401,F403
from tessera._tessera import __doc__                        # noqa: F401

# Subsystem submodules — make `tessera.mesh.Vertex` etc. importable.
from tessera._tessera import (                              # noqa: F401
    mesh,
    spacetime,
    observables,
    simulations,
    cobordism,
    chainhodge,
)

# The ``quantum`` submodule (Schwinger model / DMRG, ITensor-backed) is
# always built — ITensor/Eigen/BLAS are unconditional dependencies.
from tessera._tessera import quantum                        # noqa: F401

# Backward-compat re-exports at top level. Star-import each submodule so
# existing scripts that do `from tessera import Spacetime`, `tessera.CDT`,
# etc. continue to work.
from tessera._tessera.mesh        import *                  # noqa: F401,F403
from tessera._tessera.spacetime   import *                  # noqa: F401,F403
from tessera._tessera.observables import *                  # noqa: F401,F403
from tessera._tessera.simulations import *                  # noqa: F401,F403
# NB: cobordism is intentionally NOT star-imported to the top level. It is a
# specialized subsystem (no backward-compat scripts) and some of its names
# would shadow core ones — e.g. cobordism.Signature vs the metric
# spacetime.Signature. Access it as ``tessera.cobordism.*``.
# Quantum is also subsystem-namespaced; expose at top level for symmetry
# with the others (existing scripts already use `tessera.quantum.*`).
