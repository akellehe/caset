# The Tessera C++ API

Tessera is a package for simulating lattice spacetime interactions building on the notion of causal sets as well as causal
simplicial complexes.

Below the C++ API is documented for users. To document for developers you can edit the Doxyfile and comment out
exclusions to see the entire interface.

<!--
  MAINTENANCE / DURABILITY
  ------------------------
  This page must list *every* public header under ``include/`` via a Breathe
  ``{doxygenfile}`` directive, grouped into the per-module sections below.
  The sections mirror the ``include/`` directory tree one-to-one.

  Doxygen's INPUT (see ../Doxyfile: ``INPUT = ../src ../include`` with
  ``RECURSIVE = YES``) already covers the entire tree, so every header has a
  Breathe-addressable file compound — entries here only need the header's
  *basename* (there are no duplicate header basenames in the tree).

  When you add a new header under ``include/``, add a matching
  ``{doxygenfile}`` entry to the correct section here. The guard test
  ``tests/test_cpp_api_docs_coverage.py`` fails CI if a header is missing
  from (or stale in) this page, so this list cannot silently rot.
-->

## Core Spacetime

```{doxygenfile} Spacetime.h
```
```{doxygenfile} Metric.h
```
```{doxygenfile} Signature.h
```
```{doxygenfile} Foliation.h
```

## Simplicial Complex

```{doxygenfile} Simplex.h
```
```{doxygenfile} TemporalOrientation.h
```
```{doxygenfile} Vertex.h
```
```{doxygenfile} VertexList.h
```
```{doxygenfile} Edge.h
```
```{doxygenfile} EdgeKey.h
```
```{doxygenfile} EdgeList.h
```
```{doxygenfile} Fingerprint.h
```
```{doxygenfile} SimplexFilter.h
```
```{doxygenfile} FlatHashMap.h
```
```{doxygenfile} ForwardDeclarations.h
```

## Pachner Moves

```{doxygenfile} PachnerMove.h
```
```{doxygenfile} AddMove.h
```
```{doxygenfile} RemoveMove.h
```
```{doxygenfile} FlipMove.h
```
```{doxygenfile} IFlipMove.h
```
```{doxygenfile} ShiftMove.h
```

## Topologies

```{doxygenfile} Topology.h
```
```{doxygenfile} Toroid.h
```
```{doxygenfile} Cylinder.h
```
```{doxygenfile} Sphere.h
```
```{doxygenfile} SimplicialProduct.h
```
```{doxygenfile} SphereCircleProduct.h
```
```{doxygenfile} SimplexBoundarySphere.h
```
```{doxygenfile} SolidSimplex.h
```
```{doxygenfile} StellarSubdivision.h
```
```{doxygenfile} RealProjectivePlane.h
```
```{doxygenfile} RealProjectiveSpace.h
```
```{doxygenfile} ComplexProjectivePlane.h
```

## Simulations

```{doxygenfile} Simulation.h
```
```{doxygenfile} CDT.h
```
```{doxygenfile} ReggeSolver.h
```
```{doxygenfile} InteractionSimulation.h
```

## Observables

```{doxygenfile} Observable.h
```
```{doxygenfile} VolumeProfile.h
```
```{doxygenfile} SpacetimeVolume.h
```
```{doxygenfile} Spectral.h
```
```{doxygenfile} WilsonLoop.h
```
```{doxygenfile} MIUnits.hpp
```
```{doxygenfile} ModularityOptimizer.h
```
```{doxygenfile} SparseGraph.h
```

## Cobordism

```{doxygenfile} Cobordism.h
```
```{doxygenfile} ChainComplex.h
```
```{doxygenfile} HodgeLaplacian.h
```
```{doxygenfile} DiracKahler.h
```
```{doxygenfile} IntegerLinalg.h
```
```{doxygenfile} DijkgraafWitten.h
```
```{doxygenfile} EigenstateSynthesis.h
```
```{doxygenfile} Characteristic.h
```
```{doxygenfile} CombinatorialDimension.h
```
```{doxygenfile} Cochain.h
```
```{doxygenfile} Spectrum.h
```
```{doxygenfile} LevenbergMarquardt.h
```
```{doxygenfile} GeometrySynthesizer.h
```
```{doxygenfile} RealizabilityOracle.h
```
```{doxygenfile} BoundaryStateSpace.h
```
```{doxygenfile} PreparedBoundaryState.h
```
```{doxygenfile} Register.h
```
```{doxygenfile} MergeCobordism.h
```
```{doxygenfile} TransportCobordism.h
```
```{doxygenfile} CobordismRelaxer.h
```
```{doxygenfile} TopologyBuilder.h
```
```{doxygenfile} TorusOperatorTopology.h
```
```{doxygenfile} RegisterTopology.h
```
```{doxygenfile} TripartiteRegisterTopology.h
```

## Quantum

```{doxygenfile} SchwingerModel.hpp
```
```{doxygenfile} DMRGRunner.hpp
```
```{doxygenfile} TDVPRunner.hpp
```
```{doxygenfile} TDVPIntegrator.hpp
```
```{doxygenfile} Quench.hpp
```
```{doxygenfile} ChoiJamiolkowski.h
```
```{doxygenfile} ChoiState.hpp
```
```{doxygenfile} Holography.hpp
```
```{doxygenfile} MutualInformation.hpp
```
```{doxygenfile} Majorization.hpp
```
```{doxygenfile} KoashiImoto.hpp
```
```{doxygenfile} Schmidt.hpp
```
```{doxygenfile} CausalCompare.hpp
```
```{doxygenfile} CausetChain.hpp
```
```{doxygenfile} QuantumSimplex.hpp
```
```{doxygenfile} QuantumVertex.hpp
```

## Graph

```{doxygenfile} DualGraph.hpp
```
```{doxygenfile} SpectralGraph.hpp
```
```{doxygenfile} CSRBuilder.hpp
```
```{doxygenfile} COO.hpp
```
```{doxygenfile} IndexByKey.hpp
```

## Matter

```{doxygenfile} MatterConfiguration.h
```

## Constraints

```{doxygenfile} Constraint.h
```

## GPU / CUDA acceleration

```{doxygenfile} regge_cuda.h
```
```{doxygenfile} eigenstate_cuda.h
```

## Core utilities & infrastructure

```{doxygenfile} Poset.h
```
```{doxygenfile} Renderer.h
```
```{doxygenfile} ForceLayout.h
```
```{doxygenfile} Logger.h
```
```{doxygenfile} utils.h
```
