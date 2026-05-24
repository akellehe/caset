# Contributing to tessera

Project conventions captured here as the codebase's first contributor
doc. Established alongside the issue #56 refactor; expected to evolve.

## Naming

- **Functions and member variables**: `camelCase`.
- **Types** (classes, structs, enums, type aliases): `PascalCase`.
- **Private member variables**: trailing underscore (`leafSimplices_`,
  `iMax_`, `frontier_`).
- **Opt-in config flags**: `feature*` prefix (`featureCharges`,
  `featureQuditBasis`). Default to `false` so old runs stay reproducible.
- **Numerical tolerances**: `eps*` prefix (`epsLocalPure`, `epsKiEigen`,
  `epsKiCondState`, `epsKiSvd`). Reviewers can audit every approximation
  by grepping for `eps`.

## Layout

- `include/<area>/<Class>.h` (or `.hpp` in the quantum subsystem) is
  mirrored by `src/<area>/<Class>.cpp`.
- Areas: `mesh`, `spacetime`, `simulations`, `quantum`, `observables`,
  `graph`.
- Tests under `tests/<area>/`. The `tests/quantum/` directory builds
  only when `TESSERA_QUANTUM=1` (needs ITensor). The
  `tests/quantum_core/` directory builds always (Eigen only, no
  ITensor) — use it for tests that only need `tessera_core` symbols.

## C++ standard and dependencies

- C++20 throughout (set by CMake).
- Eigen is a `tessera_core` dependency (needed by `QuantumState` and
  `KoashiImoto`).
- ITensor is required for the optional quantum subsystem
  (`TESSERA_QUANTUM=1` or auto-detected when
  `third_party/itensor/itensor/itensor.h` exists).

## Quantum conventions

- Density matrices are `Eigen::MatrixXcd` with **runtime dimension**.
  Avoid baking in `Matrix2cd` / `Matrix4cd` — vertices in the
  post-#56 design carry states of varying dimension.
- Entropies are in **nats** (base e), not bits. `log` everywhere
  means `std::log` (natural log).
- Mutual information is computed via
  `tessera::quantum::mutualInformation(rhoAB, dimA, dimB)`. Always
  floors at 0 (numerical noise can produce small negatives on
  near-product inputs).

## Python bindings

- pybind11 v3 throughout.
- Each subsystem has a `src/<area>/Bindings.cpp` with a single
  `register_<area>(py::module&)` function called from
  `src/bindings.cpp`.
- Bindings are excluded from `tessera_core` (they pull in Python
  headers); they're compiled directly into the `_tessera` module.

## Memory discipline

- Anything held by raw pointer (`VertexPtr`, `EdgePtr`, `SimplexPtr`)
  must live at a **stable address**. The recent
  `dba4d2e` ("Spacetime: stable-address simplex storage to fix
  concurrent-sweep UAF") is the canonical example: backing storage
  reshuffling under iteration is a real bug class here.
- The `liveIdx_` indirection pattern (in `Vertex`, `Edge`, `Simplex`)
  is how we get O(1) iteration over a compact "live" view without
  invalidating pointers in the underlying storage. Use it when
  introducing similar containers.
- AddressSanitizer / UBSan runs are first-class:
  `TESSERA_ASAN=1 pip install -e .` builds an ASAN binary;
  `TESSERA_QUANTUM=1 TESSERA_ASAN=1 pytest tests/quantum/` should be
  clean on any PR that touches the simulation core.
- `TESSERA_ASSERTIONS=1` enables aggressive runtime tripwires (null
  pointer checks, duplicate-fingerprint checks, etc.). Use during
  development; the production build is `TESSERA_ASSERTIONS=0`.

## Comments

WHY, not WHAT. Default to no comments — well-named identifiers carry
most of the meaning. Add a comment when:

- The reason for a choice is non-obvious (a hidden constraint, a
  subtle invariant, a workaround for a specific bug, behaviour that
  would surprise a careful reader).
- A piece of code references a removed predecessor (the issue #56
  refactor uses `// REMOVED in #56` breadcrumbs liberally so
  git-bisect through the rewrite stays navigable).
- Math: inline LaTeX in comments is encouraged where the algorithm
  is the math (KoashiImoto.cpp does this throughout).

Avoid: explaining WHAT the code does in prose; trailing summaries of
what was just edited; "added for issue #X" pointers that rot.

## Tests

- Mixed C++ (custom `expect_*` helpers — no gtest in core) and Python
  (`unittest` / pytest).
- C++ tests print `ALL PASSED` on success; the runner returns 0/1
  accordingly. The Python wrapper
  (`tests/quantum_core/test_run_core_cpp.py`,
  `tests/quantum/test_quantum_executables.py`) re-runs the C++
  binaries inside pytest so a single `pytest tests` invocation
  exercises both layers.
- New C++ tests in `tests/quantum_core/` link only `tessera_core`
  and build regardless of `TESSERA_QUANTUM`. Use this when a test
  doesn't need ITensor.
- Hand-calculated expected values are the gold standard. When the
  expected output is computable on paper (Schmidt decomposition of a
  Bell state, mutual information of a product, etc.), put the number
  in the assertion and quote the derivation in a nearby comment so
  future readers can re-check.

## Approximations

Every tolerance-driven approximation is named with an `eps*` config
field whose default is documented at the use site. The codebase has
no silent shortcuts — pure / product / Bell / classical inputs to
`koashiImotoDecompose` fall out of the same general algorithm rather
than hitting `if`/`else` special cases. When you add a new
approximation, name it and document its trade-off (too small → false
distinction; too large → real structure collapses).

## Build / test setup

The recommended workflow uses a virtualenv with editable install:

```
python -m pip install -e ".[dev]"
TESSERA_QUANTUM=1 pytest tests/quantum/
TESSERA_QUANTUM=1 TESSERA_ASAN=1 pytest tests/quantum/   # clean = no UAF / leaks
```

For C++ iteration without re-running pip, the conftest auto-builds via
scikit-build-core. For a manual build:

```
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug -DTESSERA_QUANTUM=1
cmake --build build -j
./build/test_quantum_state_core      # tests/quantum_core/ binaries
./build/test_koashi_imoto_core
```

## Committing

- Prefer focused commits with a clear "what" in the subject and the
  "why" in the body.
- For refactors that remove an API, leave a one-line breadcrumb at
  the removal site (`// REMOVED in #<issue>: <one-line summary>`).
  git-blame plus the breadcrumb tells future maintainers exactly
  what was there and why it left.
- Do not add `Co-Authored-By` trailers.
