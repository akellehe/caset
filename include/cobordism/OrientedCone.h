// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_ORIENTEDCONE_H
#define TESSERA_COBORDISM_ORIENTEDCONE_H

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tessera::spacetime {
class Spacetime;
class PachnerMove;
}  // namespace tessera::spacetime

namespace tessera::cobordism {
using namespace ::tessera::spacetime;

/// # OrientedCone
///
/// An **orientation-safe, `dualComplexValid`-gated** stellar cone move — the
/// safety net the Emergent Color Topology epic's surgical `b₂`-growth (#457)
/// relies on. It is a thin wrapper that *reuses* the hinge-exact T1 cone
/// primitives (`spacetime::AddMove` / `spacetime::RemoveMove` in
/// `PachnerMode::PreGeometric`, the \f$ 1 \leftrightarrow (d+1) \f$ stellar
/// subdivision) rather than reimplementing them, and accepts a move **only if**
/// the resulting complex survives the full manifold check.
///
/// ## Why this is needed
/// A stellar cone builds \f$ d+1 \f$ fresh top cells. Their reference
/// orientation comes from `TemporalOrientation::orientationOf` (the
/// Ambjørn–Loll CDT convention, computed canonically from vertex times — never
/// an ad-hoc vertex-label sort), and the Lorentzian dihedral deficit is
/// evaluated in the canonical sorted-by-id frame (T1 fix #3), so for a *valid*
/// refinement the complex (causal/oriented) deficit — and hence
/// \f$ \operatorname{Im} S \f$ in `ReggeSolver::dualReggeAction` — is a true
/// relabelling invariant. The danger is a cone that quietly produces a
/// **non-manifold** or **non-orientable** local configuration: there the
/// induced orientation can flip, injecting a spurious sign into
/// \f$ \operatorname{Im} S \f$ (real physics). This class makes that
/// impossible by construction.
///
/// ## The gate
/// After applying the underlying move, the candidate complex is accepted only
/// when **both** hold:
///   1. `ChainComplex::dualComplexIsValid` — a genuine manifold: facet coface
///      counts in \f$ \{1,2\} \f$, ridge links single paths/cycles, and (the
///      #429 check) for \f$ n \geq 4 \f$ a recursive validation of every vertex
///      link as an \f$ (n-1) \f$-manifold;
///   2. `ChainComplex::orientationCovector` returns without contradiction — a
///      consistent global induced orientation exists (orientable).
/// On rejection the move is rolled back and the complex is left bit-identical
/// to its pre-cone state, so a caller can treat a rejected cone as a no-op. For
/// a *topology-preserving* refinement (this ticket, T2) the gate always passes;
/// it is the live guard the *topology-changing* surgical variant (T3) leans on.
///
/// ## Lifecycle
/// At most one cone is held applied at a time. `coneIn` / `coneOut` apply and
/// gate; `rollback` undoes the last accepted cone (delegating to the T1 move's
/// exact inverse). The move is owned so the round trip \f$ m \circ m^{-1} \f$
/// restores the complex — and every orientation sign — exactly.
class OrientedCone {
 public:
  /// Bind the cone to a spacetime. Does not mutate it.
  explicit OrientedCone(Spacetime *spacetime);
  ~OrientedCone();

  OrientedCone(const OrientedCone &) = delete;
  OrientedCone &operator=(const OrientedCone &) = delete;

  /// Gated stellar \f$ 1 \to (d+1) \f$ refinement (**cone-in**): pick a top
  /// cell (seeded), subdivide it with the T1 `AddMove(PreGeometric)`, then
  /// accept only if the result is a valid, orientable manifold. Returns
  /// `(true, "ok")` on acceptance; otherwise the move is rolled back and the
  /// reason is returned. A no-op `(false, …)` if a cone is already applied.
  std::pair<bool, std::string> coneIn(std::uint64_t seed);

  /// Gated stellar \f$ (d+1) \to 1 \f$ weld (**cone-out**): the inverse
  /// refinement, via the T1 `RemoveMove(PreGeometric)`, under the same gate.
  std::pair<bool, std::string> coneOut(std::uint64_t seed);

  /// Undo the last accepted cone (delegating to the T1 move's exact inverse),
  /// restoring the complex bit-for-bit. Returns `false` if nothing is applied.
  bool rollback();

  /// True iff a cone has been accepted and not yet rolled back.
  [[nodiscard]] bool isApplied() const { return applied_; }

  /// The induced-orientation covector of the **current** top complex
  /// (`ChainComplex::orientationCovector` over its top cells), aligned to the
  /// canonical sorted-unique top-cell order. The read-out tests assert is
  /// restored exactly across a cone round trip.
  /// @throws std::runtime_error if the current complex is non-orientable.
  [[nodiscard]] std::vector<int> orientationCovector() const;

  /// The manifold/orientability verdict on the **current** complex, the same
  /// gate `coneIn` / `coneOut` apply. `(true, "ok")` when it is a valid,
  /// orientable manifold; otherwise the first violation is named.
  [[nodiscard]] std::pair<bool, std::string> validate() const;

 private:
  /// Top cells of the bound spacetime as sorted vertex-id tuples (canonical
  /// C_d order), the input to both the manifold and the orientation checks.
  [[nodiscard]] std::vector<std::vector<std::uint64_t>> topCells() const;

  Spacetime *st_;
  std::unique_ptr<PachnerMove> lastMove_;
  bool applied_ = false;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_ORIENTEDCONE_H
