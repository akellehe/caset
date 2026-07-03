// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/EmergentRadius.h"

namespace tessera::observables {

InteriorHinges::Radii EmergentRadius::radii(const RegisterContext &ctx) const {
  return ctx.interiorHinges()->radii();
}

double EmergentRadius::computeHeadline(const RegisterContext &ctx) const {
  return ctx.interiorHinges()->radii().rDual;
}

Record EmergentRadius::record(const RegisterContext &ctx) const {
  const InteriorHinges::Radii rad = ctx.interiorHinges()->radii();
  Record::Map radius;
  radius["Vdual"] = rad.vDual;
  radius["Vprimal"] = rad.vPrimal;
  radius["n_interior_vertices"] = rad.nInteriorVertices;
  radius["r_dual"] = rad.rDual;
  radius["r_primal"] = rad.rPrimal;

  Record::Map m;
  m["radius"] = std::move(radius);
  m["n_holes"] = ctx.holesUsed();
  return Record(std::move(m));
}

}  // namespace tessera::observables
