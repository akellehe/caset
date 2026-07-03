// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/RegisterObservable.h"

#include <string>

namespace tessera::observables {

std::string RegisterObservable::skipReason(const RegisterContext &ctx) const {
  if (needsProvenance() && !hasProvenance()) {
    return std::string(kSkipNoProvenance);
  }
  if (ctx.holesUsed() < minHoles()) {
    return std::string(kSkipHolesPrefix) + std::to_string(ctx.holesUsed()) +
           std::string(kSkipHolesInfix) + std::to_string(minHoles());
  }
  if (requiredDimensions() >= 0 && ctx.dimensions() != requiredDimensions()) {
    return std::string(kSkipDimensionsPrefix) +
           std::to_string(ctx.dimensions()) + std::string(kSkipDimensionsInfix) +
           std::to_string(requiredDimensions());
  }
  if (needsCausalContent() && !ctx.causalContent()) {
    return std::string(kSkipNoCausalContent);
  }
  return "";
}

double RegisterObservable::compute(
    const std::shared_ptr<Spacetime> &spacetime) {
  // Read a default 3-hole register off the live complex and return the headline.
  RegisterContext ctx(spacetime);
  return computeHeadline(ctx);
}

double RegisterObservable::update(const std::shared_ptr<Spacetime> &spacetime) {
  return compute(spacetime);
}

}  // namespace tessera::observables
