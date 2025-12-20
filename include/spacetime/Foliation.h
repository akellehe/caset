//
// Created by andrew on 12/20/25.
//

#ifndef CASET_FOLIATION_H
#define CASET_FOLIATION_H
#include <cstdint>

namespace caset {

/// # Foliation
///
/// When someone says "foliation" in the context of lattice models of gravity; they're generally referring to an
/// arrangment by which spatial slices of the lattice manifold are sandwiched between temporal slices. This is called a
/// "preferred" foliation, and is a means of enforcing causal structure. Without preferred foliation there's no
/// requirement between shapes of simplices that can be glued together.
enum class Foliation : std::uint8_t {
  NONE = 0,
  PREFERRED = 1
};
} // caset

#endif //CASET_FOLIATION_H