#include "Simplex.h"

namespace caset {

std::unordered_set<std::shared_ptr<Face>, FaceHash, FaceEq> Simplex::facetRegistry = {};
}
