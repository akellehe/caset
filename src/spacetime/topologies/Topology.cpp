#include "spacetime/topologies/Topology.h"
#include <vector>
#include <memory>
#include <iostream>

namespace caset {
Topology::~Topology() = default;
// std::vector<std::shared_ptr<Constraint> > Topology::getConstraints() {return {};}
void Topology::build() {
  std::cout << "Building Topology (base)" << std::endl;
} // Spacetime *spacetime) {return;};
} // namespace caset
