#include "spacetime/topologies/Topology.h"
#include <vector>
#include <memory>
#include <iostream>

namespace caset {
Topology::~Topology() = default;
// std::vector<std::shared_ptr<Constraint> > Topology::getConstraints() {return {};}
void Topology::build(Spacetime *spacetime) {
  std::cout << "Building Topology (base)" << std::endl;
}
} // namespace caset
