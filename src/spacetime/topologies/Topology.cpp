#include "spacetime/topologies/Topology.h"
#include <vector>
#include <memory>

namespace caset {
Topology::~Topology() = default;
std::vector<std::shared_ptr<Constraint> > Topology::getConstraints() {return {};}
void Topology::build(Spacetime *spacetime) {return;};
} // namespace caset
