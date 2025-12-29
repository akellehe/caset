# Topology

# Parity

This method should be used when you have two simplices and you want to find some Vertices at which to join one
Simplex to the other. Two sets of ordered vertices, \f$ \mathcal{V_i} \f$ and \f$ \mathcal{V_j} \f$ "have parity"
when there is an ordered subset, \f$ V_i \f$ and \f$ V_j \f$ for which Vertex::getTime returns the same value
for each corresponding element. More specifically;

Two sets of vertices "have parity" iff
  - They have the same cardinality, \f$ N \f$
  - There exists an element \f$ v_m \memberof \f$ V_i \f$ for which there exists an an element
    \f$ v_n \memberof V_j \f$ such that \f$ v_m::getTime() = v_n::getTime() \forall (m, n) s.t. m = n \f$.
