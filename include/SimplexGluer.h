//
// Created by andrew on 12/16/25.
//

#ifndef CASET_SIMPLEXGLUER_H
#define CASET_SIMPLEXGLUER_H

#include "ForwardDeclarations.h"

#include <vector>

namespace caset {
class SimplexGluer {
  public:
    SimplexGluer(Spacetime *spacetime, const VertexPtrs &unattachedVertices_);
    ~SimplexGluer();
    SimplexGluer(const SimplexGluer &) = delete;
    SimplexGluer &operator=(const SimplexGluer &) = delete;

    const Simplices &getSimplicesToProcess() const noexcept;
  private:
    Spacetime *spacetime = nullptr;
    VertexPtrs unattachedVertices{};
    SimplexPtrSet simplicesToProcessAsSet{};
    Simplices simplicesToProcess{};
    std::vector<std::tuple<SimplexPtr, Simplices, Simplices>> brokenReferences{}; // simplex, cofaces, facets

    void stageVertex(const VertexPtr &unattached);
    void breakReferences();
};
} // caset

#endif //CASET_SIMPLEXGLUER_H