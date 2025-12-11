//
// Created by Andrew Kelleher on 12/11/25.
//
#include <gtest/gtest.h>
#include "Simplex.h"

TEST(SimplexTest, BasicEdges) {
  caset::VertexPtrs vertices = { /* ... */ };
  caset::Edges edges = { /* ... */ };

  caset::Simplex s(vertices, edges);
  ASSERT_EQ(s.size(), vertices.size());
}