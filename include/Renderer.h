// MIT License -- Copyright (c) 2025 Andrew Kelleher
#pragma once

#include <string>

namespace caset {

class Spacetime;

/// Render the spacetime to a BMP image with four orientation panels
/// (no rotation, 40 deg X, 40 deg Y, 40 deg Z).
///
/// Uses a force-directed layout: time coordinate fixed per slice,
/// spatial coordinates optimized via spring + repulsion forces.
/// The layout is computed internally and does not modify vertex state.
void renderSpacetime(const Spacetime &st, const std::string &path,
                     int panelSize = 800, int layoutIters = 500);

} // namespace caset
