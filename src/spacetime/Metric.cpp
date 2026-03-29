// MIT License
// Copyright (c) 2025 Andrew Kelleher
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//
// Created by andrew on 10/23/25.
//

#include "spacetime/Metric.h"

#include <memory>

#include "spacetime/Signature.h"
#include "Logger.h"

namespace caset {

    Metric::Metric(bool coordinateFree_, const Signature &signature_) : signature(std::make_shared<Signature>(signature_)), coordinateFree(coordinateFree_) {
    }

    [[nodiscard]] double Metric::getSquaredLength(
      const std::vector<double> &sourceCoords,
      const std::vector<double> &targetCoords
      ) const {

      if (coordinateFree) {
        CLOG(ERROR_LEVEL, "You asked a coordinate free metric to compute the squared length of an edge. That data should be store directly on the edge already.");
        throw std::runtime_error("You asked a coordinate free metric to compute the squared length of an edge. That data should be store directly on the edge already.");
      }

      auto diag = signature->getDiagonal();
      double lengthSquared = 0.0;
      for (int i = 0; i < diag.size(); ++i) {
        double delta = sourceCoords[i] - targetCoords[i];
        lengthSquared += static_cast<double>(diag[i]) * delta * delta;
      }
      return lengthSquared;
    }

    [[nodiscard]] const std::shared_ptr<Signature> &Metric::getSignature() const noexcept {
      return signature;
    }

};
