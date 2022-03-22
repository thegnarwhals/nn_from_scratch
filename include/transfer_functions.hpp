#pragma once
#include <cmath>

namespace nn {

template <typename T> Vector<T> Sigmoid(Vector<T> input) {
  Vector<T> output(input.length);
  for (unsigned int i = 0; i < input.length; i++) {
    output.elements[i] =
        static_cast<T>(1) / (static_cast<T>(1) + std::exp(-input.elements[i]));
  }
  return output;
}

template <typename T> Vector<T> SigmoidPrime(Vector<T> input) {
  return Sigmoid(input) * (1 - Sigmoid(input));
}
} // namespace nn