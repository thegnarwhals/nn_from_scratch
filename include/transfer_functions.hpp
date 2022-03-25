#pragma once
#include <cmath>

namespace nn {

/**
 * @brief      Elementwise sigmoid of a vector
 *
 * @param[in]  input  The input vector
 *
 * @tparam     T      Data type
 *
 * @return     The sigmoid of the elements of the input vector
 */
template <typename T> Vector<T> Sigmoid(Vector<T> input) {
  Vector<T> output(input.length);
  for (unsigned int i = 0; i < input.length; i++) {
    output.elements[i] =
        static_cast<T>(1) / (static_cast<T>(1) + std::exp(-input.elements[i]));
  }
  return output;
}

/**
 * @brief      Elementwise sigmoid derivative of a vector
 *
 * @param[in]  input  The input vector
 *
 * @tparam     T      Data type
 *
 * @return     The sigmoid derivative of the elements of the input vector
 */
template <typename T> Vector<T> SigmoidPrime(Vector<T> input) {
  return Sigmoid(input) * (static_cast<T>(1) - Sigmoid(input));
}
} // namespace nn