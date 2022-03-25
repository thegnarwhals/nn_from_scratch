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

/**
 * @brief      Elementwise ReLU of a vector
 *
 * @param[in]  input  The input vector
 *
 * @tparam     T      Data type
 *
 * @return     The ReLU of the elements of the input vector (max(x, 0))
 */
template <typename T> Vector<T> Relu(Vector<T> input) {
  Vector<T> output(input.length);
  for (unsigned int i = 0; i < input.length; i++) {
    output.elements[i] = input.elements[i] > 0 ? input.elements[i] : 0;
  }
  return output;
}

/**
 * @brief      Elementwise ReLU gradient of a vector
 *
 * @param[in]  input  The input vector
 *
 * @tparam     T      Data type
 *
 * @return     The ReLU gradient of the elements of the input vector (0 if
 * negative, 1 if positive)
 */
template <typename T> Vector<T> ReluPrime(Vector<T> input) {
  Vector<T> output(input.length);
  for (unsigned int i = 0; i < input.length; i++) {
    output.elements[i] = input.elements[i] > 0 ? 1 : 0;
  }
  return output;
}
} // namespace nn