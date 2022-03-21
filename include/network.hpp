#pragma once
#include <optional>
#include <utility>
#include <vector>

#include "linear_algebra.hpp"

namespace nn {

using NNType = float;
using AnnotatedData =
    std::pair<std::vector<Vector<NNType>>, std::vector<Vector<NNType>>>;

class Network {
public:
  Network(std::vector<unsigned int> layer_sizes);
  Vector<NNType> FeedForward(Vector<NNType> input);
  void Sgd(AnnotatedData training_data, unsigned int epochs,
           unsigned int mini_batch_size, NNType eta,
           std::optional<AnnotatedData> test_data = std::nullopt);
  const std::vector<unsigned int> layer_sizes;
  const unsigned int num_layers;
  std::vector<Matrix<NNType>> weights;
  std::vector<Vector<NNType>> biases;
};

} // namespace nn