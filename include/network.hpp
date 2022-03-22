#pragma once
#include <optional>
#include <utility>
#include <vector>

#include "linear_algebra.hpp"

namespace nn {

using NNType = float;

// input and ground truth (one-hot) output
using Example = std::pair<Vector<NNType>, Vector<NNType>>;
using AnnotatedData = std::vector<Example>;
using Weights = std::vector<Matrix<NNType>>;
using Biases = std::vector<Vector<NNType>>;
using DeltaNablaBAndW = std::pair<Biases, Weights>;

unsigned int OneHotToIndex(Vector<NNType> one_hot_vector);
unsigned int GetMaxIndex(Vector<NNType> vector);

class Network {
public:
  Network(std::vector<unsigned int> layer_sizes);
  Vector<NNType> FeedForward(Vector<NNType> input);
  void Sgd(AnnotatedData training_data, unsigned int epochs,
           unsigned int mini_batch_size, NNType eta,
           std::optional<AnnotatedData> test_data = std::nullopt);
  void UpdateMiniBatch(AnnotatedData mini_batch, NNType eta);
  DeltaNablaBAndW Backprop(Example example);
  static Vector<NNType> CostDerivative(Vector<NNType> output,
                                       Vector<NNType> ground_truth);
  unsigned int Evaluate(AnnotatedData test_data);
  const std::vector<unsigned int> layer_sizes;
  const unsigned int num_layers;
  Weights weights;
  Biases biases;
};

} // namespace nn