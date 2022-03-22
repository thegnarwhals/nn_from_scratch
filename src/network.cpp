#include "network.hpp"

#include <algorithm>
#include <chrono>
#include <vector>

#include "linear_algebra.hpp"
#include "transfer_functions.hpp"

namespace nn {

unsigned int OneHotToIndex(Vector<NNType> vector) {
  const auto result_it =
      std::find(std::begin(vector.elements), std::end(vector.elements),
                static_cast<NNType>(1));
  assert(result_it != std::end(vector.elements));
  const auto result_idx = std::distance(std::begin(vector.elements), result_it);
  return result_idx;
}

unsigned int GetMaxIndex(Vector<NNType> vector) {
  const auto result_it =
      std::max_element(std::begin(vector.elements), std::end(vector.elements));
  const auto result_idx = std::distance(std::begin(vector.elements), result_it);
  return result_idx;
}

Network::Network(std::vector<unsigned int> layer_sizes)
    : layer_sizes(layer_sizes), num_layers(layer_sizes.size()) {
  // Random initialisation of weights and biases
  const NNType mean = 0.f;
  const NNType stddev = 1.f;
  for (unsigned int i = 1; i < num_layers; i++) {
    biases.push_back(Vector<NNType>::Random(layer_sizes[i], mean, stddev));
  }
  for (unsigned int i = 1; i < num_layers; i++) {
    weights.push_back(Matrix<NNType>::Random(layer_sizes[i], layer_sizes[i - 1],
                                             mean, stddev));
  }
}

Vector<NNType> Network::FeedForward(Vector<NNType> input) {
  std::vector<Vector<NNType>> layer_outputs({input});
  for (unsigned int i = 1; i < num_layers; i++) {
    std::cout << "Calculating layer " << i + 1 << " of " << num_layers
              << std::endl;
    layer_outputs.push_back(
        Sigmoid(weights[i - 1] * layer_outputs.back() + biases[i - 1]));
  }
  return layer_outputs.back();
}

void Network::Sgd(AnnotatedData training_data, unsigned int epochs,
                  unsigned int mini_batch_size, NNType eta,
                  std::optional<AnnotatedData> test_data) {
  // Train the neural network using mini-batch stochastic
  // gradient descent.  The "training_data" is a list of pairs
  // "(x, y)" representing the training inputs and the desired
  // outputs.  The other non-optional parameters are
  // self-explanatory.  If "test_data" is provided then the
  // network will be evaluated against the test data after each
  // epoch, and partial progress printed out.  This is useful for
  // tracking progress, but slows things down substantially.
  const unsigned int n_test = test_data ? test_data->size() : 0;
  std::cout << "n_test: " << n_test << std::endl;

  if (test_data) {
    std::cout << "Initial evaluation: " << Evaluate(test_data.value()) << " / "
              << n_test << std::endl;
  }
  const unsigned int n_training = training_data.size();
  std::cout << "n_training: " << n_training << std::endl;
  for (unsigned int epoch_idx = 0; epoch_idx < epochs; epoch_idx++) {
    // obtain a time-based seed:
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(training_data.begin(), training_data.end(),
                 std::default_random_engine(seed));
    assert(n_training % mini_batch_size == 0);
    const unsigned int n_mini_batches = n_training / mini_batch_size;
    std::vector<AnnotatedData> mini_batches;
    for (unsigned int mini_batch_idx = 0; mini_batch_idx < n_mini_batches;
         mini_batch_idx++) {
      const unsigned int start_idx = mini_batch_idx * mini_batch_size;
      const unsigned int end_idx = (mini_batch_idx + 1) * mini_batch_size;
      mini_batches.push_back(AnnotatedData(training_data.begin() + start_idx,
                                           training_data.begin() + end_idx));
    }
    for (const auto &mini_batch : mini_batches) {
      UpdateMiniBatch(mini_batch, eta);
    }
    if (test_data) {
      std::cout << "Epoch " << epoch_idx << ": " << Evaluate(test_data.value())
                << " / " << n_test << std::endl;
    } else {
      std::cout << "Epoch " << epoch_idx << " complete" << std::endl;
    }
  }
}

void Network::UpdateMiniBatch(AnnotatedData mini_batch, NNType eta) {
  Biases nabla_b;
  for (const auto &layer_biases : biases) {
    nabla_b.push_back(Vector<NNType>::Zeros(layer_biases.length));
  }
  Weights nabla_w;
  for (const auto &layer_weights : weights) {
    nabla_w.push_back(
        Matrix<NNType>::Zeros(layer_weights.height, layer_weights.width));
  }
  for (const auto &example : mini_batch) {
    const auto delta_nabla_b_and_w = Backprop(example);
    const auto delta_nabla_b = delta_nabla_b_and_w.first;
    const auto delta_nabla_w = delta_nabla_b_and_w.second;
    for (unsigned int layer_idx = 0; layer_idx < num_layers - 1; layer_idx++) {
      nabla_b[layer_idx] += delta_nabla_b[layer_idx];
      nabla_w[layer_idx] += delta_nabla_w[layer_idx];
    }
  }
  for (unsigned int layer_idx = 0; layer_idx < num_layers - 1; layer_idx++) {
    biases[layer_idx] -= (eta / mini_batch.size()) * nabla_b[layer_idx];
    weights[layer_idx] -= (eta / mini_batch.size()) * nabla_w[layer_idx];
  }
}

DeltaNablaBAndW Network::Backprop(Example example) {
  // Return a pair "(nabla_b, nabla_w)" representing the
  // gradient for the cost function C_x. "nabla_b" and
  // "nabla_w" are layer-by-layer lists of Vectors and
  // Matrixes respectively
  Biases nabla_b;
  for (const auto &layer_biases : biases) {
    nabla_b.push_back(Vector<NNType>::Zeros(layer_biases.length));
  }
  Weights nabla_w;
  for (const auto &layer_weights : weights) {
    nabla_w.push_back(
        Matrix<NNType>::Zeros(layer_weights.height, layer_weights.width));
  }

  // Feedforward
  std::vector<Vector<NNType>> activations({example.first}); // layer activations
  std::vector<Vector<NNType>> zs;                           // z vectors
  for (unsigned int layer_idx = 0; layer_idx < num_layers - 1; layer_idx++) {
    zs.push_back(weights[layer_idx] * activations.back() + biases[layer_idx]);
    activations.push_back(Sigmoid(zs.back()));
  }

  // Backward pass
  // Calculate the gradients of the last layer
  auto delta = CostDerivative(activations.back(), example.second) *
               SigmoidPrime(zs.back());
  nabla_b.back() = delta;
  nabla_w.back() = delta.OuterProduct(activations.end()[-2]);

  // Now iterate backwards from the penultimate layer (-2)
  for (int neg_layer_idx = -2; neg_layer_idx > -num_layers; neg_layer_idx--) {
    const auto z = zs.end()[neg_layer_idx];
    delta =
        weights.end()[neg_layer_idx + 1].Transpose() * delta * SigmoidPrime(z);
    nabla_b.end()[neg_layer_idx] = delta;
    nabla_w.end()[neg_layer_idx] =
        delta.OuterProduct(activations.end()[neg_layer_idx - 1]);
  }
  return std::make_pair(nabla_b, nabla_w);
}

Vector<NNType> Network::CostDerivative(Vector<NNType> output,
                                       Vector<NNType> ground_truth) {
  return output - ground_truth;
}

unsigned int Network::Evaluate(AnnotatedData test_data) {
  unsigned int n_correct = 0;
  for (const auto &example : test_data) {
    const auto output = FeedForward(example.first);
    const auto result_idx = GetMaxIndex(output);
    const auto gt_result_idx = OneHotToIndex(example.second);
    std::cout << "Testing " << output << "(" << result_idx << ") vs "
              << example.second << "(" << gt_result_idx;
    if (result_idx == gt_result_idx) {
      n_correct++;
      std::cout << ") ... correct" << std::endl;
    } else {
      std::cout << ") ... incorrect" << std::endl;
    }
  }
  return n_correct;
}

} // namespace nn
