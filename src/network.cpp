#include "network.hpp"

#include <algorithm>
#include <chrono>
#include <vector>

#include "linear_algebra.hpp"
#include "transfer_functions.hpp"

namespace nn {

Vector<NNType> IndexToOneHot(unsigned int index, unsigned int n_indexes) {
  assert(index < n_indexes);
  auto one_hot = Vector<NNType>::Zeros(n_indexes);
  one_hot.elements[index] = static_cast<NNType>(1);
  return one_hot;
}

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
    : layer_sizes_(layer_sizes), num_layers_(layer_sizes.size()) {
  std::cout << "Randomly initialising network with layer sizes [";
  for (unsigned int layer_idx = 0; layer_idx < num_layers_ - 1; layer_idx++) {
    std::cout << layer_sizes_[layer_idx] << ", ";
  }
  std::cout << layer_sizes_[num_layers_ - 1] << "]" << std::endl;
  // Random initialisation of weights and biases
  const NNType mean = 0.f;
  const NNType stddev = 1.f;
  for (unsigned int i = 1; i < num_layers_; i++) {
    biases_.push_back(Vector<NNType>::Random(layer_sizes_[i], mean, stddev));
  }
  for (unsigned int i = 1; i < num_layers_; i++) {
    weights_.push_back(Matrix<NNType>::Random(layer_sizes_[i], layer_sizes_[i - 1],
                                             mean, stddev));
  }
}

Vector<NNType> Network::FeedForward(Vector<NNType> input) {
  std::vector<Vector<NNType>> layer_outputs({input});
  for (unsigned int i = 1; i < num_layers_; i++) {
    layer_outputs.push_back(
        Nonlinearity_(weights_[i - 1] * layer_outputs.back() + biases_[i - 1]));
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
  if (test_data) {
    std::cout << "Initial evaluation: " << Evaluate_(test_data.value()) << " / "
              << n_test << std::endl;
  }
  const unsigned int n_training = training_data.size();
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
    unsigned int mini_batch_idx = 0;
    for (const auto &mini_batch : mini_batches) {
      UpdateMiniBatch_(mini_batch, eta);
    }
    if (test_data) {
      std::cout << "Epoch " << epoch_idx << ": " << Evaluate_(test_data.value())
                << " / " << n_test << std::endl;
    } else {
      std::cout << "Epoch " << epoch_idx << " complete" << std::endl;
    }
  }
}

void Network::UpdateMiniBatch_(AnnotatedData mini_batch, NNType eta) {
  Biases nabla_b;
  for (const auto &layer_biases : biases_) {
    nabla_b.push_back(Vector<NNType>::Zeros(layer_biases.length));
  }
  Weights nabla_w;
  for (const auto &layer_weights : weights_) {
    nabla_w.push_back(
        Matrix<NNType>::Zeros(layer_weights.height, layer_weights.width));
  }
  for (const auto &example : mini_batch) {
    const auto delta_nabla_b_and_w = Backprop_(example);
    const auto delta_nabla_b = delta_nabla_b_and_w.first;
    const auto delta_nabla_w = delta_nabla_b_and_w.second;
    for (unsigned int layer_idx = 0; layer_idx < num_layers_ - 1; layer_idx++) {
      nabla_b[layer_idx] += delta_nabla_b[layer_idx];
      nabla_w[layer_idx] += delta_nabla_w[layer_idx];
    }
  }
  for (unsigned int layer_idx = 0; layer_idx < num_layers_ - 1; layer_idx++) {
    biases_[layer_idx] -= (eta / mini_batch.size()) * nabla_b[layer_idx];
    weights_[layer_idx] -= (eta / mini_batch.size()) * nabla_w[layer_idx];
  }
}

DeltaNablaBAndW Network::Backprop_(Example example) {
  // Return a pair "(nabla_b, nabla_w)" representing the
  // gradient for the cost function C_x. "nabla_b" and
  // "nabla_w" are layer-by-layer lists of Vectors and
  // Matrixes respectively
  Biases nabla_b;
  for (const auto &layer_biases : biases_) {
    nabla_b.push_back(Vector<NNType>::Zeros(layer_biases.length));
  }
  Weights nabla_w;
  for (const auto &layer_weights : weights_) {
    nabla_w.push_back(
        Matrix<NNType>::Zeros(layer_weights.height, layer_weights.width));
  }

  // Feedforward
  std::vector<Vector<NNType>> activations({example.first}); // layer activations
  std::vector<Vector<NNType>> zs;                           // z vectors
  for (unsigned int layer_idx = 0; layer_idx < num_layers_ - 1; layer_idx++) {
    zs.push_back(weights_[layer_idx] * activations.back() + biases_[layer_idx]);
    activations.push_back(Nonlinearity_(zs.back()));
  }

  // Backward pass
  // Calculate the gradients of the last layer
  auto delta = CostDerivative_(activations.back(), example.second) *
               NonlinearityPrime_(zs.back());
  nabla_b.back() = delta;
  nabla_w.back() = delta.OuterProduct(activations.end()[-2]);

  // Now iterate backwards from the penultimate layer (-2)
  for (int neg_layer_idx = -2; neg_layer_idx > -num_layers_; neg_layer_idx--) {
    const auto z = zs.end()[neg_layer_idx];
    nabla_b.end()[neg_layer_idx] =
        weights_.end()[neg_layer_idx + 1].Transpose() *
        nabla_b.end()[neg_layer_idx + 1] * NonlinearityPrime_(z);
    nabla_w.end()[neg_layer_idx] = nabla_b.end()[neg_layer_idx].OuterProduct(
        activations.end()[neg_layer_idx - 1]);
  }
  return std::make_pair(nabla_b, nabla_w);
}

Vector<NNType> Network::CostDerivative_(Vector<NNType> output,
                                       Vector<NNType> ground_truth) {
  return output - ground_truth;
}

unsigned int Network::Evaluate_(AnnotatedData test_data) {
  unsigned int n_correct = 0;
  for (const auto &example : test_data) {
    const auto output = FeedForward(example.first);
    const auto result_idx = GetMaxIndex(output);
    const auto gt_result_idx = OneHotToIndex(example.second);
    if (result_idx == gt_result_idx) {
      n_correct++;
    }
  }
  return n_correct;
}

Vector<NNType> SigmoidNetwork::Nonlinearity_(Vector<NNType> weighted_inputs) {
  return Sigmoid(weighted_inputs);
}

Vector<NNType> SigmoidNetwork::NonlinearityPrime_(Vector<NNType> weighted_inputs) {
  return SigmoidPrime(weighted_inputs);
}

Vector<NNType> ReluNetwork::Nonlinearity_(Vector<NNType> weighted_inputs) {
  return Relu(weighted_inputs);
}

Vector<NNType> ReluNetwork::NonlinearityPrime_(Vector<NNType> weighted_inputs) {
  return ReluPrime(weighted_inputs);
}

} // namespace nn
