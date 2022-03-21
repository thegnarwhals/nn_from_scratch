#include "network.hpp"

#include <vector>

#include "linear_algebra.hpp"
#include "transfer_functions.hpp"

namespace nn {

Network::Network(std::vector<unsigned int> layer_sizes)
    : layer_sizes(layer_sizes), num_layers(layer_sizes.size()) {
  // Random initialisation of weights and biases
  const NNType mean = 0.f;
  const NNType stddev = 1.f;
  for (unsigned int i = 1; i < num_layers; i++) {
    biases.push_back(Vector<NNType>::Random(layer_sizes[i], mean, stddev));
  }
  for (unsigned int i = 0; i < num_layers - 1; i++) {
    weights.push_back(Matrix<NNType>::Random(layer_sizes[i + 1], layer_sizes[i],
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
  if (test_data) {
    assert(test_data->first.size() == test_data->second.size());
  }
  const unsigned int n_test = test_data ? test_data->first.size() : 0;
  std::cout << "n_test: " << n_test << std::endl;
}
// """Train the neural network using mini-batch stochastic
// gradient descent.  The "training_data" is a list of tuples
// "(x, y)" representing the training inputs and the desired
// outputs.  The other non-optional parameters are
// self-explanatory.  If "test_data" is provided then the
// network will be evaluated against the test data after each
// epoch, and partial progress printed out.  This is useful for
// tracking progress, but slows things down substantially."""
// if test_data: n_test = len(test_data)
// n = len(training_data)
// for j in xrange(epochs):
//     random.shuffle(training_data)
//     mini_batches = [
//         training_data[k:k+mini_batch_size]
//         for k in xrange(0, n, mini_batch_size)]
//     for mini_batch in mini_batches:
//         self.update_mini_batch(mini_batch, eta)
//     if test_data:
//         print "Epoch {0}: {1} / {2}".format(
//             j, self.evaluate(test_data), n_test)
//     else:
//         print "Epoch {0} complete".format(j)

} // namespace nn
