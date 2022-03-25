#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include "network.hpp"

std::default_random_engine generator;

nn::AnnotatedData GenerateAnnotatedData(unsigned int n_examples) {
  std::normal_distribution<nn::NNType> distribution(0.f, 1.f);
  nn::AnnotatedData examples;
  for (unsigned int example_idx = 0; example_idx < n_examples; example_idx++) {
    nn::Vector input(std::valarray<nn::NNType>({distribution(generator)}));
    nn::Vector<nn::NNType> output(2);
    if (input.elements[0] > 0) {
      output = nn::Vector(std::valarray<nn::NNType>({1, 0}));
    } else {
      output = nn::Vector(std::valarray<nn::NNType>({0, 1}));
    }
    examples.push_back(std::make_pair(input, output));
  }
  return examples;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Positive or negative demo of NNLib" << std::endl;
    std::cout
        << "Takes one argument: nonlinearity type which is 'relu' or 'sigmoid'"
        << std::endl;
    return 0;
  }
  std::string nonlinearity{argv[1]};
  // Network that can decide whether a number is positive or negative
  // Input one number, two output neurons, one for positive, one for negative.
  std::vector<unsigned int> layer_sizes({1, 2});
  std::unique_ptr<nn::Network> network;
  if (nonlinearity == "relu") {
    network.reset(new nn::SigmoidNetwork(layer_sizes));
  } else if (nonlinearity == "sigmoid") {
    network.reset(new nn::ReluNetwork(layer_sizes));
  } else {
    std::cerr << "nonlinearity argument must be 'relu' or 'sigmoid'"
              << std::endl;
    return -1;
  }

  // Let's make some training data for estimating whether a number is positive
  // or negative
  constexpr unsigned int n_training = 80, n_test = 20;
  auto training_data = GenerateAnnotatedData(n_training);
  auto test_data = GenerateAnnotatedData(n_test);
  constexpr unsigned int epochs = 10, mini_batch_size = 10;
  constexpr float eta = 1.f;
  network->Sgd(training_data, epochs, mini_batch_size, eta, test_data);

  while (true) {
    // Run on user input
    nn::Vector<nn::NNType> input(1);
    std::cout << "Ctrl+C to quit, or enter a float to try out the network: "
              << std::flush;
    std::cin >> input.elements[0];
    std::cout << "Input: " << input << std::endl;
    const auto output = network->FeedForward(input);
    std::cout << "Output: " << output << std::endl;
    if (output.elements[0] > output.elements[1]) {
      std::cout << "Prediction: positive!" << std::endl;
    } else {
      std::cout << "Prediction: negative!" << std::endl;
    }
  }

  std::cout
      << "🎉 \033[1;32mJamie you're a genius! The script completed! Here's "
         "some green text to celebrate!\033[0m 🎉"
      << std::endl;
}
