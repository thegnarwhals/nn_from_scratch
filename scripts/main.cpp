#include <iostream>
#include <utility>

#include "network.hpp"

int main() {
  nn::Network network({1, 1, 1, 1});
  std::cout << "Initial weights:" << std::endl;
  for (const auto &weights : network.weights) {
    std::cout << weights << std::endl;
  }
  std::cout << "Initial biases:" << std::endl;
  for (const auto &biases : network.biases) {
    std::cout << biases << std::endl;
  }
  nn::Vector<nn::NNType> input(std::valarray<nn::NNType>{1});
  std::cout << "Input: " << input << std::endl;
  auto output = network.FeedForward(input);
  std::cout << "Output: " << output << std::endl;
  network.Sgd(std::make_pair<std::vector<nn::Vector<nn::NNType>>,
                             std::vector<nn::Vector<nn::NNType>>>({}, {}),
              0, 0, 0);
  std::cout
      << "🎉 \033[1;32mJamie you're a genius! The script completed! Here's "
         "some green text to celebrate!\033[0m 🎉"
      << std::endl;
}
