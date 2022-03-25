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

/**
 * @brief      Convert index value to "one-hot" vector
 *
 * @param[in]  index      The index
 * @param[in]  n_indexes  The total number of indexes (length of vector)
 *
 * @return     One-hot vector
 */
Vector<NNType> IndexToOneHot(unsigned int index, unsigned int n_indexes);
/**
 * @brief      "One-hot" vector to index
 *
 * @param[in]  one_hot_vector  The one-hot vector
 *
 * @return     Index
 */
unsigned int OneHotToIndex(Vector<NNType> one_hot_vector);
/**
 * @brief      Gets the index of the maximum element.
 *
 * @param[in]  vector  The vector
 *
 * @return     The index of the maximum element.
 */
unsigned int GetMaxIndex(Vector<NNType> vector);

/**
 * @brief      This interface class describes a network. It contains pure
 * virtual functions that must be overriden. Child classes are declared below.
 */
class Network {
public:
  /**
   * @brief      Constructs a new instance.
   *
   * @param[in]  layer_sizes  The layer sizes
   */
  Network(std::vector<unsigned int> layer_sizes);
  /**
   * @brief      Feed forward
   *
   * @param[in]  input  The input
   *
   * @return     The network output
   */
  Vector<NNType> FeedForward(Vector<NNType> input);
  /**
   * @brief      Stochastic gradient descent
   *
   * @param[in]  training_data    The training data
   * @param[in]  epochs           The number of epochs
   * @param[in]  mini_batch_size  The mini batch size
   * @param[in]  eta              The learning rate, eta
   * @param[in]  test_data        The optional test data
   */
  void Sgd(AnnotatedData training_data, unsigned int epochs,
           unsigned int mini_batch_size, NNType eta,
           std::optional<AnnotatedData> test_data = std::nullopt);

private:
  void UpdateMiniBatch_(AnnotatedData mini_batch, NNType eta);
  DeltaNablaBAndW Backprop_(Example example);
  static Vector<NNType> CostDerivative_(Vector<NNType> output,
                                        Vector<NNType> ground_truth);
  unsigned int Evaluate_(AnnotatedData test_data);
  virtual Vector<NNType> Nonlinearity_(Vector<NNType> weighted_inputs) = 0;
  virtual Vector<NNType> NonlinearityPrime_(Vector<NNType> weighted_inputs) = 0;
  const std::vector<unsigned int> layer_sizes_;
  const unsigned int num_layers_;
  Weights weights_;
  Biases biases_;
};

class SigmoidNetwork : public Network {
public:
  using Network::Network;
private:
  virtual Vector<NNType> Nonlinearity_(Vector<NNType> weighted_inputs) override;
  virtual Vector<NNType> NonlinearityPrime_(Vector<NNType> weighted_inputs) override;
};

class ReluNetwork : public Network {
public:
  using Network::Network;
private:
  virtual Vector<NNType> Nonlinearity_(Vector<NNType> weighted_inputs) override;
  virtual Vector<NNType> NonlinearityPrime_(Vector<NNType> weighted_inputs) override;
};

} // namespace nn