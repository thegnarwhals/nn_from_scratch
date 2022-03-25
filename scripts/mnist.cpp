#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "network.hpp"

typedef unsigned char BYTE;

/**
 * @brief      Reads a file.
 *
 * @param[in]  path  The path to read
 *
 * @return     Vector of bytes contained in the file
 */
std::vector<BYTE> ReadFile(const std::string path) {
  std::cout << "Reading " << path << std::endl;
  std::ifstream input(path, std::ios::binary);
  std::vector<BYTE> bytes(std::istreambuf_iterator<char>(input), {});
  return bytes;
}

/**
 * @brief      Convert four bytes from iterator into a number
 *
 * @param[in]  bytes  The bytes
 *
 * @return     The number
 */
unsigned int FourBytesToNumber(std::vector<BYTE>::iterator bytes) {
  return (static_cast<unsigned int>(bytes[0]) << 24) +
         (static_cast<unsigned int>(bytes[1]) << 16) +
         (static_cast<unsigned int>(bytes[2]) << 8) +
         (static_cast<unsigned int>(bytes[3]));
}

/**
 * @brief      Draws an image to the console output.
 *
 * @param[in]  image  The matrix representing the image to draw
 */
void DrawImage(nn::Matrix<nn::NNType> image) {
  assert(image.height % 2 == 0);

  // Iterate over even rows
  for (unsigned int row_idx = 0; row_idx < image.height; row_idx += 2) {
    for (unsigned int col_idx = 0; col_idx < image.width; col_idx++) {
      if (image.rows[row_idx][col_idx] < 0.5) {
        if (image.rows[row_idx + 1][col_idx] < 0.5) {
          std::cout << " "; // Black and black
        } else {
          std::cout << "▄"; // Black and white
        }
      } else {
        if (image.rows[row_idx + 1][col_idx] < 0.5) {
          std::cout << "▀"; // White and black
        } else {
          std::cout << "█"; // White and white
        }
      }
    }
    std::cout << std::endl;
  }
}

/**
 * @brief      Reads an index matrix file.
 *
 * @param[in]  path  The path to read
 *
 * @return     The images stored in the file
 */
std::vector<nn::Matrix<nn::NNType>> ReadIdxMatrixFile(const std::string path) {
  auto bytes = ReadFile(path);
  assert(bytes[0] == 0);
  assert(bytes[1] == 0);
  assert(bytes[2] == 8); // unsigned char identifier
  assert(bytes[3] == 3); // dims of file
  auto iterator = bytes.begin() + 4;
  const auto n_images = FourBytesToNumber(iterator);
  iterator += 4;
  const auto n_rows = FourBytesToNumber(iterator);
  iterator += 4;
  const auto n_cols = FourBytesToNumber(iterator);
  iterator += 4;
  assert(n_images * n_rows * n_cols + std::distance(bytes.begin(), iterator) ==
         bytes.size());
  std::vector<nn::Matrix<nn::NNType>> images;
  for (unsigned int image_idx = 0; image_idx < n_images; image_idx++) {
    nn::Matrix<nn::NNType> image(n_rows, n_cols);
    for (unsigned int row_idx = 0; row_idx < n_rows; row_idx++) {
      for (unsigned int col_idx = 0; col_idx < n_cols; col_idx++) {
        image.rows[row_idx][col_idx] =
            static_cast<nn::NNType>(*iterator) / 255.f;
        iterator++;
      }
    }
    images.push_back(image);
  }
  return images;
}

/**
 * @brief      Reads an index label file.
 *
 * @param[in]  path  The path to read
 *
 * @return     The labels stored in the file
 */
std::vector<BYTE> ReadIdxLabelFile(const std::string path) {
  auto bytes = ReadFile(path);
  assert(bytes[0] == 0);
  assert(bytes[1] == 0);
  assert(bytes[2] == 8); // unsigned char identifier
  assert(bytes[3] == 1); // dims of file
  auto iterator = bytes.begin() + 4;
  const auto n_labels = FourBytesToNumber(iterator);
  iterator += 4;
  assert(n_labels + std::distance(bytes.begin(), iterator) == bytes.size());
  return std::vector<BYTE>(iterator, bytes.end());
}

/**
 * @brief      Generate annotated data from images and labels
 *
 * @param[in]  images  The images
 * @param[in]  labels  The labels
 *
 * @return     The vectors representing the input and output of the matrix
 * corresponding to the images and labels
 */
nn::AnnotatedData
GenerateAnnotatedData(std::vector<nn::Matrix<nn::NNType>> images,
                      std::vector<BYTE> labels) {
  assert(images.size() == labels.size());
  nn::AnnotatedData annotated_data;
  for (unsigned int image_idx = 0; image_idx < images.size(); image_idx++) {
    const auto image = images[image_idx];
    nn::Vector<nn::NNType> input_vector(image.width * image.height);
    for (unsigned int row_idx = 0; row_idx < image.height; row_idx++) {
      for (unsigned int col_idx = 0; col_idx < image.width; col_idx++) {
        input_vector.elements[row_idx * image.width + col_idx] =
            image.rows[row_idx][col_idx];
      }
    }
    const auto output_vector = nn::IndexToOneHot(labels[image_idx], 10);
    const auto example = std::make_pair(input_vector, output_vector);
    annotated_data.push_back(example);
  }
  return annotated_data;
}

/**
 * @brief      Run MNIST demo of NNLib
 *
 * @param[in]  argc  The count of arguments
 * @param      argv  The arguments array (see printed message for details)
 *
 * @return     0 if successful
 */
int main(int argc, char **argv) {
  if (argc != 5) {
    std::cout << "MNIST demo of NNLib" << std::endl;
    std::cout << "Arguments are paths to the following files, in this order"
              << std::endl;
    std::cout << "1. train-images.idx3-ubyte: training set images" << std::endl;
    std::cout << "2. train-labels.idx1-ubyte: training set labels" << std::endl;
    std::cout << "3. t10k-images.idx3-ubyte:  test set images" << std::endl;
    std::cout << "4. t10k-labels.idx1-ubyte:  test set labels" << std::endl;
    std::cout << "Download from http://yann.lecun.com/exdb/mnist/" << std::endl;
    return 0;
  }
  const std::string train_images_path = argv[1];
  const std::string train_labels_path = argv[2];
  const std::string test_images_path = argv[3];
  const std::string test_labels_path = argv[4];

  const auto train_images = ReadIdxMatrixFile(train_images_path);
  const auto train_labels = ReadIdxLabelFile(train_labels_path);
  const auto training_data = GenerateAnnotatedData(train_images, train_labels);

  const auto test_images = ReadIdxMatrixFile(test_images_path);
  const auto test_labels = ReadIdxLabelFile(test_labels_path);
  const auto test_data = GenerateAnnotatedData(test_images, test_labels);

  nn::Network network(
      {training_data[0].first.length, 16, 16, training_data[0].second.length});
  constexpr unsigned int epochs = 30, mini_batch_size = 10;
  constexpr float eta = 3.f;
  network.Sgd(training_data, epochs, mini_batch_size, eta, test_data);
  for (unsigned int image_idx = 0; image_idx < test_images.size();
       image_idx++) {
    DrawImage(test_images[image_idx]);
    std::cout << "Actual: "
              << static_cast<unsigned int>(test_labels[image_idx]);
    const auto output = network.FeedForward(test_data[image_idx].first);
    std::cout << ", Network: " << nn::GetMaxIndex(output) << std::endl;
  }

  std::cout
      << "🎉 \033[1;32mJamie you're a genius! The script completed! Here's "
         "some green text to celebrate!\033[0m 🎉"
      << std::endl;
  return 0;
}
