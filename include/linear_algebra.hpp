#pragma once
#include <cassert>
#include <iostream>
#include <random>
#include <valarray>

namespace nn {

// need to define in cpp file to avoid multiple defs
extern std::default_random_engine generator;

template <typename T> class Matrix {
public:
  Matrix(unsigned int height, unsigned int width)
      : height(height), width(width),
        rows(std::valarray<std::valarray<T>>(std::valarray<T>(width), height)) {
  }
  Matrix(const Matrix<T>&) = default;
  Matrix<T> &operator+=(const Matrix<T> &other) {
    rows += other.rows;
    return *this;
  }
  Matrix<T> &operator-=(const Matrix<T> &other) {
    rows -= other.rows;
    return *this;
  }
  Matrix<T>& operator=(Matrix<T>&& other) {
    assert(height == other.height);
    assert(width == other.width);
    rows = other.rows;
    return *this;
  }
  static Matrix<T> Random(unsigned int height, unsigned int width, T mean,
                          T stddev) {
    std::normal_distribution<T> distribution(mean, stddev);
    Matrix<T> matrix(height, width);
    for (unsigned int i = 0; i < height; i++) {
      for (unsigned int j = 0; j < width; j++) {
        matrix.rows[i][j] = distribution(generator);
      }
    }
    return matrix;
  }
  static Matrix<T> Zeros(unsigned int height, unsigned int width) {
    Matrix<T> matrix(height, width);
    for (unsigned int i = 0; i < height; i++) {
      for (unsigned int j = 0; j < width; j++) {
        matrix.rows[i][j] = static_cast<T>(0);
      }
    }
    return matrix;
  }
  Matrix<T> Transpose() {
    Matrix<T> matrix_t(width, height);
    for (unsigned int i = 0; i < height; i++) {
      for (unsigned int j = 0; j < width; j++) {
        matrix_t.rows[j][i] = rows[i][j];
      }
    }
    return matrix_t;
  }
  const unsigned int height;
  const unsigned int width;
  std::valarray<std::valarray<T>> rows;
};

template <typename T> class Vector {
public:
  explicit Vector(unsigned int length)
      : length(length), elements(std::valarray<T>(length)) {}
  Vector(std::valarray<T> elements)
      : length(elements.size()), elements(elements) {}
  Vector(const Vector<T> &other)
      : length(other.length), elements(other.elements) {}
  Vector<T> &operator=(const Vector<T> &other) {
    assert(length == other.length);
    if (this != &other) {
      elements = other.elements;
    }
    return *this;
  }
  Vector<T> &operator+=(const Vector<T> &other) {
    elements += other.elements;
    return *this;
  }
  Vector<T> &operator-=(const Vector<T> &other) {
    elements -= other.elements;
    return *this;
  }
  Matrix<T> OuterProduct(const Vector<T> &other) {
    Matrix<T> out_matrix(length, other.length);
    for (unsigned int i = 0; i < out_matrix.height; i++) {
      out_matrix.rows[i] = elements[i] * other.elements;
    }
    return out_matrix;
  }
  static Vector<T> Random(unsigned int length, T mean, T stddev) {
    std::normal_distribution<T> distribution(mean, stddev);
    Vector<T> vector(length);
    for (unsigned int i = 0; i < length; i++) {
      vector.elements[i] = distribution(generator);
    }
    return vector;
  }
  static Vector<T> Zeros(unsigned int length) {
    Vector<T> vector(length);
    for (unsigned int i = 0; i < length; i++) {
      vector.elements[i] = static_cast<T>(0);
    }
    return vector;
  }

  const unsigned int length;
  std::valarray<T> elements;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const Matrix<T> &matrix) {
  auto width = matrix.width;
  auto height = matrix.height;
  assert(height > 0);
  assert(width > 0);

  // First row
  os << "[[";
  for (unsigned int j = 0; j < width - 1; j++) {
    os << matrix.rows[0][j] << ", ";
  }
  os << matrix.rows[0][width - 1] << "]";
  if (height == 1) {
    os << "]";
    return os;
  }
  os << "," << std::endl;

  // Middle rows
  if (height > 2) {
    for (unsigned int i = 1; i < height - 1; i++) {
      os << " [";
      for (unsigned int j = 0; j < width - 1; j++) {
        os << matrix.rows[i][j] << ", ";
      }
      os << matrix.rows[i][width - 1] << "]," << std::endl;
    }
  }

  // Last row
  os << " [";
  for (unsigned int j = 0; j < width - 1; j++) {
    os << matrix.rows[height - 1][j] << ", ";
  }
  os << matrix.rows[height - 1][width - 1] << "]]";
  return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const Vector<T> &vector) {
  assert(vector.length > 0);

  os << "[";
  for (unsigned int i = 0; i < vector.length - 1; i++) {
    os << vector.elements[i] << ", ";
  }
  os << vector.elements[vector.length - 1] << "]";
  return os;
}

template <typename T>
Vector<T> operator*(const Matrix<T> &matrix, const Vector<T> &vector) {
  assert(vector.length == matrix.width);
  Vector<T> out_vector(matrix.height);
  for (unsigned int i = 0; i < matrix.height; i++) {
    out_vector.elements[i] = (matrix.rows[i] * vector.elements).sum();
  }
  return out_vector;
}

template <typename T>
Vector<T> operator*(const Vector<T> &vector1, const Vector<T> &vector2) {
  Vector<T> out_vector(vector1.elements * vector2.elements);
  return out_vector;
}

template <typename T>
Vector<T> operator*(T scalar, const Vector<T> &vector) {
  Vector<T> out_vector(scalar * vector.elements);
  return out_vector;
}

template <typename T>
Vector<T> operator-(T scalar, const Vector<T> &vector) {
  Vector<T> out_vector(scalar - vector.elements);
  return out_vector;
}

template <typename T>
Matrix<T> operator*(T scalar, const Matrix<T> &matrix) {
  Matrix<T> out_matrix(matrix.height, matrix.width);
  for (unsigned int row_idx = 0; row_idx < matrix.height; row_idx++) {
    out_matrix.rows[row_idx] = scalar * matrix.rows[row_idx];
  }
  return out_matrix;
}

template <typename T>
Vector<T> operator+(const Vector<T> &vector1, const Vector<T> &vector2) {
  assert(vector1.length == vector2.length);
  Vector<T> out_vector(vector1.elements + vector2.elements);
  return out_vector;
}

template <typename T>
Vector<T> operator-(const Vector<T> &vector1, const Vector<T> &vector2) {
  assert(vector1.length == vector2.length);
  Vector<T> out_vector(vector1.elements - vector2.elements);
  return out_vector;
}

template <typename T> Vector<T> operator-(const Vector<T> &vector) {
  Vector<T> out_vector(-vector.elements);
  return out_vector;
}

} // namespace nn
