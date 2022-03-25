#include <random>

namespace nn {
std::default_random_engine generator; // Need to define this here rather than
                                      // header to avoid multiple defs
}