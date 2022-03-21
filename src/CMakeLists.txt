add_library(NNLib linear_algebra.cpp network.cpp)

target_include_directories(NNLib PUBLIC "${PROJECT_SOURCE_DIR}/include")

# C++17 required
set_property(TARGET NNLib PROPERTY CXX_STANDARD 17)
