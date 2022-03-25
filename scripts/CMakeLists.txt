# add the executable
add_executable(positive_or_negative positive_or_negative.cpp)

target_link_libraries(positive_or_negative PUBLIC NNLib)

target_include_directories(positive_or_negative PUBLIC "${PROJECT_SOURCE_DIR}/include")

# C++17 required
set_property(TARGET positive_or_negative PROPERTY CXX_STANDARD 17)

add_executable(mnist mnist.cpp)

target_link_libraries(mnist PUBLIC NNLib)

target_include_directories(mnist PUBLIC "${PROJECT_SOURCE_DIR}/include")

# C++17 required
set_property(TARGET mnist PROPERTY CXX_STANDARD 17)
