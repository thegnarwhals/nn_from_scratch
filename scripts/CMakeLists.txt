# add the executable
add_executable(NeuralNetworkFromScratch main.cpp)

target_link_libraries(NeuralNetworkFromScratch PUBLIC NNLib)

target_include_directories(NeuralNetworkFromScratch PUBLIC "${PROJECT_SOURCE_DIR}/include")

# C++17 required
set_property(TARGET NeuralNetworkFromScratch PROPERTY CXX_STANDARD 17)
