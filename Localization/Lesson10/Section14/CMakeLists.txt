cmake_minimum_required(VERSION 3.16)
project(hybrid_a_star)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
add_executable(hybrid_a_star main.cpp hybrid_breadth_first.cpp)
