cmake_minimum_required(VERSION 3.16)
project(poly_solver)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
add_executable(poly_solver main.cpp)
target_include_directories(poly_solver PUBLIC "../../Lesson8-Prediction/NaiveBayes")