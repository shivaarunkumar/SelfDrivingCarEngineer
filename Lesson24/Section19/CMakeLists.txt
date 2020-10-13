set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(jacobian main.cpp)
target_include_directories(jacobian PUBLIC "..")
install(TARGETS jacobian CONFIGURATIONS Release RUNTIME DESTINATION bin)
