cmake_minimum_required(VERSION 3.15)

project(uav6dof_controller_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_library(controller
    src/controller.cpp
)

add_executable(test_controller
    tests/test_controller.cpp
)

target_link_libraries(test_controller
    controller
)