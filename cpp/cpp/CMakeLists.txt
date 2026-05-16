cmake_minimum_required(VERSION 3.15)

project(uav6dof_controller_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED CONFIG)

include_directories(include)

add_library(controller
    src/controller.cpp
)

add_executable(test_controller
    tests/test_controller.cpp
)

target_link_libraries(test_controller
    PRIVATE controller
)

pybind11_add_module(uav_controller_cpp
    bindings/controller_bindings.cpp
)

target_link_libraries(uav_controller_cpp
    PRIVATE controller
)
