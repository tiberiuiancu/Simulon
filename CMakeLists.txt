cmake_minimum_required(VERSION 3.20)
project(simulon LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

pybind11_add_module(_sim
    csrc/bindings.cpp
    csrc/astra_bindings/topology_bridge.cc
    csrc/astra_bindings/workload_bridge.cc
)

target_include_directories(_sim PRIVATE
    csrc
    csrc/astra-sim
)

install(TARGETS _sim DESTINATION simulon)
