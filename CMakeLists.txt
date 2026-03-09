cmake_minimum_required(VERSION 3.20)
project(simulon LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

pybind11_add_module(_sim
    csrc/bindings.cpp
    csrc/astra_bindings/topology_bridge.cc
    csrc/astra_bindings/workload_bridge.cc
    csrc/astra_bindings/analytical_runner.cc
)

target_include_directories(_sim PRIVATE
    csrc
    csrc/astra-sim
)

# Add ASTRA-Sim source files for analytical backend
file(GLOB ASTRA_SIM_SOURCES
    "csrc/astra-sim/system/*.cc"
    "csrc/astra-sim/system/collective/*.cc"
    "csrc/astra-sim/system/fast-backend/*.cc"
    "csrc/astra-sim/system/memory/*.cc"
    "csrc/astra-sim/system/scheduling/*.cc"
    "csrc/astra-sim/system/topology/*.cc"
    "csrc/astra-sim/workload/*.cc"
    "csrc/astra-sim/network_frontend/analytical/*.cc"
)

# Exclude files that require physical network or have hardcoded paths we can't fix
list(FILTER ASTRA_SIM_SOURCES EXCLUDE REGEX ".*SimAiFlowModelRdma.cc")
list(FILTER ASTRA_SIM_SOURCES EXCLUDE REGEX ".*BootStrapnet.cc")
list(FILTER ASTRA_SIM_SOURCES EXCLUDE REGEX ".*PhyMultiThread.cc")
list(FILTER ASTRA_SIM_SOURCES EXCLUDE REGEX ".*AnalyticalAstra.cc")  # Main file, not needed

target_sources(_sim PRIVATE ${ASTRA_SIM_SOURCES})

# Add compile definition for analytical backend
target_compile_definitions(_sim PRIVATE ANALYTI)

install(TARGETS _sim DESTINATION simulon)
