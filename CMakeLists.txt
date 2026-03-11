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

option(SIMULON_NS3 "Build NS3 backend (requires NS3 submodule built)" OFF)

if(SIMULON_NS3)
    set(NS3_SIM_DIR "${CMAKE_SOURCE_DIR}/csrc/ns3/simulation")
    set(NS3_BUILD_DIR "${NS3_SIM_DIR}/build")

    target_sources(_sim PRIVATE
        csrc/astra-sim/network_frontend/ns3/AstraSimNetwork.cc
    )
    target_include_directories(_sim PRIVATE
        ${NS3_SIM_DIR}
        ${NS3_BUILD_DIR}/include
    )
    target_compile_definitions(_sim PRIVATE SIMULON_NS3)

    foreach(ns3_mod core network internet point-to-point applications csma)
        find_library(NS3_LIB_${ns3_mod}
            NAMES
                "ns3.36.1-${ns3_mod}-optimized"
                "ns3-${ns3_mod}"
                "ns3.optimized-${ns3_mod}-default"
            PATHS "${NS3_BUILD_DIR}/lib"
            REQUIRED)
        target_link_libraries(_sim PRIVATE ${NS3_LIB_${ns3_mod}})
    endforeach()
endif()
