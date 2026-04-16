"""setuptools shim for building the C++ pybind11 extension in-place.

Usage:
    uv sync --extra cpp
    python setup.py build_ext --inplace   # builds simulon/_mocknccl.so
"""
from setuptools import setup, Extension
import pybind11

ext = Extension(
    "simulon._mocknccl",
    sources=[
        "csrc/mocknccl/MockNcclGroup.cc",
        "csrc/mocknccl/MockNcclChannel.cc",
        "csrc/mocknccl/MockNcclLog.cc",
        "csrc/bindings.cpp",
    ],
    include_dirs=[pybind11.get_include(), "csrc"],
    extra_compile_args=["-std=c++17", "-O2"],
    language="c++",
)

setup(name="simulon-cpp", ext_modules=[ext])
