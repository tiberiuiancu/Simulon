#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_sim, m) {
    m.doc() = "simulon C++ simulation backend";
}
