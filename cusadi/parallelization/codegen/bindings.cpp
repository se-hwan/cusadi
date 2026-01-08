#include "bindings.h"
#include "../utils/cudss_interface.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("tau", &tau_binding);
    m.def("ADMM_step", &ADMM_step_binding);
    // py::class_<cudssInterface, std::shared_ptr<cudssInterface>>(m, "cudssInterface")
    //     .def(py::init<int, int>(), py::arg("batch_size"), py::arg("dim"),
    //          "Constructor that initializes cudssInterface with the given size.")
    //     .def("loadPointers", &cudssInterface::loadPointers, "Load tensors containing pointers to data.")
    //     .def("createMatrices", &cudssInterface::createMatrices, "Create cuDSS matrices from Pytorch pointers.")
    //     .def("factorizeSymbolic", &cudssInterface::factorizeSymbolic, "Analysis phase of cudssExecute.")
    //     .def("factorizeNumeric", &cudssInterface::factorizeNumeric, "Factorization phase of cudssExecute.")
    //     .def("solveLinearSystem", &cudssInterface::solveLinearSystem, "Solve phase of cudssExecute.")
    //     .def("printConstraintMatrixData", &cudssInterface::printConstraintMatrixData, "Print constraint data on GPU.")
    //     .def("printConstraintVectorData", &cudssInterface::printConstraintVectorData, "Print constraint data on GPU.");
}