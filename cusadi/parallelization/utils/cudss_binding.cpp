#include "../utils/cudss_interface.h"


template <typename T>
void declare_cudss_interface(py::module &m, const std::string &typestr) {
    using Class = cudssInterface<T>;
    std::string pyclass_name = "cudssInterface_" + typestr;

    py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str())
        .def(py::init<int>(), py::arg("batch_size"))
        .def("loadPointers", &Class::loadPointers)
        .def("setupMatrices", &Class::setupMatrices)
        .def("createMatrices", &Class::createMatrices)
        .def("createMatricesUniform", &Class::createMatricesUniform)
        .def("factorizeSymbolic", &Class::factorizeSymbolic)
        .def("factorizeNumeric", &Class::factorizeNumeric)
        .def("solveLinearSystem", &Class::solveLinearSystem)
        .def("printConstraintMatrixData", &Class::printConstraintMatrixData)
        .def("printConstraintVectorData", &Class::printConstraintVectorData)
        .def_readwrite("Ax_tensor", &Class::Ax_tensor)
        ;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    declare_cudss_interface<double>(m, "double");
    declare_cudss_interface<float>(m, "float");
}