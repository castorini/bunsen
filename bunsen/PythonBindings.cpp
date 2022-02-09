#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "bunsen/typing/TensorCheck.h"
#include "bunsen/typing/TensorCheckTypes.h"

namespace py = pybind11;


PYBIND11_MODULE(bunsen_internals, m) {
  m.def("make_byte_check", bunsen::typing::MakeByteCheck);
  m.def("make_short_check", bunsen::typing::MakeShortCheck);
  m.def("make_int_check", bunsen::typing::MakeIntCheck);
  m.def("make_long_check", bunsen::typing::MakeLongCheck);
  m.def("make_float_check", bunsen::typing::MakeFloatCheck);
  m.def("make_double_check", bunsen::typing::MakeDoubleCheck);
  m.def("make_half_check", bunsen::typing::MakeHalfCheck);
  m.def("make_floatlike_check", bunsen::typing::MakeFloatLikeCheck);
  m.def("make_intlike_check", bunsen::typing::MakeIntLikeCheck);
  m.def("make_any_check", bunsen::typing::MakeAnyCheck);

  py::class_<bunsen::typing::TensorCheck>(m, "TensorCheck")
      .def("check", &bunsen::typing::TensorCheck::Check)
      .def("with_dim_names", &bunsen::typing::TensorCheck::WithDimNames)
      .def("with_names", &bunsen::typing::TensorCheck::WithNames)
      .def("with_dim_shapes", &bunsen::typing::TensorCheck::WithDimShapes)
      .def("with_shapes", &bunsen::typing::TensorCheck::WithShapes);
}

