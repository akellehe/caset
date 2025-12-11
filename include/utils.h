#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace caset {
template<typename T>
inline py::object wrap_non_owning(T *p) {
  auto ptr = std::shared_ptr<T>(p, [](T*){/* no delete */});
  return py::cast(ptr, py::return_value_policy::reference_internal);
}

};