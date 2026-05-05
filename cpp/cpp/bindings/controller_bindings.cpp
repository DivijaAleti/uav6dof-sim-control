#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "controller.hpp"

namespace py = pybind11;

PYBIND11_MODULE(uav_controller_cpp, m) {
    py::class_<ControllerParams>(m, "ControllerParams")
        .def(py::init<>())
        .def_readwrite("kp_pos", &ControllerParams::kp_pos)
        .def_readwrite("kd_pos", &ControllerParams::kd_pos)
        .def_readwrite("kp_att", &ControllerParams::kp_att)
        .def_readwrite("kd_att", &ControllerParams::kd_att)
        .def_readwrite("mass", &ControllerParams::mass)
        .def_readwrite("gravity", &ControllerParams::gravity)
        .def_readwrite("max_thrust", &ControllerParams::max_thrust)
        .def_readwrite("max_torque", &ControllerParams::max_torque);

    py::class_<State>(m, "State")
        .def(py::init<>())
        .def_readwrite("position", &State::position)
        .def_readwrite("velocity", &State::velocity)
        .def_readwrite("quaternion", &State::quaternion)
        .def_readwrite("omega", &State::omega);

    py::class_<Reference>(m, "Reference")
        .def(py::init<>())
        .def_readwrite("position", &Reference::position)
        .def_readwrite("yaw", &Reference::yaw);

    py::class_<ControlOutput>(m, "ControlOutput")
        .def(py::init<>())
        .def_readonly("thrust", &ControlOutput::thrust)
        .def_readonly("torque", &ControlOutput::torque);

    py::class_<CascadedPIDController>(m, "CascadedPIDController")
        .def(py::init<const ControllerParams&>())
        .def("step", &CascadedPIDController::step);
}