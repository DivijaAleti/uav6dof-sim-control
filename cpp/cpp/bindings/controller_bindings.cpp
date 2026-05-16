#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "controller.hpp"

namespace py = pybind11;

PYBIND11_MODULE(uav_controller_cpp, m) {
    py::class_<ControllerParams>(m, "ControllerParams")
        .def(py::init<>())
        .def_readwrite("pos_kp", &ControllerParams::pos_kp)
        .def_readwrite("pos_ki", &ControllerParams::pos_ki)
        .def_readwrite("pos_kd", &ControllerParams::pos_kd)
        .def_readwrite("vel_kp", &ControllerParams::vel_kp)
        .def_readwrite("vel_ki", &ControllerParams::vel_ki)
        .def_readwrite("vel_kd", &ControllerParams::vel_kd)
        .def_readwrite("roll_kp", &ControllerParams::roll_kp)
        .def_readwrite("roll_ki", &ControllerParams::roll_ki)
        .def_readwrite("roll_kd", &ControllerParams::roll_kd)
        .def_readwrite("pitch_kp", &ControllerParams::pitch_kp)
        .def_readwrite("pitch_ki", &ControllerParams::pitch_ki)
        .def_readwrite("pitch_kd", &ControllerParams::pitch_kd)
        .def_readwrite("yaw_kp", &ControllerParams::yaw_kp)
        .def_readwrite("yaw_ki", &ControllerParams::yaw_ki)
        .def_readwrite("yaw_kd", &ControllerParams::yaw_kd)
        .def_readwrite("mass", &ControllerParams::mass)
        .def_readwrite("gravity", &ControllerParams::gravity)
        .def_readwrite("max_thrust", &ControllerParams::max_thrust)
        .def_readwrite("max_torque", &ControllerParams::max_torque)
        .def_readwrite("max_tilt", &ControllerParams::max_tilt)
        .def_readwrite("kz_p", &ControllerParams::kz_p)
        .def_readwrite("kz_d", &ControllerParams::kz_d)
        .def_readwrite("kx_p", &ControllerParams::kx_p)
        .def_readwrite("kx_d", &ControllerParams::kx_d)
        .def_readwrite("ky_p", &ControllerParams::ky_p)
        .def_readwrite("ky_d", &ControllerParams::ky_d);

    py::class_<State>(m, "State")
        .def(py::init<>())
        .def_readwrite("position", &State::position)
        .def_readwrite("velocity_body", &State::velocity_body)
        .def_readwrite("quaternion", &State::quaternion)
        .def_readwrite("omega", &State::omega);

    py::class_<Reference>(m, "Reference")
        .def(py::init<>())
        .def_readwrite("position", &Reference::position)
        .def_readwrite("velocity", &Reference::velocity)
        .def_readwrite("yaw", &Reference::yaw);

    py::class_<ControlOutput>(m, "ControlOutput")
        .def(py::init<>())
        .def_readonly("thrust", &ControlOutput::thrust)
        .def_readonly("torque", &ControlOutput::torque);

    py::class_<PID>(m, "PID")
        .def(py::init<double, double, double, double>())
        .def("step", &PID::step);

        py::class_<CascadedPIDController>(m, "CascadedPIDController")
        .def(py::init<const ControllerParams&>())
        .def("step", &CascadedPIDController::step);
}