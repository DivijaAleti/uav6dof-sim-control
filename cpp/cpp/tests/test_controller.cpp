#include "controller.hpp"

#include <iostream>

int main() {
    ControllerParams params{};

    params.kp_pos = {1.0, 1.0, 2.0};
    params.kd_pos = {0.2, 0.2, 0.4};

    params.kp_att = {0.1, 0.1, 0.05};
    params.kd_att = {0.01, 0.01, 0.01};

    params.mass = 1.5;
    params.gravity = 9.81;
    params.max_thrust = 25.0;
    params.max_torque = {0.05, 0.05, 0.02};

    CascadedPIDController controller(params);

    State state{};
    state.position = {0.0, 0.0, 0.0};
    state.velocity = {0.0, 0.0, 0.0};
    state.quaternion = {1.0, 0.0, 0.0, 0.0};
    state.omega = {0.0, 0.0, 0.0};

    Reference ref{};
    ref.position = {5.0, 0.0, -3.0};
    ref.yaw = 0.0;

    double dt = 0.005;

    ControlOutput u = controller.step(state, ref, dt);

    std::cout << "Thrust: " << u.thrust << "\n";
    std::cout << "Torque: "
              << u.torque[0] << ", "
              << u.torque[1] << ", "
              << u.torque[2] << "\n";

    return 0;
}