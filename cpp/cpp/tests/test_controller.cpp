#include "controller.hpp"

#include <iostream>

int main() {
    ControllerParams params{};

    params.kp_pos[0] = 1.0;
    params.kp_pos[1] = 1.0;
    params.kp_pos[2] = 2.0;

    params.kd_pos[0] = 0.2;
    params.kd_pos[1] = 0.2;
    params.kd_pos[2] = 0.4;

    params.kp_att[0] = 0.1;
    params.kp_att[1] = 0.1;
    params.kp_att[2] = 0.05;

    params.kd_att[0] = 0.01;
    params.kd_att[1] = 0.01;
    params.kd_att[2] = 0.01;

    params.mass = 1.5;
    params.gravity = 9.81;
    params.max_thrust = 25.0;

    params.max_torque[0] = 0.05;
    params.max_torque[1] = 0.05;
    params.max_torque[2] = 0.02;

    CascadedPIDController controller(params);

    State state{};
    state.position[0] = 0.0;
    state.position[1] = 0.0;
    state.position[2] = 0.0;

    state.velocity[0] = 0.0;
    state.velocity[1] = 0.0;
    state.velocity[2] = 0.0;

    state.quaternion[0] = 1.0;
    state.quaternion[1] = 0.0;
    state.quaternion[2] = 0.0;
    state.quaternion[3] = 0.0;

    state.omega[0] = 0.0;
    state.omega[1] = 0.0;
    state.omega[2] = 0.0;

    Reference ref{};
    ref.position[0] = 5.0;
    ref.position[1] = 0.0;
    ref.position[2] = -3.0;
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