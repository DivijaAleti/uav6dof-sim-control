#include "controller.hpp"

#include <cmath>
#include <iostream>

int main() {
    constexpr double pi = 3.14159265358979323846;

    ControllerParams params{};

    params.pos_kp = {1.0, 1.0, 1.0};
    params.pos_ki = {0.0, 0.0, 0.0};
    params.pos_kd = {0.0, 0.0, 0.0};

    params.vel_kp = {0.0, 0.0, 0.0};
    params.vel_ki = {0.0, 0.0, 0.0};
    params.vel_kd = {0.0, 0.0, 0.0};

    params.roll_kp = 0.8;
    params.roll_ki = 0.0;
    params.roll_kd = 0.05;

    params.pitch_kp = 0.8;
    params.pitch_ki = 0.0;
    params.pitch_kd = 0.05;

    params.yaw_kp = 0.4;
    params.yaw_ki = 0.0;
    params.yaw_kd = 0.02;

    params.mass = 1.5;
    params.gravity = 9.80665;

    params.max_thrust = {0.0, 25.0};
    params.max_torque = {0.03, 0.08, 0.04};

    params.max_tilt = 5.0 * pi / 180.0;

    params.kz_p = 1.0;
    params.kz_d = 1.5;

    params.kx_p = 0.10;
    params.kx_d = 0.20;

    params.ky_p = 0.15;
    params.ky_d = 0.20;

    CascadedPIDController controller(params);

    State state{};

    state.position = {0.0, 0.0, 0.0};
    state.velocity_body = {0.0, 0.0, 0.0};
    state.quaternion = {1.0, 0.0, 0.0, 0.0};
    state.omega = {0.0, 0.0, 0.0};

    Reference ref{};

    ref.position = {5.0, 0.0, -3.0};
    ref.velocity = {0.0, 0.0, 0.0};
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