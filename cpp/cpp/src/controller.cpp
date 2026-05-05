#include "controller.hpp"

#include <algorithm>
#include <cmath>

CascadedPIDController::CascadedPIDController(
    const ControllerParams& params
) : params_(params) {
    pos_integral_.fill(0.0);
    prev_pos_error_.fill(0.0);
}

double CascadedPIDController::saturate(
    double value,
    double limit
) const {
    return std::max(-limit, std::min(limit, value));
}

ControlOutput CascadedPIDController::step(
    const State& state,
    const Reference& ref,
    double dt
) {
    ControlOutput output{};
    output.thrust = 0.0;
    output.torque.fill(0.0);

    std::array<double, 3> pos_error{};
    std::array<double, 3> vel_cmd{};
    std::array<double, 3> vel_error{};

    for (int i = 0; i < 3; ++i) {
        pos_error[i] = ref.position[i] - state.position[i];

        pos_integral_[i] += pos_error[i] * dt;

        double d_error =
            (pos_error[i] - prev_pos_error_[i]) / dt;

        vel_cmd[i] =
            params_.kp_pos[i] * pos_error[i]
            + params_.kd_pos[i] * d_error;

        prev_pos_error_[i] = pos_error[i];
    }

    for (int i = 0; i < 3; ++i) {
        vel_error[i] = vel_cmd[i] - state.velocity[i];
    }

    double desired_accel_z =
        params_.kp_pos[2] * pos_error[2]
        + params_.kd_pos[2] * vel_error[2];

    output.thrust =
        params_.mass * (params_.gravity - desired_accel_z);

    output.thrust =
        std::max(0.0, std::min(params_.max_thrust, output.thrust));

    for (int i = 0; i < 3; ++i) {
        double torque_cmd =
            params_.kp_att[i] * vel_error[i]
            - params_.kd_att[i] * state.omega[i];

        output.torque[i] =
            saturate(torque_cmd, params_.max_torque[i]);
    }

    return output;
}