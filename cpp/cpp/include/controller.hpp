#pragma once

#include <array>

struct ControllerParams {
    double kp_pos[3];
    double kd_pos[3];
    double kp_att[3];
    double kd_att[3];

    double mass;
    double gravity;
    double max_thrust;
    double max_torque[3];
};

struct State {
    double position[3];
    double velocity[3];
    double quaternion[4];
    double omega[3];
};

struct Reference {
    double position[3];
    double yaw;
};

struct ControlOutput {
    double thrust;
    double torque[3];
};

class CascadedPIDController {
public:
    explicit CascadedPIDController(const ControllerParams& params);

    ControlOutput step(
        const State& state,
        const Reference& ref,
        double dt
    );

private:
    ControllerParams params_;

    double pos_integral_[3];
    double prev_pos_error_[3];

    double saturate(double value, double limit) const;
};