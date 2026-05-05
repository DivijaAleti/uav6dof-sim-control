#pragma once

#include <array>

struct ControllerParams {
    std::array<double, 3> kp_pos;
    std::array<double, 3> kd_pos;
    std::array<double, 3> kp_att;
    std::array<double, 3> kd_att;

    double mass;
    double gravity;
    double max_thrust;
    std::array<double, 3> max_torque;
};

struct State {
    std::array<double, 3> position;
    std::array<double, 3> velocity;
    std::array<double, 4> quaternion;
    std::array<double, 3> omega;
};

struct Reference {
    std::array<double, 3> position;
    double yaw;
};

struct ControlOutput {
    double thrust;
    std::array<double, 3> torque;
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

    std::array<double, 3> pos_integral_;
    std::array<double, 3> prev_pos_error_;

    double saturate(double value, double limit) const;
};