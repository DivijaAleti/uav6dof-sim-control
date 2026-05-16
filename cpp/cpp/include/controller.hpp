#pragma once

#include <array>

struct ControllerParams {
    std::array<double, 3> pos_kp;
    std::array<double, 3> pos_ki;
    std::array<double, 3> pos_kd;

    std::array<double, 3> vel_kp;
    std::array<double, 3> vel_ki;
    std::array<double, 3> vel_kd;

    double roll_kp;
    double roll_ki;
    double roll_kd;

    double pitch_kp;
    double pitch_ki;
    double pitch_kd;

    double yaw_kp;
    double yaw_ki;
    double yaw_kd;

    double mass;
    double gravity;

    std::array<double, 2> max_thrust;
    std::array<double, 3> max_torque;

    double max_tilt;

    double kz_p;
    double kz_d;
    double kx_p;
    double kx_d;
    double ky_p;
    double ky_d;
};

struct State {
    std::array<double, 3> position;
    std::array<double, 3> velocity_body;
    std::array<double, 4> quaternion; // [qw, qx, qy, qz]
    std::array<double, 3> omega;
};

struct Reference {
    std::array<double, 3> position;
    std::array<double, 3> velocity;
    double yaw;
};

struct ControlOutput {
    double thrust;
    std::array<double, 3> torque;
};

class PID {
public:
    PID(double kp, double ki, double kd, double i_lim);

    double step(double error, double dt);

private:
    double kp_;
    double ki_;
    double kd_;
    double i_;
    double i_lim_;
    double prev_error_;
    bool has_prev_error_;
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

    PID p_roll_;
    PID p_pitch_;
    PID p_yaw_;

    std::array<double, 3> prev_position_;
    bool has_prev_position_;

    static double clip(double x, double lo, double hi);
    static double wrapToPi(double angle);

    static std::array<double, 4> quatNorm(
        const std::array<double, 4>& q
    );

    static std::array<double, 3> quatToEulerSmall(
        const std::array<double, 4>& q
    );
};