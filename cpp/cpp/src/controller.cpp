#include "controller.hpp"

#include <algorithm>
#include <cmath>

PID::PID(double kp, double ki, double kd, double i_lim)
    : kp_(kp),
      ki_(ki),
      kd_(kd),
      i_(0.0),
      i_lim_(i_lim),
      prev_error_(0.0),
      has_prev_error_(false) {}

double PID::step(double error, double dt) {
    double safe_dt = std::max(dt, 1e-9);

    double de = 0.0;
    if (has_prev_error_) {
        de = (error - prev_error_) / safe_dt;
    }

    prev_error_ = error;
    has_prev_error_ = true;

    i_ += error * dt;
    i_ = std::max(-i_lim_, std::min(i_lim_, i_));

    return kp_ * error + ki_ * i_ + kd_ * de;
}

CascadedPIDController::CascadedPIDController(
    const ControllerParams& params
)
    : params_(params),
      p_roll_(params.roll_kp, params.roll_ki, params.roll_kd, 1.0),
      p_pitch_(params.pitch_kp, params.pitch_ki, params.pitch_kd, 1.0),
      p_yaw_(params.yaw_kp, params.yaw_ki, params.yaw_kd, 1.0),
      has_prev_position_(false) {
    prev_position_ = {0.0, 0.0, 0.0};
}

double CascadedPIDController::clip(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
}

double CascadedPIDController::wrapToPi(double angle) {
    constexpr double pi = 3.14159265358979323846;
    constexpr double two_pi = 2.0 * pi;

    angle = std::fmod(angle + pi, two_pi);
    if (angle < 0.0) {
        angle += two_pi;
    }

    return angle - pi;
}

std::array<double, 4> CascadedPIDController::quatNorm(
    const std::array<double, 4>& q
) {
    double n = std::sqrt(
        q[0] * q[0] +
        q[1] * q[1] +
        q[2] * q[2] +
        q[3] * q[3]
    );

    if (n < 1e-12) {
        return {1.0, 0.0, 0.0, 0.0};
    }

    return {
        q[0] / n,
        q[1] / n,
        q[2] / n,
        q[3] / n
    };
}

std::array<double, 3> CascadedPIDController::quatToEulerSmall(
    const std::array<double, 4>& q
) {
    double qw = q[0];
    double qx = q[1];
    double qy = q[2];
    double qz = q[3];

    double roll = std::atan2(
        2.0 * (qw * qx + qy * qz),
        1.0 - 2.0 * (qx * qx + qy * qy)
    );

    double pitch_arg = 2.0 * (qw * qy - qz * qx);
    pitch_arg = clip(pitch_arg, -1.0, 1.0);

    double pitch = std::asin(pitch_arg);

    double yaw = std::atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz)
    );

    return {roll, pitch, yaw};
}

ControlOutput CascadedPIDController::step(
    const State& state,
    const Reference& ref,
    double dt
) {
    ControlOutput output{};
    output.thrust = 0.0;
    output.torque = {0.0, 0.0, 0.0};

    std::array<double, 3> p = state.position;
    std::array<double, 4> q = quatNorm(state.quaternion);

    std::array<double, 3> v_w{};

    if (!has_prev_position_) {
        v_w = {0.0, 0.0, 0.0};
        has_prev_position_ = true;
    } else {
        double safe_dt = std::max(dt, 1e-6);
        for (int i = 0; i < 3; ++i) {
            v_w[i] = (p[i] - prev_position_[i]) / safe_dt;
        }
    }

    prev_position_ = p;

    double z = p[2];
    double vz = v_w[2];

    double z_ref = ref.position[2];
    double vz_ref = ref.velocity[2];

    double e_z = z_ref - z;
    double e_vz = vz_ref - vz;

    double az_cmd = params_.kz_p * e_z + params_.kz_d * e_vz;

    double T = params_.mass * (params_.gravity + az_cmd);
    T = clip(T, params_.max_thrust[0], params_.max_thrust[1]);

    double x = p[0];
    double vx = v_w[0];

    double x_ref = ref.position[0];
    double vx_ref = ref.velocity[0];

    double e_x = x_ref - x;
    double e_vx = vx_ref - vx;

    double ax_cmd = params_.kx_p * e_x + params_.kx_d * e_vx;

    double pitch_des = clip(
        ax_cmd / params_.gravity,
        -params_.max_tilt,
        params_.max_tilt
    );

    double y = p[1];
    double vy = v_w[1];

    double y_ref = ref.position[1];
    double vy_ref = ref.velocity[1];

    double e_y = y_ref - y;
    double e_vy = vy_ref - vy;

    double ay_cmd = params_.ky_p * e_y + params_.ky_d * e_vy;

    double roll_des = clip(
        -ay_cmd / params_.gravity,
        -params_.max_tilt,
        params_.max_tilt
    );

    std::array<double, 3> euler = quatToEulerSmall(q);

    double roll = euler[0];
    double pitch = euler[1];
    double yaw = euler[2];

    double e_roll = roll_des - roll;
    double e_pitch = pitch_des - pitch;
    double e_yaw = wrapToPi(ref.yaw - yaw);

    double tau_x = p_roll_.step(e_roll, dt);
    double tau_y = p_pitch_.step(e_pitch, dt);
    double tau_z = p_yaw_.step(e_yaw, dt);

    output.thrust = T;

    output.torque[0] = clip(
        tau_x,
        -params_.max_torque[0],
        params_.max_torque[0]
    );

    output.torque[1] = clip(
        tau_y,
        -params_.max_torque[1],
        params_.max_torque[1]
    );

    output.torque[2] = clip(
        tau_z,
        -params_.max_torque[2],
        params_.max_torque[2]
    );

    return output;
}