import numpy as np
from .dynamics import quat_to_R

class GPSSensor:
    """
    Simple GPS sensor model.
    Measures:
      - position in world frame
      - velocity in world frame

    Includes:
      - white noise
      - configurable update period
    """
    def __init__(self, pos_sigma=0.3, vel_sigma=0.3, rate_hz=10.0):
        self.pos_sigma = float(pos_sigma)
        self.vel_sigma = float(vel_sigma)
        self.dt_meas = 1.0 / float(rate_hz)
        self.t_last = -np.inf

    def ready(self, t):
        return (t - self.t_last) >= self.dt_meas - 1e-12

    def measure(self, x_true, t):
        """
        x_true = [p(3), v_b(3), q(4), w_b(3)]
        Returns:
          p_meas: noisy world position
          v_meas: noisy world velocity
        """
        p_true = x_true[0:3]
        v_b_true = x_true[3:6]
        q_true = x_true[6:10]

        R_bw = quat_to_R(q_true)      # body -> world
        v_w_true = R_bw @ v_b_true

        p_meas = p_true + np.random.randn(3) * self.pos_sigma
        v_meas = v_w_true + np.random.randn(3) * self.vel_sigma

        self.t_last = t
        return p_meas, v_meas


class BarometerSensor:
    """
    Simple barometer / altitude sensor model.
    Measures world z-position.
    """
    def __init__(self, z_sigma=0.2, rate_hz=20.0):
        self.z_sigma = float(z_sigma)
        self.dt_meas = 1.0 / float(rate_hz)
        self.t_last = -np.inf

    def ready(self, t):
        return (t - self.t_last) >= self.dt_meas - 1e-12

    def measure(self, x_true, t):
        z_true = x_true[2]
        z_meas = z_true + np.random.randn() * self.z_sigma
        self.t_last = t
        return z_meas
    
class IMUSensor:
    """
    Simple IMU sensor model.

    Outputs:
      - accelerometer measurement in body frame: specific force [m/s^2]
      - gyro measurement in body frame: angular rate [rad/s]

    Notes:
      - accelerometer measures specific force, not inertial acceleration
      - this starter version includes white noise + constant bias
      - no bias estimation yet in EKF
    """
    def __init__(
        self,
        accel_sigma=0.08,
        gyro_sigma=0.01,
        accel_bias=None,
        gyro_bias=None,
        rate_hz=100.0,
        g=9.81,
    ):
        self.accel_sigma = float(accel_sigma)
        self.gyro_sigma = float(gyro_sigma)
        self.accel_bias = np.zeros(3) if accel_bias is None else np.array(accel_bias, dtype=float)
        self.gyro_bias = np.zeros(3) if gyro_bias is None else np.array(gyro_bias, dtype=float)
        self.dt_meas = 1.0 / float(rate_hz)
        self.t_last = -np.inf
        self.g = float(g)

    def ready(self, t):
        return (t - self.t_last) >= self.dt_meas - 1e-12

    def measure(self, x_prev, x_true, dt, t):
        """
        Use two consecutive true states to approximate translational acceleration.

        State format:
          x = [p_w(3), v_b(3), q(4), w_b(3)]

        Returns:
          accel_meas_b : body-frame specific force
          gyro_meas_b  : body-frame angular velocity
        """
        v_b_prev = x_prev[3:6]
        v_b_true = x_true[3:6]
        q_true = x_true[6:10]
        w_b_true = x_true[10:13]

        # Approximate body-frame translational acceleration dv_b/dt
        v_dot_b_approx = (v_b_true - v_b_prev) / dt

        # Accelerometer measures specific force:
        # f_b = v_dot_b + w_b x v_b - g_b
        R_bw = quat_to_R(q_true)   # body -> world
        g_w = np.array([0.0, 0.0, -self.g])
        g_b = R_bw.T @ g_w

        accel_true_b = v_dot_b_approx + np.cross(w_b_true,v_b_true) - g_b
        gyro_true_b = w_b_true

        accel_meas_b = (
            accel_true_b
            + self.accel_bias
            + np.random.randn(3) * self.accel_sigma
        )
        gyro_meas_b = (
            gyro_true_b
            + self.gyro_bias
            + np.random.randn(3) * self.gyro_sigma
        )

        self.t_last = t
        return accel_meas_b, gyro_meas_b