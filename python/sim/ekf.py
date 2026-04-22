import numpy as np
from .dynamics import quat_norm, quat_to_R

def quat_derivative(q, w_b):
    """
    Quaternion kinematics:
      q_dot = 0.5 * Omega(w_b) * q
    for q = [qw, qx, qy, qz]
    """
    wx, wy, wz = w_b

    Omega = np.array([
        [0.0, -wx, -wy, -wz],
        [wx,  0.0,  wz, -wy],
        [wy, -wz,  0.0,  wx],
        [wz,  wy, -wx,  0.0],
    ])
    return 0.5 * Omega @ q

class EKF:
    def __init__(self, x0, P0, Q, R_gps, R_baro, params):
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = Q.copy()
        self.R_gps = R_gps
        self.R_baro = R_baro
        self.params = params
    '''
    def predict(self, u, dt, wind_w):
        # Nonlinear propagation
        x_pred = step_6dof(self.x, u, self.params, dt, wind_w=wind_w)
        # Simple numerical Jacobian
        def fwrap(xx):
            xx2 = xx.copy()
            xx2[6:10] = quat_norm(xx2[6:10])
            return step_6dof(xx2, u, self.params, dt, wind_w=wind_w)

        F = self._num_jacobian(fwrap, self.x)
        self.x = x_pred
        self.x[6:10] = quat_norm(self.x[6:10])
        self.P = F @ self.P @ F.T + self.Q
    '''
    def predict(self, accel_meas_b, gyro_meas_b, dt):
        """
        IMU-driven EKF prediction step.

        State:
        x = [p_w(3), v_b(3), q(4), w_b(3)]

        IMU inputs:
        accel_meas_b : specific force in body frame
        gyro_meas_b  : angular velocity in body frame
        """
        g = self.params.get("g", 9.81)

        def f_imu(xx):
            p = xx[0:3]
            v_b = xx[3:6]
            q = quat_norm(xx[6:10])
            w_b = gyro_meas_b.copy()   # gyro drives angular-rate state directly

            R_bw = quat_to_R(q)        # body -> world
            g_w = np.array([0.0, 0.0, -g])
            g_b = R_bw.T @ g_w

            # accelerometer measures specific force:
            # f_b = v_dot_b + w x v - g_b  =>  v_dot_b = f_b + g_b - w x v
            a_b = accel_meas_b + g_b

            # position kinematics in world frame
            p_dot = R_bw @ v_b

            # body-frame translational dynamics (simplified)
            # v_dot_b = a_b - w x v
            v_dot_b = a_b - np.cross(w_b, v_b)

            # quaternion kinematics
            q_dot = quat_derivative(q, w_b)

            # angular-rate state just tracks gyro measurement
            w_dot_b = np.zeros(3)

            x_dot = np.hstack([p_dot, v_dot_b, q_dot, w_dot_b])
            x_next = xx + dt * x_dot
            x_next[6:10] = quat_norm(x_next[6:10])
            x_next[10:13] = gyro_meas_b.copy()

            return x_next

        x_pred = f_imu(self.x)
        F = self._num_jacobian(f_imu, self.x)

        self.x = x_pred
        self.x[6:10] = quat_norm(self.x[6:10])
        self.P = F @ self.P @ F.T + self.Q

    def update_gps(self, p_meas, v_meas):
        # Measurement: z = [p_w, v_w]
        p = self.x[0:3]
        v_b = self.x[3:6]
        q = self.x[6:10]
        R = quat_to_R(q)
        v_w = R @ v_b

        z_hat = np.hstack([p, v_w])
        z = np.hstack([p_meas, v_meas])

        H = np.zeros((6, len(self.x)))
        H[0:3, 0:3] = np.eye(3)

        # d(v_w)/d(v_b) = R
        H[3:6, 3:6] = R

        # ignore attitude coupling in H for starter EKF (works ok with frequent GPS)
        y = z - z_hat
        S = H @ self.P @ H.T + self.R_gps
        S = S + 1e-9 * np.eye(S.shape[0])
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.x[6:10] = quat_norm(self.x[6:10])
        I = np.eye(len(self.x))
        self.P = (I - K @ H) @ self.P

    def update_baro(self, z_meas):
        # Measurement: altitude = p_z (world)
        H = np.zeros((1, len(self.x)))
        H[0, 2] = 1.0
        z_hat = np.array([self.x[2]])
        y = np.array([z_meas]) - z_hat
        S = H @ self.P @ H.T + self.R_baro
        S = S + 1e-9 * np.eye(S.shape[0])
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y).reshape(-1)
        self.x[6:10] = quat_norm(self.x[6:10])
        I = np.eye(len(self.x))
        self.P = (I - K @ H) @ self.P

    @staticmethod
    def _num_jacobian(f, x, eps=1e-6):
        n = len(x)
        fx = f(x)
        m = len(fx)
        J = np.zeros((m, n))
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = eps
            J[:, i] = (f(x + dx) - f(x - dx)) / (2*eps)
        return J