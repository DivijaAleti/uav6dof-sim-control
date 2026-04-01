import numpy as np
from .dynamics import quat_norm, quat_to_R, step_6dof

class EKF:
    def __init__(self, x0, P0, Q, R_gps, R_baro, params):
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = Q.copy()
        self.R_gps = R_gps
        self.R_baro = R_baro
        self.params = params

    def predict(self, u, dt, wind_w):
        # Nonlinear propagation
        x_pred = step_6dof(self.x, u, self.params, dt, wind_w=wind_w)
        # Simple numerical Jacobian (robust for a starter project)
        F = self._num_jacobian(lambda xx: step_6dof(xx, u, self.params, dt, wind_w=wind_w), self.x)
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