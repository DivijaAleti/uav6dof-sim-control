import numpy as np

def sat(x, lo, hi):
    return np.minimum(np.maximum(x,lo),hi)

class PID:
    def __init__(self,kp,ki,kd,i_lim=1e9):
        self.kp = np.array(kp, dtype=float)
        self.ki = np.array(ki, dtype=float)
        self.kd = np.array(kd, dtype=float)
        self.i = np.zeros_like(self.kp)
        self.i_lim = i_lim
        self.prev_e = None

    def step(self,e,dt):
        e = np.array(e, dtype=float)
        if self.prev_e is None:
            de = np.zeros_like(e)
        else:
            de = (e - self.prev_e) / max(dt, 1e-9)
        self.prev_e = e

        self.i += e * dt
        self.i = sat(self.i, -self.i_lim, self.i_lim)
        return self.kp*e + self.ki*self.i + self.kd*de
    
class CascadedPID:
    """
    Output: u = [T, tau_x, tau_y, tau_z]
    Simple architecture:
      position->velocity command
      velocity->tilt + thrust
      attitude->body rate->torque (simplified: attitude->torque)
    """
    def __init__(self, params):
        self.p_pos = PID(params["pos_kp"], params["pos_ki"], params["pos_kd"], i_lim=5.0)
        self.p_vel = PID(params["vel_kp"], params["vel_ki"], params["vel_kd"], i_lim=5.0)
        self.p_att = PID(params["att_kp"], params["att_ki"], params["att_kd"], i_lim=2.0)
        self.p_yaw = PID([params["yaw_kp"]],[params["yaw_ki"]],[params["yaw_kd"]], i_lim=1.0)

        self.m = params["m"]
        self.g = params.get("g", 9.80665)
        self.tau_lim = np.array(params.get("tau_lim", [2.0, 2.0, 1.0]))
        self.T_lim = params.get("T_lim", [0.0, 2.5*self.m*self.g])

    def step(self, x_hat, ref, dt, R_from_quat):
        # x_hat: [p(3), v_b(3), q(4), w(3)] but we will use p (world) and v (world approx)
        p = x_hat[0:3]
        v_b = x_hat[3:6]
        q = x_hat[6:10]
        R = R_from_quat(q)

        # Convert body velocity to world for outer loop
        v_w = R @ v_b

        p_ref = ref["p"]
        v_ref = ref.get("v", np.zeros(3))
        yaw_ref = ref.get("yaw", 0.0)

        # Position -> desired velocity
        e_p = p_ref - p
        v_cmd = v_ref + self.p_pos.step(e_p, dt)

        # Velocity -> desired acceleration in world
        e_v = v_cmd - v_w
        a_cmd_w = self.p_vel.step(e_v, dt)

        # Desired thrust direction = g + a_cmd
        g_w = np.array([0.0, 0.0, -self.g])
        a_total = a_cmd_w - g_w  # want to counter gravity
        # Thrust magnitude
        T = self.m * np.linalg.norm(a_total)
        T = float(np.clip(T, self.T_lim[0], self.T_lim[1]))

        # Desired body z axis in world (thrust points +z_body -> world)
        if np.linalg.norm(a_total) < 1e-6:
            z_b_des_w = np.array([0.0, 0.0, 1.0])
        else:
            z_b_des_w = a_total / np.linalg.norm(a_total)

        # Build a desired rotation from z axis + yaw (simple)
        c, s = np.cos(yaw_ref), np.sin(yaw_ref)
        x_c_w = np.array([c, s, 0.0])
        y_b_des_w = np.cross(z_b_des_w, x_c_w)
        y_norm = np.linalg.norm(y_b_des_w)
        if y_norm < 1e-6:
            y_b_des_w = np.array([0.0, 1.0, 0.0])
        else:
            y_b_des_w = y_b_des_w / y_norm
        x_b_des_w = np.cross(y_b_des_w, z_b_des_w)
        x_b_des_w = x_b_des_w / max(np.linalg.norm(x_b_des_w), 1e-9)
        R_des = np.column_stack([x_b_des_w, y_b_des_w, z_b_des_w])

        # Attitude error -> torque (small-angle approx)
        R_err = R_des.T @ R
        # vee( R_err - R_err^T ) / 2
        e_R_mat = 0.5 * (R_des.T @ R - R.T @ R_des)
        e_R = np.array([
            e_R_mat[2,1],
            e_R_mat[0,2],
            e_R_mat[1,0],
        ])
        #e_R = 0.5 * np.array([
        #    R_err[2,1] - R_err[1,2],
        #    R_err[0,2] - R_err[2,0],
        #    R_err[1,0] - R_err[0,1],
        #])
        tau = -self.p_att.step(e_R, dt)
        tau = np.clip(tau, -self.tau_lim, self.tau_lim)

        # Optional yaw torque correction (if you want explicit yaw control)
        # left minimal here because yaw is implicitly in R_des.

        return np.array([T, tau[0], tau[1], tau[2]])