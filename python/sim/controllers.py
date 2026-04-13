import numpy as np

def sat(x, lo, hi):
    return np.minimum(np.maximum(x,lo),hi)

class PID:
    def __init__(self,kp,ki,kd,i_lim=1e9):
        self.kp = np.array(kp, dtype=float)
        self.ki = np.array(ki, dtype=float)
        self.kd = np.array(kd, dtype=float)
        self.i = np.zeros_like(self.ki)
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
        self.i = np.clip(self.i, -self.i_lim, self.i_lim)
        return self.kp*e + self.ki*self.i + self.kd*de
    
def quat_to_euler_small(q):
    # q = [qw, qx, qy, qz]
    qw, qx, qy, qz = q

    roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
    pitch_arg = 2*(qw*qy - qz*qx)
    pitch_arg = np.clip(pitch_arg, -1.0, 1.0)
    pitch = np.arcsin(pitch_arg)
    yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

    return roll, pitch, yaw

class CascadedPID:
    """
    Small-angle hover controller.
    Output: u = [T, tau_x, tau_y, tau_z]
    Assumes near-hover / moderate tilt.
    """

    def __init__(self, params):
        self.p_pos = PID(params["pos_kp"], params["pos_ki"], params["pos_kd"], i_lim=2.0)
        self.p_vel = PID(params["vel_kp"], params["vel_ki"], params["vel_kd"], i_lim=2.0)

        self.p_roll  = PID([params["roll_kp"]],  [params["roll_ki"]],  [params["roll_kd"]],  i_lim=1.0)
        self.p_pitch = PID([params["pitch_kp"]], [params["pitch_ki"]], [params["pitch_kd"]], i_lim=1.0)
        self.p_yaw   = PID([params["yaw_kp"]],   [params["yaw_ki"]],   [params["yaw_kd"]],   i_lim=1.0)

        self.m = params["m"]
        self.g = params.get("g", 9.80665)
        self.tau_lim = np.array(params.get("tau_lim", [0.03, 0.08, 0.04]), dtype=float)
        self.T_lim = np.array(params.get("T_lim", [0.0, 30.0]), dtype=float)

        self.max_tilt = params.get("max_tilt", np.deg2rad(5.0))
        self.kz_p = params.get("kz_p", 1.0)
        self.kz_d = params.get("kz_d", 1.5)
        self.kx_p = params.get("kx_p", 0.10)
        self.kx_d = params.get("kx_d", 0.20)
        self.ky_p = params.get("ky_p", 0.15)
        self.ky_d = params.get("ky_d", 0.20)
        self.p_prev = None

    def step(self, x_hat, ref, dt, R_from_quat):
        p = x_hat[0:3]
        v_b = x_hat[3:6]
        q = x_hat[6:10]
        R = R_from_quat(q)
        if self.p_prev is None:
            v_w = np.zeros(3)
        else:
            v_w = (p - self.p_prev) / max(dt, 1e-6)

        self.p_prev = p.copy()
        #v_w = R @ v_b

        p_ref = ref["p"]
        v_ref = ref.get("v", np.zeros(3))
        yaw_ref = ref.get("yaw", 0.0)
        """
        # Outer loops
        e_p = p_ref - p
        e_p[0] = 0.0
        e_p[1] = 0.0
        v_cmd = v_ref + self.p_pos.step(e_p, dt)
        v_cmd[0] = 0.0
        v_cmd[1] = 0.0

        e_v = v_cmd - v_w
        e_v[0] = 0.0
        e_v[1] = 0.0
        a_cmd = self.p_vel.step(e_v, dt)
        a_cmd[0] = 0.0
        a_cmd[1] = 0.0
        """
        # trying z-only control
        z = p[2]
        vz = v_w[2]

        z_ref = p_ref[2]
        vz_ref = v_ref[2] if "v" in ref else 0.0

        e_z = z_ref - z
        e_vz = vz_ref - vz
        

        # simple PD for vertical acceleration command
        az_cmd = self.kz_p * e_z + self.kz_d * e_vz

        # Vertical thrust
        #az_cmd = a_cmd[2]
        T = self.m * (self.g + az_cmd)
        T = float(np.clip(T, self.T_lim[0], self.T_lim[1]))

        # Small-angle lateral acceleration -> desired roll/pitch
        # These signs may depend on your exact frame convention, but this is a much easier place to debug.
        # x control only
        x_pos = p[0]
        vx = v_w[0]
        x_ref = p_ref[0]
        vx_ref = v_ref[0]

        e_x = x_ref - x_pos
        e_vx = vx_ref - vx
    

        ax_cmd = self.kx_p * e_x + self.kx_d * e_vx
        #ax_cmd = 0.0
        #ay_cmd = a_cmd[1]

        pitch_des = np.clip(ax_cmd / self.g, -self.max_tilt, self.max_tilt)

        #y-control only
        y_pos = p[1]
        vy = v_w[1]

        y_ref = p_ref[1]
        vy_ref = v_ref[1]

        e_y = y_ref - y_pos
        e_vy = vy_ref - vy
        

        ay_cmd = self.ky_p * e_y + self.ky_d * e_vy
        roll_des  = np.clip(-ay_cmd / self.g, -self.max_tilt, self.max_tilt)
        #pitch_des = 0.0
        #roll_des = 0.0

        roll, pitch, yaw = quat_to_euler_small(q)

        e_roll = np.array([roll_des - roll])
        e_pitch = np.array([pitch_des - pitch])

        # wrap yaw error to [-pi, pi]
        yaw_err = yaw_ref - yaw
        yaw_err = (yaw_err + np.pi) % (2*np.pi) - np.pi
        e_yaw = np.array([yaw_err])

        tau_x = self.p_roll.step(e_roll, dt)[0]
        tau_y = self.p_pitch.step(e_pitch, dt)[0]
        tau_z = self.p_yaw.step(e_yaw, dt)[0]

        tau = np.array([tau_x, tau_y, tau_z])
        tau = np.clip(tau, -self.tau_lim, self.tau_lim)

        return np.array([T, tau[0], tau[1], tau[2]])
    

"""  
class CascadedPID:
    
    Output: u = [T, tau_x, tau_y, tau_z]
    Simple architecture:
      position->velocity command
      velocity->tilt + thrust
      attitude->body rate->torque (simplified: attitude->torque)
    
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
        y_b_des_w = np.cross(z_b_des_w,x_c_w)
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
"""