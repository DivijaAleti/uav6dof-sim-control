import numpy as np

def skew(w):
    wx, wy, wz = w
    return np.array([0, -wz, wy],
                     [wz, 0, -wx],
                     [-wy, wx, 0])

def quat_norm(q):
    return q / np.linalg.norm(q)

def quat_to_R(q):
    # body -> inertial
    qw, qx, qy, qz = q
    R = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),   1-2*(qx*qx+qy*qy)]
    ])
    return R

def quat_omega_matrix(w):
    wx, wy, wz = w
    return np.array([
        [0, -wx, -wy, -wz],
        [wx, 0, wz, -wy],
        [wy, -wz, 0, wx],
        [wz, wy, -wx, 0]
    ])

def step_6dof(x,u,params,dt,wind_w=np.zeros(3)):
    """
    x = [p(3), v_b(3), q(4), w_b(3)] => 13 states
    u = [T, tau_b(3)] => 4 inputs
    Assumption: Thrust (T) acts along +z_body axis
    """
    m = params["m"]
    I = params["I"]
    Iinv = np.linalg.inv(I)
    g = params.get("g", 9.80665)

    p = x[0:3]
    v = x[3:6]
    q = x[6:10]
    w = x[10:13]

    q = quat_norm(q)
    R = quat_to_R(q)

    T, tx, ty, tz = u
    F_thrust = np.array([0.0, 0.0, T])
    tau = np.array([tx, ty, tz])

    # Drag in body frame from aero model
    F_drag = params["aero_force"](v,q,wind_w,params)

    # Gravity in body frame: R^T * [0, 0, -g]
    g_w = np.array([0.0, 0.0, -g])
    g_b = R.T @ g_w

    F_b = F_thrust + F_drag + m * g_b

    #Dynamics
    p_dot = R @ v
    v_dot = (1.0/m) * F_b - np.cross(w,v)
    q_dot = 0.5 * (quat_omega_matrix(w) @ q)
    w_dot = Iinv @ (tau - np.cross(w, I @ w))

    # Euler step (swap for RK4 later if needed)
    p2 = p + dt * p_dot
    v2 = v + dt * v_dot
    q2 = quat_norm(q + dt * q_dot)
    w2 = w + dt * w_dot

    return np.hstack([p2, v2, q2, w2])