import numpy as np
from sim.aero import aero_drag_force
from sim.dynamics import quat_to_R
from sim.dynamics import step_6dof
from sim.ekf import EKF
from sim.controllers import CascadedPID

def main():
    dt = 0.005
    T_end = 20.0
    steps = int(T_end/dt)

    params = {
        "m": 2.0,
        "I": np.diag([0.05, 0.05, 0.09]),
        "g": 9.80665,
        "rho": 1.225,
        "CdA": np.array([0.06, 0.06, 0.10]),
        "aero_force": aero_drag_force,
    }

    # Controller gains
    ctrl_params = {
    "m": params["m"],
    "g": params["g"],

    "pos_kp": [0.2, 0.2, 0.25],
    "pos_ki": [0.0, 0.0, 0.0],
    "pos_kd": [0.05, 0.05, 0.08],

    "vel_kp": [0.4, 0.4, 0.45],
    "vel_ki": [0.0, 0.0, 0.0],
    "vel_kd": [0.03, 0.03, 0.04],

    "roll_kp": 0.3,
    "roll_ki": 0.0,
    "roll_kd": 0.02,

    "pitch_kp": 0.3,
    "pitch_ki": 0.0,
    "pitch_kd": 0.02,

    "yaw_kp": 0.3,
    "yaw_ki": 0.0,
    "yaw_kd": 0.02,

    "tau_lim": [0.03, 0.08, 0.04],
    "T_lim": [0.0, 25.0],
    "max_tilt": np.deg2rad(5.0),
    }
    
    ctrl = CascadedPID(ctrl_params)

    # True state
    x = np.zeros(13)
    x[6] = 1.0  # qw=1
    x[0:3] = np.array([0,0,0])

    # EKF init
    x0 = x.copy()
    P0 = np.eye(13) * 0.2
    Q = np.eye(13) * 1e-3
    R_gps = np.eye(6) * 0.2
    R_baro = np.eye(1) * 0.05
    ekf = EKF(x0, P0, Q, R_gps, R_baro, params)

    # Reference
    ref = {"p": np.array([5.0, 0.0, -3.0]), "yaw": 0.0}

    wind_w = np.array([2.0, 0.0, 0.0])

    for k in range(steps):
        x_ctrl = x.copy()
        x_ctrl[0:6] = ekf.x[0:6] #use estimated position and velocity only
        u = ctrl.step(x_ctrl, ref, dt, quat_to_R)
        # Propagate truth
        x = step_6dof(x, u, params, dt, wind_w=wind_w)

        # Fake sensors
        gps_period = int(0.1/dt)
        baro_period = int(0.05/dt)

        ekf.predict(u, dt, wind_w)

        if k % gps_period == 0:
            p_meas = x[0:3] + np.random.randn(3)*0.3
            # world vel from body vel:
            R = quat_to_R(x[6:10])
            v_w = R @ x[3:6]
            v_meas = v_w + np.random.randn(3)*0.3
            ekf.update_gps(p_meas, v_meas)

        if k % baro_period == 0:
            z_meas = x[2] + np.random.randn()*0.2
            ekf.update_baro(z_meas)

    print("Final true position:", x[0:3])
    print("Final est position :", ekf.x[0:3])

if __name__ == "__main__":
    main()