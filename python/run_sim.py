import numpy as np
from sim.aero import aero_drag_force
from sim.dynamics import quat_to_R
from sim.ekf import EKF
from sim.controllers import CascadedPID

def main():
    dt = 0.01
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

    # Controller gains (starter)
    ctrl_params = {
        "m": params["m"], "g": params["g"],
        "pos_kp":[1.2,1.2,2.0], "pos_ki":[0,0,0.2], "pos_kd":[0.4,0.4,0.6],
        "vel_kp":[2.5,2.5,3.0], "vel_ki":[0,0,0.2], "vel_kd":[0.2,0.2,0.3],
        "att_kp":[6.0,6.0,1.5], "att_ki":[0,0,0], "att_kd":[0.3,0.3,0.1],
        "yaw_kp":1.0,"yaw_ki":0.0,"yaw_kd":0.1,
        "tau_lim":[2.0,2.0,1.0],
        "T_lim":[0.0, 60.0],
    }
    ctrl = CascadedPID(ctrl_params)

    # True state
    x = np.zeros(13)
    x[6] = 1.0  # qw=1
    x[0:3] = np.array([0,0,0])

    # EKF init
    x0 = x.copy()
    P0 = np.eye(13) * 0.2
    Q = np.eye(13) * 1e-4
    R_gps = np.eye(6) * 0.5
    R_baro = np.eye(1) * 0.3
    ekf = EKF(x0, P0, Q, R_gps, R_baro, params)

    # Reference
    ref = {"p": np.array([5.0, 0.0, -3.0]), "yaw": 0.0}

    wind_w = np.array([2.0, 0.0, 0.0])

    for k in range(steps):
        u = ctrl.step(ekf.x, ref, dt, quat_to_R)

        # Propagate truth
        from sim.dynamics import step_6dof
        x = step_6dof(x, u, params, dt, wind_w=wind_w)

        # Fake sensors (simple)
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