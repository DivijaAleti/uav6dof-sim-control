import numpy as np
from sim.aero import aero_drag_force
from sim.dynamics import quat_to_R
from sim.dynamics import step_6dof
from sim.ekf import EKF
from sim.controllers import CascadedPID
from sim.sensors import GPSSensor, BarometerSensor, IMUSensor
from sim.plots import plot_trajectory_3d, plot_position_tracking, plot_position_error, plot_control_inputs, show_all

def main():
    dt = 0.005
    T_end = 20.0
    steps = int(T_end/dt)

    # Time history
    t_hist = np.zeros(steps)

    # True state history
    p_true_hist = np.zeros((steps, 3))
    v_true_hist = np.zeros((steps, 3))
    q_true_hist = np.zeros((steps, 4))
    w_true_hist = np.zeros((steps, 3))

    # Estimated state history
    p_est_hist = np.zeros((steps, 3))
    v_est_hist = np.zeros((steps, 3))
    q_est_hist = np.zeros((steps, 4))
    w_est_hist = np.zeros((steps, 3))

    # Reference history
    p_ref_hist = np.zeros((steps, 3))

    # Control history
    u_hist = np.zeros((steps, 4))

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

    gps = GPSSensor(pos_sigma=0.3,vel_sigma=0.3,rate_hz=10.0)
    baro = BarometerSensor(z_sigma=0.2,rate_hz=20.0)
    imu = IMUSensor(
        accel_sigma=0.03,
        gyro_sigma=0.003,
        accel_bias=np.zeros(3), #np.array([0.02,-0.01,0.03]),
        gyro_bias=np.zeros(3), #np.array([0.002,-0.001,0.0015]),
        rate_hz=1.0/dt,
        g=params["g"]
    )

    # Reference
    ref = {"p": np.array([5.0, 0.0, -3.0]), "yaw": 0.0}

    wind_w = np.array([2.0, 0.0, 0.0])

    for k in range(steps):
        t = k * dt

        x_ctrl = x.copy()
        x_ctrl[0:6] = ekf.x[0:6] #use estimated position and velocity only
        u = ctrl.step(x_ctrl, ref, dt, quat_to_R)

        # Save previous truth state for IMU finite-difference acceleration
        x_prev = x.copy()

        # Propagate truth
        x = step_6dof(x, u, params, dt, wind_w=wind_w)

        # Fake sensors
        #gps_period = int(0.1/dt)
        #baro_period = int(0.05/dt)

        # ekf.predict(u, dt, wind_w)

        # IMU prediction step
        if imu.ready(t):
            accel_meas_b, gyro_meas_b = imu.measure(x_prev,x,dt,t)
            ekf.predict(accel_meas_b,gyro_meas_b,dt)

        # GPS update when a new GPS measurement is available
        if gps.ready(t):
            p_meas, v_meas = gps.measure(x,t)
            ekf.update_gps(p_meas, v_meas)

        # Barometer update when a new barometer measurement is available
        if baro.ready(t):
            z_meas = baro.measure(x,t)
            ekf.update_baro(z_meas)

        # Log histories
        t_hist[k] = t

        p_true_hist[k, :] = x[0:3]
        v_true_hist[k, :] = x[3:6]
        q_true_hist[k, :] = x[6:10]
        w_true_hist[k, :] = x[10:13]

        p_est_hist[k, :] = ekf.x[0:3]
        v_est_hist[k, :] = ekf.x[3:6]
        q_est_hist[k, :] = ekf.x[6:10]
        w_est_hist[k, :] = ekf.x[10:13]

        p_ref_hist[k, :] = ref["p"]

        u_hist[k, :] = u


        '''
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
        '''

    pos_err = p_est_hist - p_true_hist
    vel_err = v_est_hist - v_true_hist

    pos_err_norm = np.linalg.norm(pos_err, axis=1)

    print("Final true position:", x[0:3])
    print("Final est position :", ekf.x[0:3])
    print("Final position error norm:", pos_err_norm[-1])

    # Plot using external module
    fig1 = plot_trajectory_3d(p_true_hist, p_est_hist, p_ref_hist)
    fig2 = plot_position_tracking(t_hist, p_true_hist, p_est_hist, p_ref_hist)
    fig3 = plot_position_error(t_hist, p_true_hist, p_est_hist)
    fig4 = plot_control_inputs(t_hist, u_hist)

    show_all()

if __name__ == "__main__":
    main()