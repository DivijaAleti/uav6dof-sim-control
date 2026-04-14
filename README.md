# UAV 6-DOF Simulation and Control

## Overview:

This project implements a full 6-degree-of-freedom (6-DOF) simulation and control framework for a UAV. It models the nonlinear dynamics of the vehicle and applies feedback control to track a desired position in 3D space.


The goal of this project is to build an understanding of **rigid body dynamics**, **state estimation**, and **control**, similar to what is used in real-world autonomy systems.

## Key Features:
- Full nonlinear 6-DOF rigid body dynamics
- Rotation handled using quaternions
- Modular simulation structure (dynamics, control, estimation)
- Position tracking with feedback control
- Simulation under disturbances (e.g. wind)
- Estimated state vs. true state comparison

## System Architecture:
The simulation is structures into key components:
- Dynamics Model
    - Translational and rotational equations of motion
    - Quaternion kinematics from infinitesimal rotation
    - Forces: Gravity + Quadratic Drag + Thrust (control input)
- Controller
    - Cascaded PID control
    - Position control loop (outer loop)
    - Attitude control loop (inner loop)
    - Tracks reference position in 3D
- Estimator
    - Extended Kalman filter (EKF)
    - Predicts state estimate and uncertainty covariance
    - Updates using GPS and barometer measurements
- Simulation Loop
    - Time integration of dynamics
    - Controller + estimator update at each step

## Project Structure:
```
uav6dof-sim-control/
|-- python/        
|-- .gitignore
|-- README.md
```

## How to Run:
1. Clone the repository
```
git clone https://github.com/DivijaAleti/uav6dof-sim-control.git
cd uav6dof-sim-control
```
2. Install dependencies
```
pip install numpy
```
3. Run simulation
```
python python/run_sim.py
```

## Example Results:
Typical output:
```
Final true position: [ 5.23556097  0.01588853 -3.116533  ]
Final est position : [ 5.1235101  -0.19631099 -3.07746272]
```

The controller is able to drive the UAV close to the reference:
```
Reference: [5.0, 0.0, -3.0]
```

## Key Learnings:
- Importance of frame transformations (world <-> body)
- Sensitivity of control to state estimation errors
- Effect of disturbances on tracking performance
- Effect of controller gains on performance

## Future Work:
- Add realistic sensor models (IMU, GPS)
- Add EKF-based sensor fusion
- Use RK4 integration for simulation of nonlinear dynamics 
- Implement trajectory tracking (not just point stabilization)
- Run Monte-Carlo simulations
- Port full pipeline to C++ for real-time performance

## Author:
Divija Aleti<br>
Aerospace Engineer | Controls | Autonomy