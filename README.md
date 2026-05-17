# UAV 6-DOF Simulation and Control

## Overview

This project implements a full 6-DOF UAV simulation, control, and state estimation framework. It models nonlinear rigid-body dynamics and uses feedback control with EKF-based state estimation to track a desired 3D position.

The system integrates **rigid body dynamics**, **state estimation**, and **control**, similar to real-world UAV and autonomy stacks.

---

## Key Features

- Nonlinear 6-DOF rigid body dynamics (13-state system)
- Quaternion-based attitude representation
- Aerodynamic drag + disturbance modeling (wind)
- Cascaded PID controller (position → attitude → torque)
- Extended Kalman Filter (EKF) with IMU-driven prediction
- Sensor models:
  - IMU (accelerometer + gyro)
  - GPS (position + velocity)
  - Barometer (altitude)
- Position tracking with closed-loop control using **estimated state**
- Control saturation and rate limiting
- **C++ controller integrated with Python (pybind11)**
- Estimated state vs. true state comparison
- Python vs. C++ controller comparison

## System Architecture

### Dynamics
- 13-state nonlinear UAV model:
- Body-frame dynamics + quaternion attitude
- Forces: gravity, thrust, quadratic drag
- Wind modeled in world frame

### Control
- Cascaded PID:
- Thrust from vertical control, roll/pitch from lateral control
- Yaw control loop
- Uses **EKF-estimated state**
- Includes saturation + rate limiting
- Implemented in Python and C++ (pybind11)

### Estimation (EKF)
- IMU-driven prediction (accelerometer + gyro)
- Updates:
    - GPS → position, velocity
    - Barometer → altitude
- Numerical Jacobian + quaternion normalization

### Sensors
- IMU, GPS, barometer (noise + different rates)

### Simulation Loop
- Propagate dynamics → generate sensor data  
- EKF predict + update  
- Compute control from estimated state  
- Apply rate limits and log results

## Project Structure
```
uav6dof-sim-control/
|-- cpp/cpp/
||-- bindings/controller_binding.cpp
||-- include/controller.hpp
||-- src/controller.cpp
||-- tests/test_controller.cpp
||-- CMakeLists.txt
|
|-- python/
||-- run_sim.py
||-- sim/
|||-- aero.py
|||-- controllers.py
|||-- dynamics.py
|||-- ekf.py
|||-- plots.py
|||-- sensors.py 
|      
|-- .gitignore
|-- README.md
```

## How to Run
1. Clone the repository
```
git clone https://github.com/DivijaAleti/uav6dof-sim-control.git
cd uav6dof-sim-control
```
2. Install dependencies
```
pip install numpy matplotlib
python3 -m pip install pybind11
```
3. Build C++ module
```
cd cpp/cpp
mkdir -p build && cd build
cmake .. && make
```
4. Run simulation using Python controller<br>
In python/run_sim.py, uncomment line 170 and comment out line 173. Then, run:
```
python python/run_sim.py
```
5. Run simulation using C++ controller<br>
In python/run_sim.py, uncomment line 173 and comment out line 170. Then, run:
```
python python/run_sim.py
```

## Example Results
Python controller output:
```
Final true position: [ 5.559, -0.049, -2.987]
Final est position : [ 5.523, -0.064, -3.090]
Final error norm   : 0.11 m
```
C++ controller output:
```
Final true position: [5.503, 0.002, -2.955]
Final est position : [5.534, 0.003, -3.081]
Final error norm   : 0.13 m
```

## Results Analysis
### Tracking Performance
- Both controllers successfully track the reference:
```
Reference: [5.0, 0.0, -3.0]
```
- Final position error is ~0.1-0.3 m, indicating stable closed-loop performance.
### Estimation Behavior
- EKF tracks position with small bounded error (~0.1-0.2 m)
- Noise in y-direction is higher due to:
    - No direct lateral measurement (GPS noise dominates)
    - Small-angle lateral control assumptions
### Key Observations
- Differences arise due to:
    - Numerical implementation differences (Python vs C++)
    - State handling and floating-point precision
    - Minor differences in velocity estimation (finite difference)
- Both implementations are consistent and stable, validating controller correctness

## Key Learnings
- State estimation quality directly impacts control performance
- IMU-driven EKF enables high-frequency prediction between GPS updates
- Frame transformations (body to world) are critical
- Controller tuning significantly affects tracking vs oscillation tradeoff
- Rate limiting improves realism and stability
- C++ integration enables transition from simulation to deployable control

## Future Work
- Use RK4 integration for simulation of nonlinear dynamics 
- Implement trajectory tracking (not just point stabilization)
- Run Monte-Carlo simulations
- Add RMSE and performance metrics
- IMU bias estimation in EKF

## Author
Divija Aleti<br>
Aerospace Engineer | Controls | Autonomy