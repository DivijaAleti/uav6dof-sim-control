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
Python controller output:<br>
<img width="80" height="60" alt="Python_1" src="https://github.com/user-attachments/assets/0fd17099-481b-4922-b2f2-9a4d356e3526" />
<img width="80" height="60" alt="Python_2" src="https://github.com/user-attachments/assets/30d5edb9-517f-4601-8d49-fefeb3c386cd" />
<img width="80" height="60" alt="Python_3" src="https://github.com/user-attachments/assets/4e89cf10-09d5-4d18-8f2a-a2884df3ad39" />
<img width="80" height="60" alt="Python_4" src="https://github.com/user-attachments/assets/2395b3b6-26c8-46ab-905b-b53d44f24010" />
```
Final true position: [ 5.559, -0.049, -2.987]
Final est position : [ 5.523, -0.064, -3.090]
Final error norm   : 0.11 m
```
C++ controller output:<br>
<img width="80" height="60" alt="Cpp_1" src="https://github.com/user-attachments/assets/ea72be9c-b452-4ddf-9cf3-b09844795abf" />
<img width="80" height="60" alt="Cpp_2" src="https://github.com/user-attachments/assets/54fe2c26-7f8b-4236-9356-32010c96f8c9" />
<img width="80" height="60" alt="Cpp_3" src="https://github.com/user-attachments/assets/875fd027-2f08-4c99-a172-b158fec54304" />
<img width="80" height="60" alt="Cpp_4" src="https://github.com/user-attachments/assets/2ca4522c-9bfb-4f49-a195-0a4714e1d8c2" />
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
