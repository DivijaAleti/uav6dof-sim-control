import numpy as np
from .dynamics import quat_to_R

def aero_drag_force(v_b, q, wind_w, params):
    rho = params.get("rho", 1.225)
    CdA = params.get("CdA", np.array([0.08, 0.08, 0.12]))
    R = quat_to_R(q)
    wind_b = R.T @ wind_w
    v_rel = v_b - wind_b # relative velocity in body frame

    # axis-wise quadratic drag: -0.5 * rho * CdA * |v_rel| * v_rel
    return -0.5 * rho * CdA * (np.abs(v_rel) * v_rel)