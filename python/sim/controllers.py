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