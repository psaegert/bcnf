from typing import Callable

import numpy as np
from scipy.integrate import odeint
from tqdm import tqdm


# physics simulation with ODE integration
def ballistic_ODE(v: np.ndarray = np.array([10, 10, 10]),   # velocity
                  t: np.ndarray = np.zeros((100, 3)),       # time
                  g: np.ndarray = np.array([0, 0, -9.81]),  # gravitational acceleration
                  w: np.ndarray = np.array([-10, 10, 0]),   # wind
                  b: float = 0.1,                           # drag coefficient
                  m: float = 1.0,                           # mass
                  rho: float = 1.225,                       # air density
                  r: float = 0.1,                           # radius of ball
                  a: np.ndarray = np.array([0, 0, 0])       # thrust
                  ) -> np.ndarray:
    # gravity - buoyancy - drag + wind + thrust
    dvdt = g - g * rho * (4 / 3) * (np.pi * r**3) / m - (0.5 * b / m) * (v**2 * v / np.linalg.norm(v) - w**2 * w / np.linalg.norm(w)) + a

    return dvdt


def physics_ODE_simulation(
        x0_x: float,
        x0_y: float,
        x0_z: float,
        v0_x: float,
        v0_y: float,
        v0_z: float,
        g_x: float,
        g_y: float,
        g_z: float,
        w_x: float,
        w_y: float,
        w_z: float,
        b: float,
        m: float,
        rho: float,
        r: float,
        a_x: float,
        a_y: float,
        a_z: float,
        T: float = 10.0,
        dt: float = 0.1,
        break_on_impact: bool = True,
        *args, **kwargs  # absorb any additional arguments
    ) -> np.ndarray:

    x0 = np.array([x0_x, x0_y, x0_z])
    v0 = np.array([v0_x, v0_y, v0_z])
    g = np.array([g_x, g_y, g_z])
    w = np.array([w_x, w_y, w_z])
    a = np.array([a_x, a_y, a_z])

    # create time grid
    t = np.arange(0, T, dt)

    # solve ODE
    v_sol = odeint(ballistic_ODE, v0, t, args=(g, w, b, m, rho, r, a))

    # calculate position
    x_sol = np.zeros((v_sol.shape[0], 3))
    x_sol[0] = x0

    for i in range(1, v_sol.shape[0]):
        x_sol[i] = x_sol[i - 1] + v_sol[i] * dt

        # if the object hits the ground, calculate the point of impact and break the loop
        if x_sol[i, 2] < 0 and break_on_impact:
            t = -x_sol[i - 1, 2] / v_sol[i, 2]
            x_sol[i] = x_sol[i - 1] + v_sol[i] * t     # point of impact

            x_sol[i:] = x_sol[i]                      # object doesn't move after impact
            v_sol[i:] = np.zeros(3)                   # object doesn't move after impact
            break

    return x_sol


# calculate point of impact for given parameters

def calculate_point_of_impact(
        x0_x: float,
        x0_y: float,
        x0_z: float,
        v0_x: float,
        v0_y: float,
        v0_z: float,
        g_x: float,
        g_y: float,
        g_z: float,
        w_x: float,
        w_y: float,
        w_z: float,
        b: float,
        m: float,
        rho: float,
        r: float,
        a_x: float,
        a_y: float,
        a_z: float,
        dt: float = 0.1,
        *args, **kwargs  # absorb any additional arguments
    ) -> np.ndarray:

    x0 = np.array([x0_x, x0_y, x0_z])
    v0 = np.array([v0_x, v0_y, v0_z])
    g = np.array([g_x, g_y, g_z])
    w = np.array([w_x, w_y, w_z])
    a = np.array([a_x, a_y, a_z])
    

    # initial time
    t = 0.0

    while t < 120.0:    # if it takes too long, it be be errorneous (e.g. ball keeps rising)
        # solve ODE for one time step
        v_sol = odeint(ballistic_ODE, v0, [t, t + dt], args=(g, w, b, m, rho, r, a))

        # calculate position
        x_sol = x0 + v0 * dt

        # when the object hits the ground, calculate the point of impact and break the loop
        if x_sol[2] < 0:
            t = -x0[2] / v0[2]
            x_sol = x0 + v0 * t  # point of impact
            return x_sol

        # update initial position and velocity
        x0 = x_sol
        v0 = v_sol[1]

        # update time
        t += dt

    # HACK
    return np.array([999, 999, 999])  # if the object doesn't hit the ground within 120 seconds, return a point far away to filter it out
