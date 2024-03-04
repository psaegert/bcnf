from typing import Callable

import numpy as np
from scipy.integrate import odeint

# simple physics simulation


def physics_simulation(x0: np.ndarray = np.array([0, 0, 1.8]),                              # initial position
                       v0: np.ndarray = np.array([10, 10, 10]),                             # initial velocity
                       g: np.ndarray = np.array([0, 0, -9.81]),                             # gravitational acceleration
                       T: float = 10.0,                                                     # total run time in seconds
                       dt: float = 0.1                                                      # time step
                       ) -> np.ndarray:

    num_steps = int(T / dt)

    position = np.zeros((num_steps, 3))
    velocity = np.zeros((num_steps, 3))

    position[0] = x0
    velocity[0] = v0

    for i in range(1, num_steps):
        velocity[i] = velocity[i - 1] + g * dt
        position[i] = position[i - 1] + velocity[i] * dt

        # if the object hits the ground, calculate the point of impact and break the loop
        if position[i, 2] < 0:
            t = -position[i - 1, 2] / velocity[i, 2]
            position[i] = position[i - 1] + velocity[i] * t     # point of impact

            position[i:] = position[i]                      # object doesn't move after impact
            velocity[i:] = np.zeros(3)                      # object doesn't move after impact
            break

    return position


# physics simulation with ODE integration

def ballistic_ODE(v: np.ndarray = np.array([10, 10, 10]),   # velocity
                  t: np.ndarray = np.zeros((100, 3)),       # time
                  g: np.ndarray = np.array([0, 0, -9.81]),  # gravitational acceleration
                  w: np.ndarray = np.array([-10, 10, 0]),   # wind
                  b: float = 0.1,                           # drag coefficient
                  m: float = 1.0,                           # mass
                  a: np.ndarray = np.array([0, 0, 0])       # thrust
                  ) -> np.ndarray:

    dvdt = g - (b / m) * (v**2 * v / np.linalg.norm(v) - w**2 * w / np.linalg.norm(w)) + a

    return dvdt


def physics_ODE_simulation(x0: np.ndarray = np.array([0, 0, 1.8]),      # initial position
                           v0: np.ndarray = np.array([10, 10, 10]),     # initial velocity
                           g: np.ndarray = np.array([0, 0, -9.81]),     # gravitational acceleration
                           w: np.ndarray = np.array([-10, 10, 0]),      # wind
                           b: float = 0.1,                              # drag coefficient
                           m: float = 1.0,                              # mass
                           a: np.ndarray = np.array([0, 0, 0]),         # thrust
                           T: float = 10.0,                             # total run time in seconds
                           dt: float = 0.1,                             # time step
                           break_on_impact: bool = True                # break the simulation when the object hits the ground
                           ) -> np.ndarray:

    # create time grid
    t = np.arange(0, T, dt)

    # solve ODE
    v_sol = odeint(ballistic_ODE, v0, t, args=(g, w, b, m, a))

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


def get_data(
        x0_pdf: Callable = lambda size: np.random.uniform(-10, 10, size=size),
        v0_pdf: Callable = lambda size: np.random.uniform(-10, 10, size=size),
        g_pdf: Callable = lambda size: np.random.normal(9.81, 0.1, size=size) * np.array([0, 0, -1]),
        w_pdf: Callable = lambda size: np.random.uniform(-10, 10, size=size),
        b_pdf: Callable = lambda size: np.random.uniform(0, 1, size=size),
        m_pdf: Callable = lambda size: np.random.uniform(0.5, 1.5, size=size),
        a_pdf: Callable = lambda size: np.random.uniform(0, 0, size=size),
        T: float = 5.0,
        dt: float = 0.1,
        N: int = 1,
        break_on_impact: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a dataset from prior parameter distributions and the simulation model.

    Parameters
    ----------
    x0_pdf : function
        A function that returns a sample from the prior distribution of the initial position.
    v0_pdf : function
        A function that returns a sample from the prior distribution of the initial velocity.
    g_pdf : function
        A function that returns a sample from the prior distribution of the gravitational acceleration.
    w_pdf : function
        A function that returns a sample from the prior distribution of the wind.
    b_pdf : function
        A function that returns a sample from the prior distribution of the drag coefficient.
    m_pdf : function
        A function that returns a sample from the prior distribution of the mass.
    a_pdf : function
        A function that returns a sample from the prior distribution of the thrust.
    T : float
        The total run time in seconds.
    dt : float
        The time step.
    N : int
        The number of simulations to run.

    Returns
    -------
    X : array
        The simulated data, shape (N, int(T / dt), 3).
    y : array
        The parameters used to simulate the data, shape (N, 14).
    """

    x0 = x0_pdf(size=(N, 3))
    v0 = v0_pdf(size=(N, 3))
    g = g_pdf(size=(N, 3))
    w = w_pdf(size=(N, 3))
    b = b_pdf(size=(N,))
    m = m_pdf(size=(N,))
    a = a_pdf(size=(N, 3))

    # Run the simulation
    X = np.zeros((N, int(T / dt), 3))
    for i in range(N):
        X[i] = physics_ODE_simulation(x0[i], v0[i], g[i], w[i], b[i], m[i], a[i], T, dt, break_on_impact=break_on_impact)

    # Stack the parameters into a single vector for each simulation
    y = np.column_stack([x0, v0, w, b, m, a])

    return X, y


# calculate point of impact for given parameters

def calculate_point_of_impact(x0: np.ndarray = np.array([0, 0, 1.8]),      # initial position
                              v0: np.ndarray = np.array([10, 10, 10]),     # initial velocity
                              g: np.ndarray = np.array([0, 0, -9.81]),     # gravitational acceleration
                              w: np.ndarray = np.array([-10, 2.7, 0]),      # wind
                              b: float = 0.1,                              # drag coefficient
                              m: float = 1.0,                              # mass
                              a: np.ndarray = np.array([0, 0, 0]),         # thrust
                              dt: float = 0.1                              # time step
                              ) -> np.ndarray:

    # initial time
    t = 0.0

    while True:
        # solve ODE for one time step
        v_sol = odeint(ballistic_ODE, v0, [t, t + dt], args=(g, w, b, m, a))

        # calculate position
        x_sol = x0 + v0 * dt

        # when the object hits the ground, calculate the point of impact and break the loop
        if x_sol[2] < 0:
            t = -x0[2] / v0[2]
            x_sol = x0 + v0 * t  # point of impact
            break

        # update initial position and velocity
        x0 = x_sol
        v0 = v_sol[1]

        # update time
        t += dt

    return x_sol
