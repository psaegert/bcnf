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
                           dt: float = 0.1                              # time step
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
        if x_sol[i, 2] < 0:
            t = -x_sol[i - 1, 2] / v_sol[i, 2]
            x_sol[i] = x_sol[i - 1] + v_sol[i] * t     # point of impact

            x_sol[i:] = x_sol[i]                      # object doesn't move after impact
            v_sol[i:] = np.zeros(3)                   # object doesn't move after impact
            break

    return x_sol
