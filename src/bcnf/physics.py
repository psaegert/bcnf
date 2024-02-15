import numpy as np


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
