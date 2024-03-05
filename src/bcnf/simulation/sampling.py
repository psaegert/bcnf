import numpy as np
import pickle
import os
from tqdm import tqdm

from bcnf.simulation.camera import record_trajectory
from bcnf.simulation.physics import physics_ODE_simulation
from bcnf.utils import get_dir


def get_cams_position(cam_radiants: np.ndarray
                      ) -> list[np.ndarray]:
    cams = []
    for cam in cam_radiants:
        cams.append(np.array([-25 * np.cos(cam), 25 * np.sin(cam), 1.5]))
    return cams


def accept_visibility(visibility: float
                      ) -> bool:
    # assuming uniform visibility distribution: acceptance rate of ~50 %
    if visibility > 0.75:
        return True
    elif 1 / (1 + np.exp(-(visibility - 0.5) * 10)) > np.random.uniform(0, 1):  # modified sigmoid
        return True
    else:
        return False


def sample_ballistic_parameters(num_cams: int = 2
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray, float, float, float, float]:
    # pos
    r_x = np.sqrt(np.random.uniform(0, 1)) * 40           # 40 m, i.e. can stand in- and outside the camera circle
    phi = np.random.uniform(0, 2 * np.pi)

    x0_x = r_x * np.cos(phi)
    x0_y = r_x * np.sin(phi)
    x0_z = np.random.uniform(1, 2)                      # 1 - 2 m above ground as normal human height

    x0 = np.array([x0_x, x0_y, x0_z])

    # velo
    r_v = np.sqrt(np.random.uniform(0, 1)) * 25         # 25 m/s; 90km/h is possbible for a human throw
    phi_v = np.random.uniform(0, 2 * np.pi)

    v0_x = r_v * np.cos(phi_v)
    v0_y = r_v * np.sin(phi_v)
    v0_z = np.random.uniform(-25, 25)                   # bomb it down to the ground or up to the sky

    v0 = np.array([v0_x, v0_y, v0_z])

    # grav
    g_z = np.random.uniform(-0.08 * 9.81, -2.5 * 9.81)  # from ~0.779 m/s^2 up to ~24.79 m/s^2 (Triton to Jupiter)

    g = np.array([0, 0, g_z])

    # wind
    r_x = np.sqrt(np.random.uniform(0, 1)) * 20         # includes up to the definition of "stürmischer Wind" (https://de.wikipedia.org/wiki/Windgeschwindigkeit)
    phi_x = np.random.uniform(0, 2 * np.pi)

    w_x = r_x * np.cos(phi_x)
    w_y = r_x * np.sin(phi_x)
    w_z = np.random.uniform(-10, 10)

    w = np.array([w_x, w_y, w_z])

    # b
    # density of atmosphere
    rho = np.random.uniform(0, 1.5)                     # excluding venus with insane 67 kg/m^3...

    # area of thown object
    A = np.random.uniform(0.003, 0.15)                  # from a small ball (3 cm radius) to a big ball (20 cm radius)

    # drag coefficient
    Cd = np.random.uniform(0.04, 1.42)                  # reference for 3D objects (https://en.wikipedia.org/wiki/Drag_coefficient)

    b = rho * A * Cd                                    # b

    # mass
    m = np.random.uniform(0.056, 0.62)                  # from a tennis ball (56 g) to a basketball (620 g)

    # thrust
    r_a = np.cbrt(np.random.uniform(0, 1)) * 5         # 5 m/s^2 thrust at max in any direction
    phi_a = np.random.uniform(0, 2 * np.pi)
    theta_a = np.random.uniform(0, np.pi)

    a_x = r_a * np.sin(theta_a) * np.cos(phi_a)
    a_y = r_a * np.sin(theta_a) * np.sin(phi_a)
    a_z = r_a * np.cos(theta_a)

    a = np.array([a_x, a_y, a_z])

    # second cam position
    l_array = np.random.uniform(0, 2 * np.pi, size=(num_cams - 1))

    # average radius of ball
    r = (A / np.pi)**0.5

    return x0, v0, g, w, b, m, a, l_array, r, A, Cd, rho


def generate_data(n: int = 100,
                  type: str = 'parameters',  # 'render', 'trajectory', or 'parameters'
                  SPF: float = 1 / 30,
                  T: float = 3,
                  ratio: tuple = (16, 9),
                  fov_horizontal: float = 70.0,
                  cam1_pos: np.ndarray = np.array([-25, 0, 1.5]),
                  print_acc_rej: bool = False,
                  name: str = 'data',
                  num_cams: int = 2
                  ) -> None:
    pbar = tqdm(total=n)

    accepted_count = 0
    rejected_count = 0

    data: dict[str, list] = {
        'cams': [],
        'traj': [],
        'x0_x': [],
        'x0_y': [],
        'x0_z': [],
        'v0_x': [],
        'v0_y': [],
        'v0_z': [],
        'g': [],
        'w_x': [],
        'w_y': [],
        'w_z': [],
        'b': [],
        'A': [],
        'Cd': [],
        'rho': [],
        'm': [],
        'a_x': [],
        'a_y': [],
        'a_z': [],
        'l': [],
        'r': []
    }

    while accepted_count < n:
        x0, v0, g, w, b, m, a, l, r, A, Cd, rho = sample_ballistic_parameters(num_cams=num_cams)

        # first check: will the ball actually come down again?
        if g[2] + a[2] > 0:
            rejected_count += 1
            continue

        # second check: in how many frames is the ball visible?
        traj = physics_ODE_simulation(x0, v0, g, w, b, m, a, T, SPF)
        l_array = np.concatenate([cam1_pos, l])
        cams_pos = get_cams_position(l_array)

        cams = []
        for cam in cams_pos:
            cams.append(record_trajectory(traj, ratio, fov_horizontal, cam, make_gif=False, radius=r))

        vis = np.sum([np.sum(cam) for cam in cams]) / (len(cams) * len(cams[0]))

        if not accept_visibility(vis):
            rejected_count += 1
            continue

        # add to list
        if type == 'render':
            # append cam1, cam2 and parameters
            data['cams'].append(cams)
            data['traj'].append(traj)
            data['x0_x'].append(x0[0])
            data['x0_y'].append(x0[1])
            data['x0_z'].append(x0[2])
            data['v0_x'].append(v0[0])
            data['v0_y'].append(v0[1])
            data['v0_z'].append(v0[2])
            data['g'].append(g[2])
            data['w_x'].append(w[0])
            data['w_y'].append(w[1])
            data['w_z'].append(w[2])
            data['b'].append(b)
            data['A'].append(A)
            data['Cd'].append(Cd)
            data['rho'].append(rho)
            data['m'].append(m)
            data['a_x'].append(a[0])
            data['a_y'].append(a[1])
            data['a_z'].append(a[2])
            data['l'].append(l)
            data['r'].append(r)

        elif type == 'parameters':
            data['x0_x'].append(x0[0])
            data['x0_y'].append(x0[1])
            data['x0_z'].append(x0[2])
            data['v0_x'].append(v0[0])
            data['v0_y'].append(v0[1])
            data['v0_z'].append(v0[2])
            data['g'].append(g[2])
            data['w_x'].append(w[0])
            data['w_y'].append(w[1])
            data['w_z'].append(w[2])
            data['b'].append(b)
            data['A'].append(A)
            data['Cd'].append(Cd)
            data['rho'].append(rho)
            data['m'].append(m)
            data['a_x'].append(a[0])
            data['a_y'].append(a[1])
            data['a_z'].append(a[2])
            data['l'].append(l)
            data['r'].append(r)

        elif type == 'trajectory':
            data['traj'].append(traj)
            data['x0_x'].append(x0[0])
            data['x0_y'].append(x0[1])
            data['x0_z'].append(x0[2])
            data['v0_x'].append(v0[0])
            data['v0_y'].append(v0[1])
            data['v0_z'].append(v0[2])
            data['g'].append(g[2])
            data['w_x'].append(w[0])
            data['w_y'].append(w[1])
            data['w_z'].append(w[2])
            data['b'].append(b)
            data['A'].append(A)
            data['Cd'].append(Cd)
            data['rho'].append(rho)
            data['m'].append(m)
            data['a_x'].append(a[0])
            data['a_y'].append(a[1])
            data['a_z'].append(a[2])
            data['l'].append(l)
            data['r'].append(r)
        else:
            raise ValueError('type must be one of "render", "trajectory", or "parameters"')

        accepted_count += 1

        pbar.update(1)
        if print_acc_rej:
            pbar.set_postfix(accepted=accepted_count, rejected=rejected_count)

    pbar.close()

    with open(os.path.join(get_dir('data', 'bcnf_data', create=True), name + '.pkl'), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
