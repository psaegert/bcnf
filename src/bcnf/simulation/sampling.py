import os
import pickle

import numpy as np
from dynaconf import Dynaconf
from tqdm import tqdm

from bcnf.simulation.camera import record_trajectory
from bcnf.simulation.physics import calculate_point_of_impact, physics_ODE_simulation
from bcnf.utils import get_dir


def sample_from_config(values: dict
                       ) -> float:
    distribution_type = values['distribution']
    if distribution_type == 'uniform':
        return np.random.uniform(values.get('min'), values.get('max'))
    elif distribution_type == 'gaussian':
        return np.random.normal(0, 1)  # mean and std are used as parameters for the standard normal distribution later
    elif distribution_type == 'gamma':
        return np.random.gamma(values.get('shape'), values.get('scale'))
    else:
        raise ValueError(f'Unknown distribution type: {distribution_type}')


def get_cams_position(cam_radiants: np.ndarray = np.array([0, 0]),
                      cam_circle_radius: float = 25,
                      cam_heights: np.ndarray = np.array([1, 1])
                      ) -> list[np.ndarray]:
    cams = []
    for cam_radiant, cam_height in (cam_radiants, cam_heights):
        cams.append(np.array([-cam_circle_radius * np.cos(cam_radiant), cam_circle_radius * np.sin(cam_radiant), cam_height]))
    return cams


def accept_visibility(visibility: float
                      ) -> bool:
    # assuming uniform visibility distribution between 0 and 1: acceptance rate of ~50 %
    if visibility > 0.75:
        return True
    elif 1 / (1 + np.exp(-(visibility - 0.5) * 10)) > np.random.uniform(0, 1):  # modified sigmoid
        return True
    else:
        return False


def accept_traveled_distance(distance: float,
                             ) -> bool:
    # assuming uniform distance distribution between 0 m and 50 m: acceptance rate of ~70 %
    ratio = distance / 50  # 50 m is just a base reference (diameter of the camera circle)
    if ratio > 0.75:
        return True
    elif np.sqrt(ratio) > np.random.uniform(0, 1):  # square root acceptance
        return True
    else:
        return False


def sample_ballistic_parameters(num_cams: int = 2,
                                cfg_file: str = f'{get_dir()}/configs/config.yaml'
                                ) -> tuple:
    config = Dynaconf(settings_files=[cfg_file])

    # pos
    if config['x0']['x0_xy']['distribution'] == 'gaussian':
        r_x = np.sqrt(np.abs(sample_from_config(config['x0']['x0_xy']))) * config['x0']['x0_xy']['std'] + config['x0']['x0_xy']['mean']
    elif config['x0']['x0_xy']['distribution'] == 'uniform':
        r_x = sample_from_config(config['x0']['x0_xy'])

    phi = np.random.uniform(0, 2 * np.pi)

    x0_x = r_x * np.cos(phi)
    x0_y = r_x * np.sin(phi)

    if config['x0']['x0_z']['distribution'] == 'gaussian':
        x0_z = sample_from_config(config['x0']['x0_z']) * config['x0']['x0_z']['std'] + config['x0']['x0_z']['mean']
    elif config['x0']['x0_z']['distribution'] == 'uniform':
        x0_z = sample_from_config(config['x0']['x0_z'])

    x0 = np.array([x0_x, x0_y, x0_z])

    # velo
    if config['v0']['v0_xy']['distribution'] == 'gaussian':
        r_v = np.sqrt(np.abs(sample_from_config(config['v0']['v0_xy']))) * config['v0']['v0_xy']['std'] + config['v0']['v0_xy']['mean']
    elif config['v0']['v0_xy']['distribution'] == 'uniform':
        r_v = sample_from_config(config['v0']['v0_xy'])

    phi_v = np.random.uniform(0, 2 * np.pi)

    v0_x = r_v * np.cos(phi_v)
    v0_y = r_v * np.sin(phi_v)

    if config['v0']['v0_z']['distribution'] == 'gaussian':
        v0_z = sample_from_config(config['v0']['v0_z']) * config['v0']['v0_z']['std'] + config['v0']['v0_z']['mean']
    elif config['v0']['v0_z']['distribution'] == 'uniform':
        v0_z = sample_from_config(config['v0']['v0_z'])

    v0 = np.array([v0_x, v0_y, v0_z])

    # wind
    if config['w']['w_xy']['distribution'] == 'gaussian':
        r_x = np.sqrt(np.abs(sample_from_config(config['w']['w_xy']))) * config['w']['w_xy']['std'] + config['w']['w_xy']['mean']
    elif config['w']['w_xy']['distribution'] == 'uniform':
        r_x = sample_from_config(config['w']['w_xy'])

    phi_w = np.random.uniform(0, 2 * np.pi)

    w_x = r_x * np.cos(phi_w)
    w_y = r_x * np.sin(phi_w)

    if config['w']['w_z']['distribution'] == 'gaussian':
        w_z = sample_from_config(config['w']['w_z']) * config['w']['w_z']['std'] + config['w']['w_z']['mean']
    elif config['w']['w_z']['distribution'] == 'uniform':
        w_z = sample_from_config(config['w']['w_z'])

    w = np.array([w_x, w_y, w_z])

    # thrust

    if config['a']['distribution'] == 'gaussian':
        r_a = np.cbrt(np.abs(sample_from_config(config['a']))) * config['a']['std'] + config['a']['mean']
    elif config['a']['distribution'] == 'uniform':
        r_a = sample_from_config(config['a'])

    phi_a = np.random.uniform(0, 2 * np.pi)
    theta_a = np.random.uniform(0, np.pi)

    a_x = r_a * np.sin(theta_a) * np.cos(phi_a)
    a_y = r_a * np.sin(theta_a) * np.sin(phi_a)
    a_z = r_a * np.cos(theta_a)

    a = np.array([a_x, a_y, a_z])

    # grav
    g_z = sample_from_config(config['g'])

    g = np.array([0, 0, -g_z])

    # b
    # density of atmosphere
    rho = sample_from_config(config['rho'])

    # radius of ball
    r = sample_from_config(config['r_ball'])

    # area of thown object
    A = np.pi * r**2

    # drag coefficient
    Cd = sample_from_config(config['Cd'])

    b = rho * A * Cd

    # mass
    m = sample_from_config(config['m'])

    # second cam position
    cam_radian_array = [sample_from_config(config['cam_radian']) for _ in range(num_cams - 1)]

    # cam radius
    cam_radius = sample_from_config(config['cam_radius'])

    # cam angles
    cam_angles = [sample_from_config(config['cam_angle']) for _ in range(num_cams)]

    # cam heights
    cam_heights = [sample_from_config(config['cam_height']) for _ in range(num_cams)]

    return x0, v0, g, w, b, m, a, cam_radian_array, r, A, Cd, rho, cam_radius, cam_angles, cam_heights


def generate_data(n: int = 100,
                  type: str = 'parameters',  # 'render', 'trajectory', or 'parameters'
                  SPF: float = 1 / 30,
                  T: float = 4,
                  ratio: tuple = (16, 9),
                  fov_horizontal: float = 70.0,
                  cam1_radian: float = 0.0,
                  print_acc_rej: bool = False,
                  name: str = 'data',
                  num_cams: int = 2,
                  config_file: str = f'{get_dir()}/configs/config.yaml',
                  ) -> None:
    cam1_radian = np.array([cam1_radian])
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
        'cam_radian': [],
        'r': [],
        'cam_radius': [],
        'cam_angles': [],
        'cam_heights': []
    }

    while accepted_count < n:
        x0, v0, g, w, b, m, a, cam_radian_array, r, A, Cd, rho, cam_radius, cam_angles, cam_heights = sample_ballistic_parameters(num_cams=num_cams, cfg_file=config_file)

        # first check: will the ball actually come down again?
        if g[2] + a[2] > 0:
            rejected_count += 1
            continue

        # second check: is x0_z > 0?
        if x0[2] < 0:
            rejected_count += 1
            continue

        # third check: how far does the ball travel?
        poi = calculate_point_of_impact(x0, v0, g, w, b, m, rho, r, a)
        distance = np.linalg.norm(poi - x0)

        if not accept_traveled_distance(distance):
            rejected_count += 1
            continue

        # last check: in how many frames is the ball visible?
        traj = physics_ODE_simulation(x0, v0, g, w, b, m, rho, r, a, T, SPF)
        cam_radian_array = np.concatenate([cam1_radian, cam_radian_array])
        cams_pos = get_cams_position(cam_radian_array, cam_radius, cam_heights)

        cams = []
        for cam, angle in zip(cams_pos, cam_angles):
            cams.append(record_trajectory(traj, ratio, fov_horizontal, cam, make_gif=False, radius=r, viewing_angle=angle))

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
            data['cam_radian'].append(cam_radian_array)
            data['r'].append(r)
            data['cam_radius'].append(cam_radius)
            data['cam_angles'].append(cam_angles)
            data['cam_heights'].append(cam_heights)

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
            data['cam_radian'].append(cam_radian_array)
            data['r'].append(r)
            data['cam_radius'].append(cam_radius)
            data['cam_angles'].append(cam_angles)
            data['cam_heights'].append(cam_heights)

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
            data['cam_radian'].append(cam_radian_array)
            data['r'].append(r)
            data['cam_radius'].append(cam_radius)
            data['cam_angles'].append(cam_angles)
            data['cam_heights'].append(cam_heights)
        else:
            raise ValueError('type must be one of "render", "trajectory", or "parameters"')

        accepted_count += 1

        if accepted_count % 100 == 0:
            with open(os.path.join(get_dir('data', 'bcnf-data', create=True), name + '.pkl'), 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        pbar.update(1)
        if print_acc_rej:
            pbar.set_postfix(accepted=accepted_count, rejected=rejected_count, ratio=accepted_count / (accepted_count + rejected_count))

    pbar.close()

    with open(os.path.join(get_dir('data', 'bcnf-data', create=True), name + '.pkl'), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)