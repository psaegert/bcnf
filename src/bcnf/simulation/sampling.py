from typing import Callable

import numpy as np
from dynaconf import Dynaconf
from tqdm import tqdm

from bcnf.simulation.camera import record_trajectory
from bcnf.simulation.physics import calculate_point_of_impact, physics_ODE_simulation
from bcnf.utils import get_dir


def generate_data_old(
        x0_pdf: Callable = lambda size: np.random.uniform(0, 10, size=size),
        v0_pdf: Callable = lambda size: np.random.uniform(-10, 10, size=size) + np.array([0, 0, 9]),
        g_pdf: Callable = lambda size: np.random.normal(9.81, 0.1, size=size) * np.array([0, 0, -1]),
        w_pdf: Callable = lambda size: np.random.normal(0, 1, size=size) * np.array([1, 1, 0.1]),
        b_pdf: Callable = lambda size: np.random.uniform(0, 1, size=size),
        m_pdf: Callable = lambda size: np.random.uniform(0.5, 1.5, size=size),
        rho_pdf: Callable = lambda size: np.random.uniform(1.0, 1.5, size=size),
        r_pdf: Callable = lambda size: np.random.uniform(0.05, 0.15, size=size),
        a_pdf: Callable = lambda size: np.random.uniform(0, 0, size=size),
        T: float = 2.0,
        dt: float = 1 / 30,
        N: int = 1,
        break_on_impact: bool = False) -> dict[str, np.ndarray]:
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
    rho = rho_pdf(size=(N,))
    r = r_pdf(size=(N,))
    a = a_pdf(size=(N, 3))

    # Run the simulation
    X = np.zeros((N, int(T / dt), 3))
    for i in tqdm(range(N)):
        # X[i] = physics_ODE_simulation(x0[i], v0[i], g[i], w[i], b[i], m[i], rho[i], r[i], a[i], T, dt, break_on_impact=break_on_impact)
        X[i] = physics_ODE_simulation(
            x0_x=x0[i, 0], x0_y=x0[i, 1], x0_z=x0[i, 2],
            v0_x=v0[i, 0], v0_y=v0[i, 1], v0_z=v0[i, 2],
            g_x=g[i, 0], g_y=g[i, 1], g_z=g[i, 2],
            w_x=w[i, 0], w_y=w[i, 1], w_z=w[i, 2],
            b=b[i], m=m[i], rho=rho[i], r=r[i], a_x=a[i, 0], a_y=a[i, 1], a_z=a[i, 2],
            T=T, dt=dt, break_on_impact=break_on_impact)

    # Stack the parameters into a single vector for each simulation
    return {
        'trajectory': X,
        'x0_x': x0[:, 0],
        'x0_y': x0[:, 1],
        'x0_z': x0[:, 2],
        'v0_x': v0[:, 0],
        'v0_y': v0[:, 1],
        'v0_z': v0[:, 2],
        'g_x': g[:, 0],
        'g_y': g[:, 1],
        'g_z': g[:, 2],
        'w_x': w[:, 0],
        'w_y': w[:, 1],
        'w_z': w[:, 2],
        'b': b,
        'm': m,
        'rho': rho,
        'r': r,
        'a_x': a[:, 0],
        'a_y': a[:, 1],
        'a_z': a[:, 2]
    }


def sample_from_config(values: dict
                       ) -> float:
    distribution_type = values['distribution']
    if distribution_type == 'uniform':
        if "min" not in values or "max" not in values:
            raise ValueError('min and max must be defined for uniform distribution')
        return np.random.uniform(values['min'], values['max'])
    elif distribution_type == 'gaussian':
        return np.random.normal(0, 1)  # mean and std are used as parameters for the standard normal distribution later
    elif distribution_type == 'gamma':
        if "shape" not in values or "scale" not in values:
            raise ValueError('shape and scale must be defined for gamma distribution')
        return np.random.gamma(values['shape'], values['scale'])
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


def accept_traveled_distance(distance: float) -> bool:
    # assuming uniform distance distribution between 0 m and 50 m: acceptance rate of ~70 %
    ratio = distance / 50  # 50 m is just a base reference (diameter of the camera circle)
    if ratio > 0.75:
        return True
    elif np.sqrt(ratio) > np.random.uniform(0, 1):  # square root acceptance
        return True
    else:
        return False


def sample_ballistic_parameters(
        num_cams: int = 2,
        cfg_file: str = f'{get_dir()}/configs/config.yaml') -> dict:

    config = Dynaconf(settings_files=[cfg_file])

    # pos
    if config['x0']['x0_xy']['distribution'] == 'gaussian':
        r_x = np.sqrt(np.abs(sample_from_config(config['x0']['x0_xy']))) * config['x0']['x0_xy']['std'] + config['x0']['x0_xy']['mean']
    elif config['x0']['x0_xy']['distribution'] == 'uniform':
        r_x = np.sqrt(sample_from_config(config['x0']['x0_xy']))

    phi = np.random.uniform(0, 2 * np.pi)

    x0_x = r_x * np.cos(phi)
    x0_y = r_x * np.sin(phi)

    if config['x0']['x0_z']['distribution'] == 'gaussian':
        x0_z = sample_from_config(config['x0']['x0_z']) * config['x0']['x0_z']['std'] + config['x0']['x0_z']['mean']
    elif config['x0']['x0_z']['distribution'] == 'uniform':
        x0_z = sample_from_config(config['x0']['x0_z'])

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

    # grav
    g_z = - sample_from_config(config['g'])

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
    cam_heights = [sample_from_config(config['cam_heights']) for _ in range(num_cams)]

    # return x0, v0, g, w, b, m, a, cam_radian_array, r, A, Cd, rho, cam_radius, cam_angles, cam_heights

    return {
        'x0_x': x0_x,
        'x0_y': x0_y,
        'x0_z': x0_z,
        'v0_x': v0_x,
        'v0_y': v0_y,
        'v0_z': v0_z,
        'g_x': 0,
        'g_y': 0,
        'g_z': g_z,
        'w_x': w_x,
        'w_y': w_y,
        'w_z': w_z,
        'b': b,
        'm': m,
        'a_x': a_x,
        'a_y': a_y,
        'a_z': a_z,
        'cam_radian_array': cam_radian_array,
        'r': r,
        'A': A,
        'Cd': Cd,
        'rho': rho,
        'cam_radius': cam_radius,
        'cam_angles': cam_angles,
        'cam_heights': cam_heights
    }


def generate_data(
        config_file: str = f'{get_dir()}/configs/config.yaml',
        n: int = 100,
        output_type: str = 'parameters',  # 'render', 'trajectory', or 'parameters'
        dt: float = 1 / 30,
        T: float = 4,
        ratio: tuple = (16, 9),
        fov_horizontal: float = 70.0,
        cam1_radian: float = 0.0,
        num_cams: int = 2,
        break_on_impact: bool = True,
        do_filter: bool = True,
        verbose: bool = False) -> dict[str, list]:

    if output_type not in ['videos', 'trajectory', 'parameters']:
        raise ValueError('output_type must be one of "render", "trajectory", or "parameters"')

    accepted_count = 0
    rejected_count = {
        "runaway": 0,
        "start_underground": 0,
        "visibility": 0,
        "distance": 0
    }

    data: dict[str, list] = {}

    pbar = tqdm(total=n, disable=not verbose)

    while accepted_count < n:
        parameters = sample_ballistic_parameters(num_cams=num_cams, cfg_file=config_file)

        # first check: will the ball actually come down again?
        if parameters['g_z'] + parameters['a_z'] > 0 and do_filter:
            rejected_count['runaway'] += 1
            pbar.set_postfix(
                accepted=accepted_count,
                rejected_runaway=rejected_count['runaway'],
                rejected_start_underground=rejected_count['start_underground'],
                rejected_visibility=rejected_count['visibility'],
                rejected_distance=rejected_count['distance'],
                ratio=accepted_count / (accepted_count + sum(rejected_count.values())))
            continue

        # second check: is x0_z > 0?
        if parameters['x0_z'] < 0 and do_filter:
            rejected_count['start_underground'] += 1
            pbar.set_postfix(
                accepted=accepted_count,
                rejected_runaway=rejected_count['runaway'],
                rejected_start_underground=rejected_count['start_underground'],
                rejected_visibility=rejected_count['visibility'],
                rejected_distance=rejected_count['distance'],
                ratio=accepted_count / (accepted_count + sum(rejected_count.values())))
            continue

        # third check: how far does the ball travel?
        poi = calculate_point_of_impact(**parameters)

        # Check that poi and x0 have the same shape
        distance = float(np.linalg.norm(poi - np.array([parameters['x0_x'], parameters['x0_y'], parameters['x0_z']])))

        if not accept_traveled_distance(distance) and do_filter:
            rejected_count['distance'] += 1
            pbar.set_postfix(
                accepted=accepted_count,
                rejected_runaway=rejected_count['runaway'],
                rejected_start_underground=rejected_count['start_underground'],
                rejected_visibility=rejected_count['visibility'],
                rejected_distance=rejected_count['distance'],
                ratio=accepted_count / (accepted_count + sum(rejected_count.values())))
            continue

        trajectory = physics_ODE_simulation(T=T, dt=dt, break_on_impact=break_on_impact, **parameters)

        # Prepend the first camera radian to the other camera radians
        parameters["cam_radian_array"] = np.insert(parameters["cam_radian_array"], 0, cam1_radian)
        cams_pos = get_cams_position(parameters["cam_radian_array"], parameters["cam_radius"], parameters["cam_heights"])

        videos = []
        for cam, angle in zip(cams_pos, parameters["cam_angles"]):
            videos.append(record_trajectory(trajectory, ratio, fov_horizontal, cam, make_gif=False, radius=parameters["r"], viewing_angle=angle))

        vis = np.sum([np.sum(cam) for cam in videos]) / (len(videos) * len(videos[0]))

        if not accept_visibility(vis) and do_filter:
            rejected_count['visibility'] += 1
            pbar.set_postfix(
                accepted=accepted_count,
                rejected_runaway=rejected_count['runaway'],
                rejected_start_underground=rejected_count['start_underground'],
                rejected_visibility=rejected_count['visibility'],
                rejected_distance=rejected_count['distance'],
                ratio=accepted_count / (accepted_count + sum(rejected_count.values())))
            continue

        if output_type == 'videos':
            parameters['videos'] = videos
            parameters['trajectory'] = trajectory
        elif output_type == 'trajectory':
            parameters['trajectory'] = trajectory

        # Complete the parameters dictionary
        if len(data) == 0:
            # Create the keys for the data dictionary
            for key in parameters.keys():
                data[key] = []

        for key in parameters.keys():
            data[key].append(parameters[key])

        accepted_count += 1

        pbar.update(1)
        pbar.set_postfix(
            accepted=accepted_count,
            rejected_runaway=rejected_count['runaway'],
            rejected_start_underground=rejected_count['start_underground'],
            rejected_visibility=rejected_count['visibility'],
            rejected_distance=rejected_count['distance'],
            ratio=accepted_count / (accepted_count + sum(rejected_count.values())))
    pbar.close()

    return data
