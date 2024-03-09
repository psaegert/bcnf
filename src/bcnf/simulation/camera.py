import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from bcnf.utils import get_dir


def rotate_vector(vector: np.ndarray,
                  angle_degrees: float = 45  # angle [+45 upwards and -45 downwards]
                  ) -> np.ndarray:
    # Convert angle from degrees to radians
    angle_radians = angle_degrees * np.pi / 180

    # tranfor vector to spherical coorinates
    r = np.linalg.norm(vector)
    theta = np.arccos(vector[2] / r)
    phi = np.arctan2(vector[1], vector[0])

    # Update the spherical coordinates
    theta -= angle_radians

    # Transform the spherical coordinates back to Cartesian coordinates
    rotated_vector = np.array([r * np.sin(theta) * np.cos(phi),
                               r * np.sin(theta) * np.sin(phi),
                               r * np.cos(theta)])

    return rotated_vector


def record_trajectory(trajectory: np.ndarray = np.array([[0, 0, 0]]),
                      ratio: tuple = (16, 9),
                      fov_horizontal: float = 70.0,
                      cam_pos: np.ndarray = np.array([-25, 0, 1.5]),
                      make_gif: bool = True,
                      gif_name: str = 'trajectory',
                      radius: float = 0.1143,
                      viewing_angle: float = 0.0
                      ) -> np.ndarray:
    aspect_ratio = ratio[0] / ratio[1]

    fov_vertical = fov_horizontal / aspect_ratio

    phi = (fov_horizontal / 2) * (np.pi / 180)
    theta = (fov_vertical / 2) * (np.pi / 180)

    film = []
    for position in trajectory:
        img = camera(
            ball_pos=position,
            ratio=ratio,
            fov_horizontal=fov_horizontal,
            cam_pos=cam_pos,
            radius=radius,
            viewing_angle=viewing_angle
        )
        film.append(img)

    film = np.array(film)

    if make_gif:
        fig, ax = plt.subplots(figsize=(ratio[0], ratio[1]))
        ax.set_xlabel('horizontal angle')
        ax.set_ylabel('vertical angle')
        ax.title.set_text(f'{gif_name}')
        gif = []
        for i in range(trajectory.shape[0]):
            gif.append([ax.imshow(film[i], extent=[-phi, phi, -theta, theta], cmap='hot', animated=True)])

        ani = animation.ArtistAnimation(fig, gif, interval=33, blit=True, repeat_delay=3000)
        ani.save(f'{get_dir()}/tests/gifs/{gif_name}.gif', writer='imagemagick')
    return film


def camera(ball_pos: np.ndarray = np.array([0, 0, 1.5]),
           ratio: tuple = (16, 9),
           fov_horizontal: float = 70.0,
           cam_pos: np.ndarray = np.array([-25, 0, 1.5]),
           show_plot: bool = False,
           radius: float = 0.1143,
           viewing_angle: float = 0.0
           ) -> np.ndarray:
    # ratio
    aspect_ratio = ratio[0] / ratio[1]
    fov_vertical = fov_horizontal / aspect_ratio

    # calculate the azimuthal and polar angle in radian, symmetrical to the camera orientation
    phi = (fov_horizontal / 2) * (np.pi / 180)
    theta = (fov_vertical / 2) * (np.pi / 180)

    # calculate camera vectors
    focus_point = np.array([0, 0, cam_pos[2]])  # where is the camera looking at

    '''
    cam_dir = (focus_point - cam_pos) / np.linalg.norm(focus_point - cam_pos)  # normalized vector of camera orientation
    cam_orthogonal_z = np.array([0, 0, 1])  # z-axis of the camera, here always upwards
    cam_orthogonal = np.cross(cam_dir, cam_orthogonal_z)  # parallel to the camera screen
    '''

    # calculate normalized camera looking direction
    cam_dir = (focus_point - cam_pos) / np.linalg.norm(focus_point - cam_pos)  # normalized vector of camera orientation
    # rotate according to angle
    cam_dir = rotate_vector(cam_dir, viewing_angle)
    # calculate orthogonal vector with x and y remaining the same, aka rotate 90 upwards
    cam_orthogonal_z = rotate_vector(cam_dir, 90)
    cam_orthogonal = np.cross(cam_dir, cam_orthogonal_z)  # parallel to the camera screen

    # ball properties
    # ball_radius = 0.1143  # stadard radius of a football in meters; here covers 90% of normal gaussian
    ball_radius = radius

    # sampling
    n = 5000
    samples = np.random.normal(loc=ball_pos, scale=ball_radius / 1.644854, size=(n, 3))  # a std of 1.644854 covers 90%

    # calculate projections for each sample

    v = samples - cam_pos
    v = v / np.linalg.norm(v, axis=1)[:, None]

    # calculate the azimuthal angle ph
    project_on_cam_orthogonal = np.dot(v, cam_orthogonal)
    project_on_cam_dir = np.dot(v, cam_dir)
    ph = np.arctan2(project_on_cam_orthogonal, project_on_cam_dir)

    # calculate the polar angle th
    project_on_cam_orthogonal_z = np.dot(v, cam_orthogonal_z)
    project_on_cam_dir = np.dot(v, cam_dir)
    th = np.arctan2(project_on_cam_orthogonal_z, project_on_cam_dir)

    # fill a 2d histogram representing the camera image

    if show_plot:
        fig, ax = plt.subplots(figsize=(ratio[0], ratio[1]))
        ax.hist2d(ph, th, bins=[ratio[0] * 10, ratio[1] * 10], range=[[-phi, phi], [-theta, theta]], density=True, cmap='hot')
        ax.set_xlabel('horizontal angle')
        ax.set_ylabel('vertical angle')
        plt.show()

    vals = np.histogram2d(ph, th, bins=[ratio[0] * 10, ratio[1] * 10], range=[[-phi, phi], [-theta, theta]])

    # check for NaNs if the hisogram was empty (ball outside of fov)
    if np.any(vals[0]):
        # Normalize the histogram
        vals = vals[0] / np.sum(vals[0])
    else:
        # If histogram is empty, create an array filled with zeros of the same shape
        vals = np.zeros_like(vals[0])

    # reverse and transpose vals[0] to return the correct orientation of the image
    return np.flipud(vals.T)
