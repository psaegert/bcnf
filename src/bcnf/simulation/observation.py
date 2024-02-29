import numpy as np


def gaussian_observation_noise(p: np.ndarray,  # position array
                               std: float = 0.1
                               ) -> np.ndarray:
    # add gaussian noise, if the object is still in the air
    p[p[:, -1] > 0] = p[p[:, -1] > 0] + np.random.normal(0, std, p[p[:, -1] > 0].shape)
    return p


# project on x, z plane
def simple_2D_camera_observation(p: np.ndarray,  # position array (n, 3)
                                 noise: bool = False,
                                 std: float = 0.1
                                 ) -> np.ndarray:
    # project on x, z plane
    if noise:
        return gaussian_observation_noise(p[:, [0, 2]], std=std)
    else:
        return p[:, [0, 2]]
