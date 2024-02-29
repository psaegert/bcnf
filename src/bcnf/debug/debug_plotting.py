import matplotlib.pyplot as plt
import numpy as np

# compare simple physics simulation with ODE integration


def debug_plotting(p: np.ndarray,       # position array
                   q: np.ndarray        # position array
                   ) -> None:

    if p.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # show time as colour
        t = np.arange(p.shape[0])
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=t, cmap='viridis')
        ax.scatter(q[:, 0], q[:, 1], q[:, 2], c=t, cmap='Reds')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # limit the x, y axis to 50
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)

        plt.show()

    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # show time as colour
        t = np.arange(p.shape[0])
        ax.scatter(p[:, 0], p[:, 1], c=t, cmap='viridis')
        ax.scatter(q[:, 0], q[:, 1], c=t, cmap='Reds')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')

        # limit the x, y axis to 50
        ax.set_xlim(-50, 50)

        plt.show()


def show_camera_image(image: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))

    phi = 0.6108652381980153
    theta = 0.3436116964863836

    ax.set_xlabel('horizontal angle')
    ax.set_ylabel('vertical angle')

    ax.imshow(image, extent=[-phi, phi, -theta, theta], aspect='equal', cmap='hot')

    plt.show()
