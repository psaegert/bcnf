import matplotlib.pyplot as plt
import numpy as np

# compare simple physics simulation with ODE integration


def debug_plotting(p: np.ndarray,       # position array (n, 3)
                   q: np.ndarray        # position array (n, 3)
                   ) -> None:

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # show time as colour
    t = np.arange(p.shape[0])
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=t, cmap='viridis')
    ax.scatter(q[:, 0], q[:, 1], q[:, 2], c=t, cmap='Reds')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.show()
