from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class BasePlot(ABC):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.fig: plt.Figure

    @abstractmethod
    def create_plot(self) -> None:
        pass

    def save_plot(self, filepath: str, filename: str) -> None:
        self.fig.savefig(filepath + filename, dpi=300)

    def show_plot(self) -> None:
        canvas = FigureCanvas(self.fig)
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()

        # Get the ARGB image, preserving colors better
        X = np.frombuffer(s, np.uint8).reshape((height, width, 4))

        # Display the image
        plt.imshow(X)
        plt.axis('off')  # Don't display axes
        plt.show()
