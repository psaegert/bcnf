from abc import ABC, abstractmethod
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class BasePlot(ABC):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.column_names = self.data.columns
        self.columns_count = len(self.column_names)
        self.figs: List[plt.Figure] = []

    @abstractmethod
    def create_plots(self) -> None:
        pass

    def save_plots(self, filepath: str, base_filename: str) -> None:
        for i, fig in enumerate(self.figs):
            fig.savefig(f"{filepath}{base_filename}_{i}.png", dpi=300)

    def show_plots(self) -> None:
        for fig in self.figs:
            canvas = FigureCanvas(fig)
            canvas.draw()
            s, (width, height) = canvas.print_to_buffer()

            # Get the ARGB image, preserving colors better
            X = np.frombuffer(s, np.uint8).reshape((height, width, 4))

            # Display the image
            plt.imshow(X)
            plt.axis('off')  # Don't display axes
            plt.show()
