from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd


class BasePlot(ABC):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.fig: plt.Figure

    @abstractmethod
    def create_plot(self) -> None:
        pass

    def save_plot(self, filepath: str, filename: str) -> None:
        self.fig.savefig(filepath + filename)
