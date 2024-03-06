import pandas as pd
from abc import ABC, abstractmethod

class BasePlot(ABC):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    @abstractmethod
    def create_plot(self) -> None:
        pass