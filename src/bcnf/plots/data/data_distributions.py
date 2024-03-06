import pandas as pd

from bcnf.plots.core import BasePlot


class DataDistributionPlot(BasePlot):
    def __init__(self, data: pd.DataFrame):
        super().__init__(data)

    def create_plot(self) -> None:
        pass