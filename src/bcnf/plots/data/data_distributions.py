import matplotlib.pyplot as plt
import pandas as pd

from bcnf.plots.core import BasePlot


class DataDistributionPlot(BasePlot):
    def __init__(self, data: pd.DataFrame):
        super().__init__(data)

    def create_plot(self, bins: int = 50) -> None:
        column_names = self.data.columns
        columns_count = len(column_names)

        rows = int(columns_count ** 0.5)
        cols = int(columns_count / rows) + (columns_count % rows > 0)

        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(8, 2 * rows))

        for i, column in enumerate(column_names):
            ax = axes[i // cols, i % cols]
            ax.hist(self.data[column], bins=bins)
            ax.set_title(column)

            # Adjust y-axis
            if self.data[column].min() >= 0:
                ax.set_xlim(left=0)  # If data is all >= 0, set lower limit to 0
            if self.data[column].max() <= 0:
                ax.set_xlim(right=0)  # If data is all <= 0, set upper limit to 0
            else:
                upper = max(0,
                            abs(ax.get_xlim()[0]),
                            abs(ax.get_xlim()[1]))  # Get current upper limit
                ax.set_xlim(left=-upper, right=upper)  # Center y-axis at 0

        plt.tight_layout()
        plt.suptitle("Distribution of parameters for generated data", fontsize=16)
        fig.text(0.04, 0.5, 'counts', va='center', rotation='vertical', fontsize=12)
        plt.subplots_adjust(top=0.90, left=0.1, right=0.97, bottom=0.03)
        plt.show()
