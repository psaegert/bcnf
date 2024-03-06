import matplotlib.pyplot as plt
import pandas as pd

from bcnf.plots.core import BasePlot


class DataDistributionPlot(BasePlot):
    """
    DataDistributionPlot inherits from BasePlot

    Class for creating plots for the distribution of the data.
    One plot is created with several subplots, one for each column in the data.
    Each subplot sonstitutes a histogram of the data in the corresponding column.

    Attributes
    ----------
    None

    Methods
    -------
    create_plots(bins: int = 50)
        Creates the plots and stores them in the figs attribute
    """
    def __init__(self, data: pd.DataFrame):
        """
        Parameters
        ----------
        data : pd.DataFrame
            data that will be used to create the plots
        """
        super().__init__(data)

    def create_plots(self, bins: int = 50) -> None:
        """
        Creates the plots and stores them in the figs attribute

        Parameters
        ----------
        bins : int
            Number of bins to use for the histograms

        Returns
        -------
        None
        """
        rows = int(self.columns_count // 5)
        cols = int(self.columns_count / rows) + (self.columns_count % rows > 0)

        fig, axes = plt.subplots(nrows=rows,
                                 ncols=cols,
                                 figsize=(8, 2 * rows))

        for i, column in enumerate(self.column_names):
            ax = axes[i // cols, i % cols]
            ax.hist(self.data[column], bins=bins)
            ax.set_title(column)

            # Adjust y-axis
            if self.data[column].min() >= 0:
                ax.set_xlim(left=0)  # If data is all >= 0, set lower limit to 0
            elif self.data[column].max() <= 0:
                ax.set_xlim(right=0)  # If data is all <= 0, set upper limit to 0
            else:
                upper = max(0,
                            abs(ax.get_xlim()[0]),
                            abs(ax.get_xlim()[1]))  # Get current upper limit
                ax.set_xlim(left=-upper, right=upper)  # Center y-axis at 0

        plt.tight_layout()
        plt.suptitle("Distribution of parameters for generated data", fontsize=16)
        fig.text(0.02, 0.5, 'normalized counts', va='center', rotation='vertical', fontsize=12)
        plt.subplots_adjust(top=0.90, left=0.1, right=0.97, bottom=0.03)

        self.figs.append(fig)
        plt.close(fig)
