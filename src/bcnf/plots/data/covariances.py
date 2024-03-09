import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bcnf.plots.core import BasePlot


# custom function for covariance coefficients (standard df.corr() ignores arrays filled with 0s, wich we do have)
def cov_coeff(a: np.ndarray,
              b: np.ndarray) -> float:
    # if a or b is only zeros, return 0
    if np.all(a == 0) and np.all(b == 0):
        return 1
    elif np.all(a == 0) or np.all(b == 0):
        return 0
    else:
        return np.corrcoef(a, b)[0, 1]


class DataConvariancePlot(BasePlot):
    """
    DataConvariancePlot inherits from BasePlot

    Class for creating plots for the covariance of the data.
    As many plots are created as there are columns in the data.
    Each plot consits of as many subplots as there are columns in the data.
    A subplot is a 2D histograms of the data of a column pair.
    One additional plot is created, which is the matrix of the covariance of the data.

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
        self._create_covariance_plot()
        self._create_all_pairs_plot(bins)

    def _create_covariance_plot(self) -> None:
        """
        Creates the covariance plot and stores it in the figs attribute

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        fig, ax = plt.subplots()

        ax.matshow(self.data.corr(method=cov_coeff))
        ax.set_xticks(range(self.data.shape[1]))
        ax.set_xticklabels(self.data.columns, rotation=90)
        ax.set_yticks(range(self.data.shape[1]))
        ax.set_yticklabels(self.data.columns)

        plt.suptitle("Correlation of parameters for generated data", fontsize=16)
        plt.subplots_adjust(top=0.80, left=0.03, right=0.97, bottom=0.03)

        self.figs.append(fig)
        plt.close(fig)

    def _create_all_pairs_plot(self, bins: int = 50) -> None:
        """
        Creates the all pairs plots and stores it in the figs attribute

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

        for i, column_i in enumerate(self.column_names):
            fig, axes = plt.subplots(nrows=rows,
                                     ncols=cols,
                                     figsize=(10, 2 * rows))
            for j, column_j in enumerate(self.column_names):
                ax = axes[j // cols, j % cols]
                ax.hist2d(self.data.iloc[:, i],
                          self.data.iloc[:, j],
                          bins=bins)
                ax.set_xlabel(column_i)
                ax.set_ylabel(column_j)

            plt.suptitle("Covariance of parameter pairs for generated data", fontsize=16)
            plt.subplots_adjust(wspace=0.7, hspace=0.5)
            plt.subplots_adjust(top=0.90, left=0.1, right=0.97, bottom=0.07)

            self.figs.append(fig)
            plt.close(fig)
