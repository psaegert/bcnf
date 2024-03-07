from abc import ABC, abstractmethod
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class BasePlot(ABC):
    """
    Base class for creating plots for the report

    Attributes
    ----------
    data : pd.DataFrame
        data that will be used to create the plots
    column_names : List[str]
        List of strings representing the column names of the data
    columns_count : int
        Number of columns in the data
    figs : List[plt.Figure]
        List of figures that stores the created plots

    Methods
    -------
    create_plots()
        Creates the plots and stores them in the figs attribute
    save_plots(filepath: str, base_filename: str)
        Saves the plots to the specified filepath with the given base_filename
    show_plots()
        Shows the plots in e.g. a Jupyter notebook
    """

    def __init__(self, data: pd.DataFrame):
        """
        Parameters
        ----------
        data : pd.DataFrame
            data that will be used to create the plots
        """
        self.data = data
        self.column_names = self.data.columns
        self.columns_count = len(self.column_names)
        self.figs: List[plt.Figure] = []

    @abstractmethod
    def create_plots(self) -> None:
        """
        Creates the plots and stores them in the figs attribute

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pass

    def save_plots(self, filepath: str, base_filename: str) -> None:
        """
        Saves the plots to the specified filepath with the given base_filename.
        The plots will be saved with the base_filename and an index number

        Parameters
        ----------
        filepath : str
            The path to the directory where the plots will be saved
        base_filename : str
            The base filename for the plots.

        Returns
        -------
        None
        """
        for i, fig in enumerate(self.figs):
            fig.savefig(f"{filepath}{base_filename}_{i}.png", dpi=300)

    def show_plots(self) -> None:
        """
        Shows the plots in e.g. a Jupyter notebook

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
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
