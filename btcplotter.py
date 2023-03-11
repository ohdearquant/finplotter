import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

class OptimizedDataHandler:
    """
    A class that handles optimization of data in pandas DataFrame for analysis
    
    ...

    Attributes
    ----------
    file : str
        the file path of the data
    
    df : pandas.DataFrame
        the pandas DataFrame containing the data
    
    rtn_df : pandas.DataFrame
        the pandas DataFrame containing the daily return data
    
    Methods
    -------
    create_df(datecol: str, rtn: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        Creates a pandas DataFrame from the data file.
        If rtn is True, then a pandas DataFrame containing the daily return data is also created.
    
    """
    
    def __init__(self, file: str):
        """
        Constructs all the necessary attributes for the OptimizedDataHandler object.
        
        Parameters
        ----------
        file : str
            the file path of the data
        """
        self.file = file
        self.df, self.rtn_df = self.create_df('Date', rtn=True)
    
    def create_df(self, datecol: str, rtn: bool = False):
        """
        Creates a pandas DataFrame from the data file.
        
        Parameters
        ----------
        datecol : str
            the name of the column that contains the dates in the data file
        rtn : bool, optional
            whether to create a pandas DataFrame containing the daily return data or not, by default False
        
        Returns
        -------
        pandas.DataFrame
            the pandas DataFrame containing the data
        Tuple[pandas.DataFrame, pandas.DataFrame]
            the pandas DataFrame containing the data and the daily return data, if rtn is True
        """
        self.df = pd.read_csv(self.file, parse_dates=[datecol], infer_datetime_format=True)
        if datecol not in self.df.columns:
            raise ValueError(f"{datecol} is not a column in the data frame")
        print(f"There are {len(self.df)} datapoints in the data")
        if rtn:
            self.rtn_df = pd.DataFrame({
                "Date": [dt.strftime('%b %-d, %Y') for dt in self.df[datecol]],
                "Return": (self.df['Close'].pct_change() * 100)
            }).dropna().reset_index(drop=True)
            mean, std = np.mean(self.rtn_df['Return']), np.std(self.rtn_df['Return'])
            print(f"There are {len(self.rtn_df)} daily return datapoints in the data with a mean of {mean:.04}, and a std of {std:.04}")
            return self.df, self.rtn_df
        return self.df


class OptimizedPlotter(OptimizedDataHandler):
    """
    A class that extends OptimizedDataHandler to plot data in various ways.

    ...

    Attributes
    ----------
    file : str
        The path to the file containing the data.

    Methods
    -------
    plot_data(yaxis, xaxis, x_label='Date', y_label='Daily Returns %', plot_type='l', plot_size=(14, 7), bins=100, title=None)
        Plots the data based on the input parameters.
    plot_line(yaxis, xaxis, plot_size, bins)
        Plots a line graph of the data with y-axis as daily returns and x-axis as date.
    plot_hist(yaxis, xaxis, plot_size, bins)
        Plots a histogram of the data with y-axis as daily returns and x-axis as date.
    plot_log(yaxis, xaxis, plot_size, bins)
        Plots a logarithmic line graph of the data with y-axis as daily returns and x-axis as date.
    format_plot(x_label='Date', y_label='Price', title=None, show_legend=False, grid_width=2, grid_style='--', x_ticks_rotation=0, y_ticks_rotation=0, font_size=8)
        Formats the plot by setting axis labels, title, font size, color, and other attributes.
    """
    
    def __init__(self, file: str) -> None:
        """
        Constructs a new OptimizedPlotter object.

        Parameters
        ----------
        file : str
            The path to the file containing the data.
        """
        super().__init__(file)

    def plot_data(self, yaxis: str, xaxis: str, x_label: str = 'Date', y_label: str = "Daily Returns %", plot_type: str = "l", plot_size: tuple[int, int] = (14, 7), bins: int = 100, title: str = None):
        """
        Plots the data.

        Parameters
        ----------
        yaxis : str
            The column name of the data for the y-axis.
        xaxis : str
            The column name of the data for the x-axis.
        x_label : str, optional
            The label for the x-axis, by default 'Date'.
        y_label : str, optional
            The label for the y-axis, by default "Daily Returns %".
        plot_type : str, optional
            The type of plot to create, by default "l".
            Choose either 'l' for line plot, 'h' for histogram, or 'lg' for log line plot.
        plot_size : tuple[int, int], optional
            The size of the plot, by default (14, 7).
        bins : int, optional
            The number of bins to use in the histogram, by default 100.
        title : str, optional
            The title of the plot, by default None.
        """
        plot_methods = {'l': self.plot_line, 'h': self.plot_hist, 'lg': self.plot_log}
        if plot_type not in plot_methods.keys():
            raise ValueError(f"Invalid plot type: {plot_type}. Choose either 'l' for line plot, 'h' for histogram, or 'lg' for log line plot.")
        
        plot_methods[plot_type](yaxis, xaxis, plot_size, bins)
        self.format_plot(x_label, y_label, title)
    
    def plot_line(self, yaxis: str, xaxis: str, plot_size: tuple[int, int], bins: int):
        """
        Plots a line graph of the data.

        Parameters
        ----------
        yaxis : str
            The column name of the data for the y-axis.
        xaxis : str
            The column name of the data for the x-axis.
        plot_size : tuple[int, int]
            The size of the plot.
        bins : int
            The number of bins to use in the histogram.
        """
        self.df.set_index(xaxis)[yaxis].plot(figsize=plot_size)
    
    def plot_hist(self, yaxis: str, xaxis: str, plot_size: tuple[int, int], bins: int):
        """
        Plots a histogram of the data.

        Parameters
        ----------
        yaxis : str
            The column name of the data for the y-axis.
        xaxis : str
            The column name of the data for the x-axis.
        plot_size : tuple[int, int]
            The size of the plot.
        bins : int
            The number of bins to use in the histogram.
        """
        self.df.set_index(xaxis)[yaxis].hist(bins=bins, density=True, figsize=plot_size)
    
    def plot_log(self, yaxis: str, xaxis: str, plot_size: tuple[int, int], bins: int):
        """
        Plots a logarithmic line graph of the data.

        Parameters
        ----------
        yaxis : str
            The column name of the data for the y-axis.
        xaxis : str
            The column name of the data for the x-axis.
        plot_size : tuple[int, int]
            The size of the plot.
        bins : int
            The number of bins to use in the histogram.
        """
        ax = self.df.set_index(xaxis)[yaxis].plot(kind='line', figsize=plot_size, logy=True)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set(yticks=[200, 1000, 5000, 10000, 20000, 40000, 50000, 60000, 70000])
    
    def format_plot(self, x_label: str = 'Date', y_label: str = "Price", title: str = None, show_legend: bool = False, grid_width: int = 2, grid_style: str = '--', x_ticks_rotation: int = 0, y_ticks_rotation: int = 0, font_size: int = 8):
        """
        Formats the plot.

        Parameters
        ----------
        x_label : str, optional
            The label for the x-axis, by default 'Date'.
        y_label : str, optional
            The label for the y-axis, by default "Price".
        title : str, optional
            The title of the plot, by default None.
        show_legend : bool, optional
            Whether or not to show the legend of the plot, by default False.
        grid_width : int, optional
            The width of the grid lines, by default 2.
        grid_style : str, optional
            The style of the grid lines, by default '--'.
        x_ticks_rotation : int, optional
            The number of degrees to rotate the x-axis tick labels, by default 0.
        y_ticks_rotation : int, optional
            The number of degrees to rotate the y-axis tick labels, by default 0.
        font_size : int, optional
            The font size of the tick labels, by default 8.
        """
        plt.ylabel(y_label, labelpad=16, fontsize=font_size)
        plt.xlabel(x_label, labelpad=16, fontsize=font_size)
        plt.title(f"{title} Price Chart", fontsize=12)
        plt.xticks(rotation=x_ticks_rotation, fontsize=font_size)
        plt.yticks(rotation=y_ticks_rotation, fontsize=font_size)
        plt.grid(linewidth=grid_width, linestyle=grid_style, axis='y')
        if show_legend:
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.show()

if __name__ == '__main__':
    a = OptimizedPlotter('BTC-USD.csv')
    a.plot_data('Close', 'Date', plot_type='lg', y_label='Price', plot_size=(10, 5), title='BTC log')
    a.plot_data('Close', 'Date', plot_type='l', y_label='Price', plot_size=(10, 5), title='BTC')