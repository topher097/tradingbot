""" V2.0 of plotCreator.py to utilize parallel processing """

import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick2_ohlc
import mplfinance as mpl
import numpy as np
import pandas as pd
from dataclasses import dataclass
import os
import pickle
import json
import math
from pathlib import Path
from loggerSettings import plotlogger
import chime
from Technical import Technical as tech
from joblib import Parallel, delayed
import multiprocessing

import matplotlib
matplotlib.use("Agg")

@dataclass
class paramHandler():
    """ Parameters for creating plots from data """
    dataDirectory : str = "F:\MarketData\CRYPTO\pickle"     # Directory of where data files can be found
    plotDirectory : str = "F:\MarketData\CRYPTO\plotsV2"      # Directory of where to save plots
    timePeriod : list
    pair : list
    maxPlots : int = 10000          # Maximum number of plots to create for each file
    window : int = 30               # Window size for the plots
    step : int = 1                  # Step through data for plots
    candleType : str = 'ha'         # Candlestick type, either Heiken Ashi (ha) or regular OHCL (ohcl)
    save : bool = True              # Whether to save the plots or not
    showN : int = 0                 # How many plots to show, if any

    def __init__(self) -> None:
        self.dataFiles = set([f for f in os.listdir(paramHandler.dataDirectory) if os.path.isfile(os.path.join(paramHandler.dataDirectory, f))])
        self.basedir = Path(__file__).parent.parent
        self.methodFileName = "bin\\methods.json"
        self.methodFilePath = os.path.join(self.basedir, self.methodFileName)


@dataclass
class plotWorks(paramHandler):
    def __init__(self, pair, time) -> None:
        super().__init__()
        plotWorks.get_methods(self)                                 # Get methods for the data
        klines = plotWorks.get_kline_data(self, pair, time)         # Get data for pair and timePeriod

        # Get the plot directory for pair
        self.plotPairDirectory = plotWorks.get_plot_directory(self, pair, time, self.window)
        # Calcualte the number of plots to generate
        self.numPlots = int(((len(klines) - self.window -1)/self.step) - 1)
        if self.numPlots > self.maxPlots: self.numPlots = self.maxPlots

        # Run plot creator
        for i in range(1, self.numPlots):
            plotWorks.plot(self)

    def plot(self, ):
        plotWorks.calculate_methods(self)       # Calculate the methods for the current data


        # Save plot
        if self.save:
            plotFilePath = plotWorks.get_plot_filepath(self, pair, self.candleType, self.window, self.step, time, self.signal)
            plt.savefig(plotFilePath, bbox_inches='tight', pad_inches=0, dpi=1000)

    def save_plot(self, plotFilePath, figure):
        plt.savefig(plotFilePath, bbox_inches='tight', pad_inches=0, dpi=1000)

    def get_kline_data(self, pair, time):
        dataFilePath = plotWorks.get_file_filepath(self, pair, time)
        with open(dataFilePath, 'rb') as f:
            klines = pickle.load(f)
        return klines

    def get_methods(self):
        try:
            with open(self.methodFilePath, "r") as file:
                data            = file.read()
                self.methods    = json.loads(data)
            plotlogger.info(f"Successfully loaded methods from '{self.methodFileName}'")
        except Exception as e:
            plotlogger.error(f"Failed to load methods, error: {e}")

    def get_data_filename(self, pair, time):
        return f"{pair}_{time}.pkl"

    def get_file_filepath(self, pair, time):
        name = plotWorks.get_data_filename(self, pair, time)
        return os.path.join(self.dataDirectory, name)

    def get_plot_filename(self, pair, candleType, window, step, time, signal):
        return f"{pair}_{candleType}_{window}_{step}_{time}_{signal}.jpg"

    def get_plot_directory(self, pair, time, window):
        return os.path.join(self.plotDirectory, pair, time, str(window))

    def get_plot_filepath(self, pair, candleType, window, step, time, signal):
        name = plotWorks.get_plot_filename(self, pair, candleType, window, step, time, signal)
        directory = plotWorks.get_plot_directory(self, pair, time, window)
        return os.path.join(directory, name)


def run(pair, time):
    plotWorks(pair, time)


if __name__ == "__main__":
    pairs = ["ETHUSDT", 
             "BTCUSDT", 
             "LTCUSDT", 
             "BNBUSDT", 
             "XRPUSDT", 
             "XLMUSDT", 
             "ADAUSDT", 
             "NEOUSDT", 
             "ATOMUSDT", 
             "BATUSDT", 
             "DOGEUSDT", 
             "BCHUSDT"]
    timePeriods = ['5m', '10m', '15m', '30m', '1h', '2h', '4h', '1d']
    num_cores = multiprocessing.cpu_count()

    for pair in pairs:
        p = Parallel(n_jobs=num_cores)(delayed(target=run(), args=(pair, timePeriods.pop())))
        p.start()