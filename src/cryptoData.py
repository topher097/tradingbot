""" V2.0 of plotCreator.py to utilize parallel processing """

import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick2_ohlc
import numpy as np
import pandas as pd
from dataclasses import dataclass
import os
import pickle
import json
from pathlib import Path
from loggerSettings import plotlogger
import chime
from Technical import Technical as tech
from joblib import Parallel, delayed
import multiprocessing

import matplotlib
matplotlib.use("Cairo")

@dataclass
class paramHandler:
    """ Parameters for creating plots from data """
    dataDirectory : str = "F:\MarketData\CRYPTO\pickle"         # Directory of where data files can be found
    methodFileName : str = "bin\\methods.json"                  # Path of where methods JSON is located
    plotDirectory : str = "F:\MarketData\CRYPTO\plotsV2"        # Directory of where to save plots
    maxPlots : int = 100000         # Maximum number of plots to create for each file
    window : int = 30               # Window size for the plots
    step : int = 1                  # Step through data for plots
    candleType : str = 'ha'         # Candlestick type, either Heiken Ashi (ha) or regular OHCL (ohcl) or Renko (renko)
    save : bool = True              # Whether to save the plots or not
    showN : int = 0                 # How many plots to show, if any
    w : int = 256                   # Pixel width of plot image
    h : int = 256                   # Pixel height of plot image

    def __init__(self) -> None:
        self.dataFiles : set = set([f for f in os.listdir(paramHandler.dataDirectory) if os.path.isfile(os.path.join(paramHandler.dataDirectory, f))])
        self.basedir : str = Path(__file__).parent.parent
        self.methodFilePath : str = os.path.join(self.basedir, paramHandler.methodFileName)



"""Candlestick calculators"""
@dataclass
class CalculateHeikenAshiCandlesticks:

    def calculate_candles(klines: pd.DataFrame):
        """Calculating the Heiken Ashi candles from OHCLV data and overwriting."""
        # Make new dataframe to for HA data
        df = pd.DataFrame(index=klines.index, columns=['open', 'high', 'close', 'low', 'volume'])
        df.index.name = 'date'
        for c in klines.columns:
            klines[c] = pd.to_numeric(klines[c], downcast="float")
        # Calculate HA close
        df['close'] = (klines['open'] + klines['high'] + klines['low'] + klines['close']) / 4
        # Calculate HA open
        for i in range(0, len(klines)):
            if i == 0:
                df['open'].iat[i] = (klines['open'].iat[i] + klines['close'].iat[i]) / 2
            else:
                df['open'].iat[i] = (klines['open'].iat[i-1] + klines['close'].iat[i-1]) / 2
        # Calculate HA high
        df['high']=df[['open','close','high']].max(axis=1)
        # Calculate HA low
        df['low']=df[['open','close','low']].min(axis=1)
        # Add volume
        df['volume'] = klines['volume']
        # Return the klines data with the HA candles overwriting the regular OHCL candles
        return df

@dataclass
class CalculateRenkoCandlesticks:
    klines : pd.DataFrame
    def calculate_candles(self, klines):
        """Calculating the Renko block candles and adding to the klines dataframe."""
        # Still need to code this...
        return klines


"""Data aggregation for candlestick data"""
class DataAggregation:
    def aggregate_candlesticks(self, klines, timeframe):
        df = klines
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], downcast="float")
        ohlc = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': np.sum}
        df_new = df.resample(timeframe, offset=0).agg(ohlc).dropna()
        return df_new       

@dataclass
class Aggregate5Minute:
    """Aggregate the kline data into 5 minute candles."""
    klines : pd.DataFrame
    def aggregate_candlesticks(self, klines):
        return DataAggregation.aggregate_candlesticks(self, klines, '5MIN')

@dataclass
class Aggregate10Minute:
    """Aggregate the kline data into 10 minute candles."""
    klines : pd.DataFrame
    def aggregate_candlesticks(self, klines):
        return DataAggregation.aggregate_candlesticks(self, klines, '10MIN')

@dataclass
class Aggregate15Minute:
    """Aggregate the kline data into 15 minute candles."""
    klines : pd.DataFrame
    def aggregate_candlesticks(self, klines):
        return DataAggregation.aggregate_candlesticks(self, klines, '15MIN')

@dataclass
class Aggregate30Minute:
    """Aggregate the kline data into 30 minute candles."""
    klines : pd.DataFrame
    def aggregate_candlesticks(self, klines):
        return DataAggregation.aggregate_candlesticks(self, klines, '30MIN')

@dataclass
class Aggregate1Hour:
    """Aggregate the kline data into 1 hour candles."""
    klines : pd.DataFrame
    def aggregate_candlesticks(self, klines):
        return DataAggregation.aggregate_candlesticks(self, klines, '1H')

@dataclass
class Aggregate2Hour:
    """Aggregate the kline data into 2 hour candles."""
    klines : pd.DataFrame
    def aggregate_candlesticks(self, klines):
        return DataAggregation.aggregate_candlesticks(self, klines, '2H')

@dataclass
class Aggregate4Hour:
    """Aggregate the kline data into 4 hour candles."""
    klines : pd.DataFrame
    def aggregate_candlesticks(self, klines):
        return DataAggregation.aggregate_candlesticks(self, klines, '4H')

@dataclass
class Aggregate1Day:
    """Aggregate the kline data into 1 day candles."""
    klines : pd.DataFrame
    def aggregate_candlesticks(self, klines):
        return DataAggregation.aggregate_candlesticks(self, klines, '1D')

@dataclass
class Aggregate1Week:
    """Aggregate the kline data into 1 week candles."""
    klines : pd.DataFrame
    def aggregate_candlesticks(self, klines):
        return DataAggregation.aggregate_candlesticks(self, klines, '1W')


"""Methods calculators"""
@dataclass
class MethodsCalculator:
    methods : dict
    klines : pd.DataFrame

    def calculate_methods(klines, methods):
        df = pd.DataFrame(index=klines.index)
        df.index.name = 'date'
        for method in methods.keys():
            name = methods[method]['name']
            attributes = methods[method]
            def calcMethods(name):
                if 'PSAR' in name:      return tech.getParabolicSAR(klines, attributes)
                elif 'SMA' in name:     return tech.getSMA(klines, attributes)
                elif 'EMA' in name:     return tech.getEMA(klines, attributes)
                elif 'MA' in name:      return tech.getMA(klines, attributes)
                elif "BOLL" in name:    return tech.getBollingerBands(klines, attributes)
                elif 'RSI' in name:     return tech.getRSI(klines, attributes)
                elif 'ROC' in name:     return tech.getROC(klines, attributes)
                elif 'MACD' in name:    return tech.getMACD(klines, attributes)
                elif 'MFI' in name:     return tech.getMFI(klines, attributes)
                elif 'MOM' in name:     return tech.getMOM(klines, attributes)
                else:                   raise Exception(f"'{name}' is not recognized")
            # Calculate the method
            try:
                df[name] = calcMethods(name)
            except Exception as e:
                plotlogger.error(e)
        # Remove any lines with NaN
        df.dropna(how='any', axis=0, inplace=True)
        # Add methods dataframe to the kline dataframe and return
        klines = pd.merge(klines, df, left_index=True, right_index=True)
        return klines

    

@dataclass
class dataWorks(paramHandler):

    def __init__(self, pair, time) -> None:
        super().__init__()
        # self.methods = get_methods(self.methodFileName, self.methodFilePath)    # Get methods for the data
        self.OHCLVklines = dataWorks.get_kline_data(self, pair, time)           # Get OHCLV candlestick data for pair and timePeriod       

    def get_kline_data(self, pair, time):
        """Get the klines data for the given pair and time period, 
           if file for that period for pair does not exist, 
           aggregate the one minute candle data and save to new file."""
        klines = pd.DataFrame()
        dataFilePath = dataWorks.get_file_filepath(self, pair, time)
        if os.path.isfile(dataFilePath):
            with open(dataFilePath, 'rb') as f:
                klines = pickle.load(f)
            plotlogger.debug(f"Loaded aggregated data from '{dataFilePath}'")
        else:
            dataFilePath = dataWorks.get_file_filepath(self, pair, '1m')
            with open(dataFilePath, 'rb') as f:
                oneKlines = pickle.load(f)
            try:
                if time == '5m':
                    klines = Aggregate5Minute.aggregate_candlesticks(self, klines=oneKlines)
                elif time == '10m':
                    klines = Aggregate10Minute.aggregate_candlesticks(self, klines=oneKlines)
                elif time == '15m':
                    klines = Aggregate15Minute.aggregate_candlesticks(self, klines=oneKlines)
                elif time == '30m':
                    klines = Aggregate30Minute.aggregate_candlesticks(self, klines=oneKlines)
                elif time == '1h':
                    klines = Aggregate1Hour.aggregate_candlesticks(self, klines=oneKlines)
                elif time == '2h':
                    klines = Aggregate2Hour.aggregate_candlesticks(self, klines=oneKlines)
                elif time == '4h':
                    klines = Aggregate4Hour.aggregate_candlesticks(self, klines=oneKlines)
                elif time == '1d':
                    klines = Aggregate1Day.aggregate_candlesticks(self, klines=oneKlines)
                elif time == '1w':
                    klines = Aggregate1Week.aggregate_candlesticks(self, klines=oneKlines)
                else:
                    raise Exception(f"Unknown timePeriod for candle aggregation")
                newdataFilePath = dataWorks.get_file_filepath(self, pair, time)
                with open(newdataFilePath, 'wb') as f:
                    pickle.dump(klines, f, protocol=pickle.HIGHEST_PROTOCOL)
                plotlogger.debug(f"Calculated and saved aggregated data to '{newdataFilePath}'")
            except Exception as e:
                plotlogger.error(e)
        return klines

    def get_data_filename(self, pair, time):
        return f"{pair}_{time}.pkl"

    def get_file_filepath(self, pair, time):
        name = dataWorks.get_data_filename(self, pair, time)
        return os.path.join(self.dataDirectory, name)


class plotWorks(paramHandler):
    """Creates a plot with the candlestick data and the methods for the data"""
    def __init__(self, pair, time, candleType, klines, methods):
        super().__init__() 
        # Object variables
        self.pair = pair
        self.time = time
        self.candleType = candleType
        self.klines = klines
        self.methods = methods
        # Colors
        self.colorGreen  = '#77d879'
        self.colorRed    = '#db3f3f'
        self.colors = ['blue', 'indigo', 'orange', 'gold', 'white', 'yellow', 'aqua', 'slategray']
        # Candle sizes
        self.candleWidth = 1.0
        self.wickWidth = self.candleWidth * 0.1875
        # Get the plot directory for pair
        self.plotPairDirectory = plotWorks.get_plot_directory(self, pair, time, self.window)
        # Calcualte the number of plots to generate
        self.numPlots = int(((len(self.klines) - self.window - 1)/self.step) - 1)
        if self.numPlots > self.maxPlots: self.numPlots = self.maxPlots
        #print(f"{self.numPlots=}, {len(self.klines)}, {self.window=}, {self.step=}, {self.maxPlots=}")

        # Plot
        plotWorks.plot(self, self.klines)

    # Plot candles
    def plot(self, klines:pd.DataFrame):
        # Get numpy array of times from index
        times = ((klines.index - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')).to_numpy()
        # Get set of plot files in 
        filesInDir = set(list(os.listdir(self.plotPairDirectory)))
        # Create figure and plot axes
        plt.ioff()
        fig = plt.figure(figsize=(self.w/1000,self.h/1000), dpi=1000, facecolor='black', edgecolor='black')
        ax1 = fig.add_subplot(1,1,1)
        # Run plot creator
        for i in range(1, self.numPlots):
            # Debug information
            if i%1000 == 0 or i == self.numPlots:
                plotlogger.debug(f"{i}/{self.numPlots}   {self.pair=}, {self.candleType=}, {self.time=}, {self.window=}")
            # Start and stop indeces of dataframe
            start = i*self.step
            stop  = i*self.step + self.window 
            # Signal
            signal = klines['signal'][stop]
            # Time index
            timeIndex = times[stop]
            # See if plot exists, if it does then continue, if not then continue with plot creation
            plotFileName = plotWorks.get_plot_filename(self, self.pair, self.candleType, self.window, self.step, timeIndex, signal)
            if plotFileName in filesInDir:
                continue
            plotFilePath = os.path.join(self.plotPairDirectory, plotFileName)
            try:
                # Plot candles for the window
                if self.candleType == "OHCLV" or self.candleType == "HA":
                    plotWorks.plot_candle_objects(self, ax1, klines, start, stop)
                else:
                    plotWorks.plot_renko_objects(self, ax1, klines, start, stop)
                # Plot the methods
                plotWorks.plot_methods(self, ax1, klines, self.methods, start, stop)
                # Remove whitespace around edges
                plt.gca().set_axis_off()
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0,0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.draw()
                # Save plot
                if self.save:
                    plt.savefig(plotFilePath, bbox_inches='tight', pad_inches=0, dpi=1000)
                else:
                    plt.show()
                # Clear figure axes 
                ax1.clear()
            except Exception as e:
                plotlogger.error(e)
        # Close all plot figures
        plt.close('all')

    def plot_candle_objects(self, ax1, klines:pd.DataFrame, start, stop):
        """Plot the candles on the plot axes"""
        lines, patches =  candlestick2_ohlc(ax=ax1,
                                            opens=klines['open'][start:stop],
                                            highs=klines['high'][start:stop],
                                            lows=klines['low'][start:stop],
                                            closes=klines['close'][start:stop],
                                            width=self.candleWidth,
                                            alpha=1.0,
                                            colorup=self.colorGreen, 
                                            colordown=self.colorRed)
        lines.set_linewidth(self.wickWidth)         # Set the width of the candle wick
        patches.set_edgecolor(c=None)               # Remove edges of candle objects

    def plot_renko_objects(self, ax1, klines: pd.DataFrame, start, stop):
        """Plot the renko boxes on the plot axes"""
        pass

    def plot_methods(self, ax1, klines:pd.DataFrame, methods:dict, start, stop):
        """Plot the methods on the plot axes"""
        # Get method names
        methodLabels = list(methods.keys())
        methodNames = [methods[method]['name'] for method in methodLabels]
        for method in methodLabels:
            methodName = methods[method]['name']
            plotAttributes = methods[method]['plot']
            if plotAttributes['bool'] == 'true':
                yData = klines[methodName].to_numpy(dtype='f8')[start:stop]
                xData = np.arange(yData.size)
                color = self.colors[methodNames.index(methodName)]
                if plotAttributes['type'] == 'line':
                    ax1.plot(xData, yData, c=color, lw=0.3, ls=plotAttributes['linestyle'])
                elif plotAttributes['type'] == 'scatter':
                    ax1.scatter(xData, yData, c=color, s=0.5, lw=0.1, marker='.', edgecolors=None )
                else:
                    raise Exception(f"Invalid plot type in methods.json '{plotAttributes['type']}'")

    def get_plot_filename(self, pair, candleType, window, step, time, signal):
        """Return the filename for the figure"""
        return f"{pair}_{candleType}_{window}_{step}_{time}_{signal}.jpg"

    def get_plot_directory(self, pair, time, window):
        """Return the directory to save figures"""
        plotDirectory = os.path.join(self.plotDirectory, pair, time, str(window))
        # If directory does not exist, create it
        if not os.path.isdir(plotDirectory):
            os.makedirs(plotDirectory)
        return plotDirectory 

    def get_plot_filepath(self, pair, candleType, window, step, time, signal):
        """Return the file path for the figure"""
        name = plotWorks.get_plot_filename(self, pair, candleType, window, step, time, signal)
        return os.path.join(self.plotPairDirectory, name)

def get_methods(methodFileName, methodFilePath):
    """Get the methods to compute on the data from the JSON file."""
    try:
        with open(methodFilePath, "r") as file:
            data    = file.read()
            methods = json.loads(data)
        #plotlogger.info(f"Successfully loaded methods from '{methodFileName}'")
        return methods
    except Exception as e:
        plotlogger.error(f"Failed to load methods, error: {e}")
    
def generate_signals(klines: pd.DataFrame, neutral_pctchg: float, shift: int):
    """Generate the signals give the candlestick data for given t+step time and neutral percent change cutoff"""
    # Function to calculate the signal for row
    def signalCalc(pctchg_at_step, npct):
        if pctchg_at_step <= -npct: s = "BEAR"
        elif -npct < pctchg_at_step < npct: s = "NEUTRAL"
        elif npct <= pctchg_at_step: s = "BULL"
        else: s = "NULL"
        return s
    # Calculate the percent change of close from kline data
    klines['close_pctchg'] = klines['close'].pct_change()
    # Calculate the signals and save to new dataframe
    signals = pd.DataFrame(signalCalc(row.close_pctchg, neutral_pctchg) for row in klines.itertuples())
    signals.index = klines.index
    signals.index.name = 'date'
    # Shift the signals given step
    signals = signals.shift(periods=shift, fill_value='NULL')
    # Add signals dataframe to the klines dataframe
    klines['signal'] = signals
    klines.dropna(how='any', axis=0, inplace=True)
    # Return the klines dataframe with added signals for given step
    return klines

def get_proper_kline_data(OHCLVklines, methods, candleType):
    """Return candlestick data with the calculated methods."""
    if candleType == 'OHCLV':
        return MethodsCalculator.calculate_methods(klines=OHCLVklines, methods=methods)
    elif candleType == 'HA':
        return MethodsCalculator.calculate_methods(klines=CalculateHeikenAshiCandlesticks.calculate_candles(OHCLVklines), methods=methods)
    elif candleType == 'RENKO':
        return MethodsCalculator.calculate_methods(klines=CalculateRenkoCandlesticks.calculate_candles(OHCLVklines), methods=methods)

def clean_1m_klines(dataFilePath, dataFiles, dataDirPath, pair):
    with open(dataFilePath, 'rb') as file:
        old_klines = pickle.load(file)
    klines = pd.DataFrame(index=old_klines.index)   
    klines['open'] = old_klines['open']
    klines['high'] = old_klines['high']
    klines['close'] = old_klines['close']
    klines['low'] = old_klines['low']
    klines['volume'] = old_klines['volume']
    with open(dataFilePath, 'wb') as file:
        pickle.dump(klines, file, protocol=pickle.HIGHEST_PROTOCOL)

    for file in dataFiles:
        if pair in file and file not in dataFilePath:
            print(file)
            os.remove(os.path.join(dataDirPath, file))
            
    

def run(pair: str, time: str, plot: bool, candleType: str):
    """
    Runs the data loading, calculates the methods and candlesticks given candleType, 
    then calculates signal calcs for the calculated candlestick data

    Args:
        pair (str): trading pair
        time (str): time period for candle
        plot (bool): boolean to create the plots or not
        candleType (str): what candlestick type to use (OHCLV, HA, Renko)
    """
    # Get the OHCLV candlestick data from file for the correct pair and timePeriod
    data = dataWorks(pair, time)
    OHCLVklines = data.OHCLVklines

    # Get the dataframe for the correct candlestick type with methods from methods JSON file
    handle = paramHandler()
    methods = get_methods(handle.methodFileName, handle.methodFilePath)
    klines = get_proper_kline_data(OHCLVklines, methods, candleType)

    # Generate the signals for the kline data given percent change cutoff and t+step 
    klines = generate_signals(klines, neutral_pctchg=0.0001, shift=1)

    # If wanting to create plots
    if plot:
        plotWorks(pair, time, candleType, klines, methods)


if __name__ == "__main__":
    # Pairs to run
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

    # dataDirPath = "F:\MarketData\CRYPTO\pickle"
    # dataFiles = [f for f in os.listdir(dataDirPath) if os.path.isfile(os.path.join(dataDirPath, f))]
    # for pair in pairs:
    #     handle = paramHandler()
    #     name = f"{pair}_1m.pkl"
    #     dataFilePath = os.path.join(handle.dataDirectory, name)
    #     clean_1m_klines(dataFilePath, dataFiles, dataDirPath, pair)
    # exit()

    # Time periods to run
    timePeriods = ['5m', '10m', '15m', '30m', '1h', '2h', '4h', '1d']
    
    # Determine if wanting plots to be generated
    plot = True
    # Determine what candlestick type to use: OHCLV (Regular candles), HA (Heiken Ashi), RENKO (Renko)
    candleType = 'HA'

    # Run the pairs and each timeframe in parallel processes
    num_cores = multiprocessing.cpu_count()
    #num_cores = 1
    for pair in pairs:
        exec = Parallel(n_jobs=num_cores, verbose=10)
        tasks = (delayed(run)(pair, timePeriod, plot, candleType) for timePeriod in timePeriods)
        p = exec(tasks)
        plotlogger.debug(f"Finished creating plots for {pair}")
        #chime.play_tone(freq=500, burst_time=0.25, bursts=2) 
    chime.play_tone(freq=720, burst_time=2, bursts=1)
    exit()