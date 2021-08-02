import matplotlib.pyplot as plt
import plotly
import numpy as np
import pandas as pd
import dataclasses
import os
import pickle
import json
import math
from pathlib import Path
from loggerSettings import plotlogger

from Technical import Technical as tech


class data:
    def __init__(self, klines) -> None:
        self.open = klines["open"]
        self.close = klines["close"]
        self.high= klines["high"]
        self.low = klines["low"]
        self.volume = klines["volume"]
        self.ha_open = klines["ha_open"]
        self.ha_close = klines["ha_close"]
        self.ha_high = klines["ha_high"]
        self.ha_low = klines["ha_low"]

    
class plotter:
    def __init__(self, pair, dataFilePath=None, dataDirPath=None, window=60, step=5, maxImageCount=1, showN=0) -> None:
        """ Load methods and data """
        # Other filepaths
        self.basedir = Path(__file__).parent.parent
        self.methodFileName = "bin\\methods.json"
        self.methodFilePath = os.path.join(self.basedir, self.methodFileName)

        # Load methods for plotter
        plotter.loadMethodsJSON(self)
        # Get data from file
        if not os.path.isfile(dataFilePath) or not dataFilePath or not dataDirPath:
            raise Exception(f"File '{dataFilePath}' either doesn't exist or cannot be found or data directory not passed")
        else:
            with open(dataFilePath, 'rb') as file:
                klines = pickle.load(file)
                defaultColumnNames = klines.columns
                plotlogger.debug(f"Loaded data from '{dataFilePath}'")
                for c in defaultColumnNames:
                    pass
                    klines[c] = pd.to_numeric(klines[c], downcast="float")
    
        """ Calculate methods to dataframe """
        # Calculate data from methods and add to dataframe
        df = pd.DataFrame(index=klines.index)
        df.index.name = 'date'
        for method in self.methods.keys():
            name = self.methods[method]['name']
            attributes = self.methods[method]
            def calcMethods(self, name):
                if 'PSAR' in name:      return tech.getParabolicSAR(self, klines, attributes)
                elif 'SMA' in name:     return tech.getSMA(self, klines, attributes)
                elif 'EMA' in name:     return tech.getEMA(self, klines, attributes)
                elif 'MA' in name:      return tech.getMA(self, klines, attributes)
                elif "BOLL" in name:    return tech.getBollingerBands(self, klines, attributes)
                elif 'RSI' in name:     return tech.getRSI(self, klines, attributes)
                elif 'ROC' in name:     return tech.getROC(self, klines, attributes)
                elif 'MACD' in name:    return tech.getMACD(self, klines, attributes)
                elif 'MFI' in name:     return tech.getMFI(self, klines, attributes)
                elif 'MOM' in name:     return tech.getMOM(self, klines, attributes)
                else:                   raise Exception(f"'{name}' is not recognized")
            # Calculate the method
            df[name] = calcMethods(self, name)
        # Remove any lines with NaN
        df.dropna(how='any', axis=0, inplace=True)
        print(df.head())
             

        """ Create and save plots """
        # Calcualte how many plots to create for dataset
        numPlots = math.floor((klines.size - window)/step)
        if numPlots > maxImageCount: numPlots = maxImageCount   # Make sure max number of plots isn't surpassed
        for i in range(0, numPlots):
            # Create plots given window and step
            xData = klines.index(axis=0)
            candlesticks = 0


            # Create plot image filename and path
            plotFileName = dataFilePath
            # Only draw n amount of plots to show (if showing)
            if i < showN:
                plt.draw()
        
        # If showing plots
        if showN > 0:
            plt.show()


    
    def loadMethodsJSON(self):
        # Load the methods and their values from JSON file
        try:
            with open(self.methodFilePath, "r") as file:
                data            = file.read()
                self.methods    = json.loads(data)
            plotlogger.info(f"Successfully loaded methods from '{self.methodFileName}'")
        except Exception as e:
            plotlogger.error(f"Failed to load methods, error: {e}")

    def createChunks(self):
        pass

        

if __name__ == "__main__":
    # Get file of data to plot
    dataDirPath = "D:\MarketData\CRYPTO\pickle"
    dataFiles = [f for f in os.listdir(dataDirPath) if os.path.isfile(os.path.join(dataDirPath, f))]
    
    timePeriod = '1d'
    dataFiles = [file for file in dataFiles if timePeriod in file]

    for file in dataFiles:
        split = file.split('_')
        pair = split[0]
        try:
            window = 30
            step = 1
            maxImageCount = 100
            showN = 1
            plotter(pair=pair, dataFilePath=os.path.join(dataDirPath, file), dataDirPath=dataDirPath, window=window, step=step, maxImageCount=maxImageCount, showN=showN)
        except Exception as e:
            plotlogger.error(e)
        exit()