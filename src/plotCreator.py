import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick2_ohlc
import mplfinance as mpl
import numpy as np
import pandas as pd
import dataclasses
import os
import pickle
import json
import math
from pathlib import Path
from loggerSettings import plotlogger
import chime
from datetime import datetime
import time as t
from Technical import Technical as tech
from joblib import Parallel, delayed
import multiprocessing

import matplotlib
matplotlib.use("Agg")

   
class plotter:
    def __init__(self, pair, timePeriod=None, plotDirPath=None, w=50, h=50, save=True, dataType='ohcl', dataFilePath=None, dataDirPath=None, window=60, step=5, maxImageCount=1, showN=0) -> None:
        """ Load methods and data """
        # Other filepaths
        self.basedir = Path(__file__).parent.parent
        self.methodFileName = "bin\\methods.json"
        self.methodFilePath = os.path.join(self.basedir, self.methodFileName)
        # Load methods for plotter
        plotter.loadMethodsJSON(self)
        # Get data from file
        if not os.path.isfile(dataFilePath) or not dataFilePath or not dataDirPath or not timePeriod or not plotDirPath:
            raise Exception(f"File '{dataFilePath}' either doesn't exist or cannot be found or data directory not passed")
        else:
            with open(dataFilePath, 'rb') as file:
                klines = pickle.load(file)
                defaultColumnNames = klines.columns
                plotlogger.debug(f"Loaded data from '{dataFilePath}'")
                for c in defaultColumnNames:
                    try:
                        klines[c] = pd.to_numeric(klines[c], downcast="float")
                    except:
                        pass

        """ Calculate the signals """
        # Function to calculate the signal for row
        def signalCalc(pct_chg_next, npct):
            if pct_chg_next <= -npct: s = "BEAR"
            elif -npct < pct_chg_next < npct: s = "NEUTRAL"
            elif npct <= pct_chg_next: s = "BULL"
            else: s = "NULL"
            return s
        
        def loopHA(df, npct):
            new_df = pd.DataFrame(signalCalc(row.ha_pct_chg, npct) for row in df.itertuples())
            new_df.index = df.index
            return new_df
        
        def loopOHCL(df, npct):
            new_df = pd.DataFrame(signalCalc(row.pct_chg, npct) for row in df.itertuples())
            new_df.index = df.index
            return new_df

        if 'ha_signal' not in klines.columns and 'signal' not in klines.columns:
        #if True:
            plotlogger.info(f"Signals not determined for '{dataFilePath}', calculating...")
            npct = 0.0001   # Neutral is +- 0.01%
            klines['ha_pct_chg']    = klines['ha_close'].pct_change()
            klines['pct_chg']       = klines['close'].pct_change()
            klines.dropna(how='any', axis=0, inplace=True)
            # Calc for OHCL
            signal = loopOHCL(klines, npct)
            # Calc for HA
            ha_signal = loopHA(klines, npct)
            klines['signal'] = signal
            klines['ha_signal'] = ha_signal
            with open(dataFilePath, 'wb') as f:
                pickle.dump(klines, f, protocol=pickle.HIGHEST_PROTOCOL)
                plotlogger.info(f"Added signals to '{dataFilePath}'")
            
        else:
            plotlogger.info(f"Signals already determined for dataset")
    
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
        # Add df to klines, drop any index rows that don't have methods data
        klines = pd.merge(klines,df, left_index=True, right_index=True)#.query('_merge=="left_only"').drop('_merge', axis=1)

        """ Create and save plots """
        # Plot dir path
        plotFileDir = os.path.join(plotDirPath, pair, timePeriod, str(window))
        if not os.path.isdir(plotFileDir):
            os.makedirs(plotFileDir)
            plotlogger.debug(f"Created {plotFileDir}")
            filesInDir = None
        else:
            filesInDir = set(list(os.listdir(plotFileDir)))
        
        # Signals
        ha_signals = klines['ha_signal'].to_numpy()
        signals = klines['signal'].to_numpy()
        #print(f"index dtype: {klines.index.dtype}, first elements={klines.index[0:5]}")
        if len(klines.index) < 10:
            raise Exception('No indexes in data')
        times = ((klines.index - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')).to_numpy()

        # Calcualte how many plots to create for dataset
        numPlots = int(((len(klines) - window -1)/step) - 1)
        if numPlots > maxImageCount: numPlots = maxImageCount   # Make sure max number of plots isn't surpassed
        self.colorGreen  = '#77d879'
        self.colorRed    = '#db3f3f'
        self.candleWidth = 1.0

        # Create plots
        for i in range(1, numPlots):
            if i%1000 == 0 or i == numPlots-1:
                plotlogger.debug(f"{i}/{numPlots}   {pair=}, {dataType=}, {timePeriod=}, {window=}")
            start       = i*step
            stop        = i*step + window      
            try:
                # Make file name
                if dataType == 'ohcl':
                    sig = signals[stop]
                elif dataType == 'ha':
                    sig = ha_signals[stop]
                else:
                    sig = "NULL"
                    raise Exception(f"Invalid plot data type '{dataType}'")
                time = times[stop]
                label = f"{pair}_{dataType}_{window}_{step}_{time}_{sig}"

                # If the plot file already exists, go to next
                plotFileName = f"{label}.jpg"    # JPG is smaller, but is not lossless like PNG
                plotFilePath = os.path.join(plotFileDir, plotFileName)
                if plotFileName in filesInDir:
                    continue
                
                # Create plots given window and step
                fig = plt.figure(num=i, figsize=(w/1000,h/1000), dpi=1000, facecolor='black', edgecolor='black')
                ax1 = fig.add_subplot(1,1,1)
                if dataType == 'ohcl':
                    opens  = klines['open'][start:stop]
                    highs  = klines['high'][start:stop]
                    lows   = klines['low'][start:stop]
                    closes = klines['close'][start:stop]
                    lines, patches =  candlestick2_ohlc(ax=ax1,
                                                        opens=opens,
                                                        highs=highs,
                                                        lows=lows,
                                                        closes=closes,
                                                        width=self.candleWidth,
                                                        alpha=1.0,
                                                        colorup=self.colorGreen, 
                                                        colordown=self.colorRed)
                    lines.set_linewidth(self.candleWidth*0.1875)
                    patches.set_edgecolor(c=None)
                elif dataType == 'ha':
                    opens  = klines['ha_open'][start:stop]
                    highs  = klines['ha_high'][start:stop]
                    lows   = klines['ha_low'][start:stop]
                    closes = klines['ha_close'][start:stop]
                    lines, patches =  candlestick2_ohlc(ax=ax1,
                                                        opens=opens,
                                                        highs=highs,
                                                        lows=lows,
                                                        closes=closes,
                                                        width=self.candleWidth,
                                                        alpha=1.0,
                                                        colorup=self.colorGreen, 
                                                        colordown=self.colorRed)
                    lines.set_linewidth(self.candleWidth*0.1875)
                    patches.set_edgecolor(c=None)
                
                # Plot the methods
                for j in range(len(df.columns)):
                    methodName = df.columns[j]
                    plotAttributes = self.methods[f'method{j+1}']['plot']
                    if plotAttributes['bool'] == 'true':
                        yData = klines[methodName].to_numpy(dtype='f8')[i:i+window]
                        xData = np.arange(yData.size)
                        color = colors[j]
                        if plotAttributes['type'] == 'line':
                            ax1.plot(xData, yData, c=color, lw=0.3, ls=plotAttributes['linestyle'])
                        elif plotAttributes['type'] == 'scatter':
                            ax1.scatter(xData, yData, c=color, s=0.5, lw=0.1, marker='.', edgecolors=None )
                        else:
                            raise Exception(f"Invalid plot type in methods.json '{plotAttributes['type']}'")
                        
                # Remove whitespace around edges
                plt.gca().set_axis_off()
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0,0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.draw()
                
                # Create plot image filename and path
                if save:
                    plt.savefig(plotFilePath, bbox_inches='tight', pad_inches=0, dpi=1000)
                if numPlots >= 10:
                    plt.close('all')
            except Exception as e:
                print(f"{i} {e}") 
        

    def loadMethodsJSON(self):
        # Load the methods and their values from JSON file
        try:
            with open(self.methodFilePath, "r") as file:
                data            = file.read()
                self.methods    = json.loads(data)
            plotlogger.info(f"Successfully loaded methods from '{self.methodFileName}'")
        except Exception as e:
            plotlogger.error(f"Failed to load methods, error: {e}")
        

if __name__ == "__main__":
    # List of plot colors for methods
    colors = ['blue', 'indigo', 'orange', 'gold', 'white', 'yellow', 'aqua', 'slategray']

    # Get file of data to plot
    plotDirPath = "F:\MarketData\CRYPTO\plots"
    dataDirPath = "F:\MarketData\CRYPTO\pickle"
    dataFiles = [f for f in os.listdir(dataDirPath) if os.path.isfile(os.path.join(dataDirPath, f))]    
    timePeriods = ['5m', '10m', '15m', '30m', '1h', '2h', '4h', '1d']
    for timePeriod in timePeriods:
        dataFilesTime = [file for file in dataFiles if f"_{timePeriod}" in file]
        for file in dataFilesTime:
            split = file.split('_')
            pair = split[0]
            try:
                window = 30              # Window size
                step = 1                # Step to move window
                maxImageCount = 25000       # Plots to render
                showN = 0               # Plots to show
                w = 256                 # Pixels
                h = 256                 # Pixels
                dataType = 'ohcl'       # Either classic 'ohcl' or 'ha' for heiken ashi
                plotlogger.debug(f'Running plotter for {pair}')
                save = True
                plotter(pair=pair, timePeriod=timePeriod, w=w, h=h, save=save, dataType=dataType, dataFilePath=os.path.join(dataDirPath, file), dataDirPath=dataDirPath, plotDirPath=plotDirPath, window=window, step=step, maxImageCount=maxImageCount, showN=showN)
                plotlogger.debug(f"Finished creating plots for {pair}")
            except Exception as e:
                with open("D:\\Github\\tradingbot\\src\\fix_data.csv", 'a') as f:
                    f.write(os.path.join(dataDirPath, file) + '\n')
                plotlogger.error(e)
            # If showing plots
            if showN > 0:   
                showFigs = plt.get_figlabels()[showN:]
                [plt.close(i) for i in showFigs]
                plt.show()
            
        chime.play_tone(freq=500, burst_time=0.25, bursts=2) 
    chime.play_tone(freq=720, burst_time=2, bursts=1)
    exit()
