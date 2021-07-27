"""
https://towardsdatascience.com/how-and-why-i-got-75gb-of-free-foreign-exchange-tick-data-9ca78f5fa26c
"""


# libraries you will need
import datetime as dt
import numpy as np
import pandas as pd
#import tables as tb
import requests
import fxcmpy
from fxcmpy import fxcmpy_tick_data_reader as tdr
import os
import gzip
from io import StringIO
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import chime

# Construct the years, weeks and symbol lists required for the scraper.
years = [2017, 2018, 2019, 2020, 2021]
weeks = list(range(53))
symbols = []
for pair in tdr.get_available_symbols():
    if pair not in symbols:
        symbols.append(pair)

# Scrape data
directory = "E:\MarketData\FX"
scrape_directory = os.path.join(directory, 'scrape')
total = len(symbols)*len(weeks)*len(years)
count = 0

for symbol in symbols:    
    for year in years:
        for week in weeks:
            count += 1
            fileName = f'{symbol}_{year}_w{week}.csv.gz'
            savePath = os.path.join(scrape_directory, fileName)
            try:
                if not os.path.isfile(savePath):
                    print(f'{count}/{total}    Downloading {fileName}')
                    url = f"https://tickdata.fxcorporate.com/{symbol}/{year}/{week}.csv.gz"
                    r = requests.get(url, stream=True)
                    with open(savePath, 'wb') as file:
                        for chunk in r.iter_content(chunk_size=1024):
                            file.write(chunk)
            except Exception as e:
                print(e)
    chime.play_tone(freq=1000, bursts=3, burst_time=0.1)

# Check all the files for each currency pair was downloaded (should be 104 for each)
total = 0
for symbol in symbols:
    count = 0
    for file in os.listdir(scrape_directory):
        if file[:6] == symbol:
            count+=1
    total += count
    print(f"{symbol} files downloaded = {count} ")
print(f"\nTotal files downloaded = {total}")

# Save files to the HDF5 file
hdf5_file = "fx.h5"
for file in os.listdir(scrape_directory):
    if file.endswith('.gz'):
        print(f"\nExtracting: {file}")
        
        # extract gzip file and assign to Dataframe
        codec = 'utf-16'
        hdf5Path = os.path.join(directory, hdf5_file)
        gzPath = os.path.join(scrape_directory, file)
        try:
            with gzip.GzipFile(gzPath, 'r') as f:
                data = f.read()
            data_str = data.decode(codec)
            data_pd = pd.read_csv(StringIO(data_str))
            
            # pad missing zeros in microsecond field
            data_pd['DateTime'] = data_pd.DateTime.str.pad(26, side='right', fillchar='0')
            
            # assign Datetime column as index
            data_pd.set_index('DateTime', inplace=True)
            
            # sample start and end to determine date format
            sample1 = data_pd.index[1]
            sample2 = data_pd.index[-1]
            
            # determine datetime format and supply srftime directive
            for row in data_pd:
                if data_pd.index[3] == '/':
                    if sample1[0:2] == sample2[0:2]:
                        data_pd.index = pd.to_datetime(data_pd.index, format="%m/%d/%Y %H:%M:%S.%f")
                    elif sample1[3:5] == sample2[3:4]:
                        data_pd.index = pd.to_datetime(data_pd.index, format="%d/%m/%Y %H:%M:%S.%f")
                elif data_pd.index[5] == '/':
                    if sample1[9:11] == sample2[9:11]:
                        data_pd.index = pd.to_datetime(data_pd.index, format="%Y/%d/%m %H:%M:%S.%f")
                    elif sample1[6:8] == sample2[6:8]:
                        data_pd.index = pd.to_datetime(data_pd.index, format="%Y/%m/%d %H:%M:%S.%f")
        
            #print("\nDATA SUMMARY:")
            #print(data_pd.info())
            
            # Load data into database
            store = pd.HDFStore(hdf5Path)
            symbol = file[:6]
            store.append(symbol, data_pd, format='t') 
            store.flush()
            print("\nH5 DATASTORE SUMMARY:")
            print(store.info()+"\n"+"-"*75)
            store.close()
        except Exception as e:
            print(f"error extracting data from {file}, error: {e}")

# Play chime when done
chime.play_tone(freq=520, bursts=5, burst_time=0.5)

