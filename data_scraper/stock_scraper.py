import investpy
from get_all_tickers import get_tickers as gt
import pandas as pd
import pickle
from bs4 import BeautifulSoup
import csv
import os
import sys
import urllib.request as request
from collections import OrderedDict

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(os.path.join(parentdir, 'src'))
import chime

"""
Get list of stock tickers
Then get historical data as far back as possible
Then save to a csv file and pickle file (pd dataframe)
"""


def retrieve():
    request.urlretrieve(source, cache)

def extract():
    source_page = open(cache, encoding="utf8").read()
    soup = BeautifulSoup(source_page, "html.parser")
    table = soup.find("table", {"class": "wikitable sortable"})

    # Fail now if we haven't found the right table
    header = table.findAll("th")
    if header[0].text.rstrip() != "Symbol" or header[1].string != "Security":
        raise Exception("Can't parse Wikipedia's table!")

    # Retrieve the values in the table
    records = []
    symbols = []
    rows = table.findAll("tr")
    for row in rows:
        fields = row.findAll("td")
        if fields:
            symbol = fields[0].text.rstrip()
            # fix as now they have links to the companies on WP
            name = fields[1].text.replace(",", "")
            sector = fields[3].text.rstrip()
            records.append((symbol, name, sector))
            symbols.append(symbol + "\n")

    header = ["symbol", "name", "sector"]
    df = pd.DataFrame(records, columns=header)
    df.to_csv(dataPathCSV)


def getCSV():
    retrieve()
    extract()
    try:
        os.rmdir('/scripts')
    except:
        pass

def getTickers():
    with open(dataPathCSV, 'r') as file:
        df = pd.read_csv(file)
    tickers = df['symbol'].tolist()
    tickers = [ticker.upper().replace(".", "") for ticker in tickers]
    return tickers

def getHistoricalData(tickers, chime_=False):
    total = len(tickers)
    count = 0
    for ticker in tickers:
        count += 1
        dataPathPickle = os.path.join(candleDataDir, f"pickle\{ticker}.pkl")
        dataPathCSV = os.path.join(candleDataDir, f"csv\{ticker}.csv")
        try:
            if not os.path.isfile(dataPathCSV) and not os.path.isfile(dataPathPickle):
                df = investpy.get_stock_historical_data(stock=ticker,
                                                country='United States',
                                                from_date='01/01/2000',
                                                to_date='07/01/2021')

                # Drop the "currency" column and make all lowercase
                df = df.drop('Currency', axis=1)
                df.columns = [x.lower() for x in df.columns]
                df = df.rename_axis('date')

                # Save to pickle file
                with open(dataPathPickle, 'wb') as file:
                    pickle.dump(df, file, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Save to csv file
                df.to_csv(dataPathCSV, sep=',')
                print(f"{count}/{total}   Saved historical data for '{ticker}'")
            else:
                print(f"{count}/{total}   Data for ticker '{ticker}' already exists, skipping")
        except Exception as e:
            print(f"Error with ticker '{ticker}, error: {e}'")
    print(f"Done saving data for S&P500")
    if chime_:
        chime.play_tone(freq=1000, bursts=5, burst_time=0.1)


if __name__ == "__main__":
    """ S&P500 tickers """
    datadir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'bin'))
    dataPathCSV = os.path.join(datadir, 'sp500Tickers.csv')

    if not os.path.isdir("scripts/tmp"):
        os.makedirs("scripts/tmp")

    source = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    cache = os.path.join("scripts/tmp", "List_of_S%26P_500_companies.html")
    
    # Get tickers and other info and save to csv file
    getCSV()
    # Get list of tickers from csv file
    tickers = getTickers()
    # Get historical data for tickers
    candleDataDir = "E:\MarketData\STOCK\SP500"
    #getHistoricalData(tickers, chime_=False)

    """ TOP 2500 TICKERS """
    tickers = gt.get_biggest_n_tickers(10000)
    tickers = list(OrderedDict.fromkeys(tickers))[0:2500]    # Remove duplicates, get top 2500
    # Save dataframe of tickers in order
    dataPathCSV = os.path.join(datadir, 'top2500Tickers.csv')
    df = pd.DataFrame(tickers, columns=['symbol'])
    df.to_csv(dataPathCSV)
    candleDataDir = "E:\MarketData\STOCK\TOP2500"
    #getHistoricalData(tickers, chime_=True)
    



