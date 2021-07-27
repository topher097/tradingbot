import investpy
import pandas as pd
import pickle
from bs4 import BeautifulSoup
import csv
import os
from os import mkdir
from os.path import exists, join
import urllib.request as request

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
            records.append([symbol, name, sector])
            symbols.append(symbol + "\n")

    header = ["Symbol", "Name", "Sector"]
    writer = csv.writer(open(dataPathCSV, "w"), lineterminator="\n")
    writer.writerow(header)
    # Sorting ensure easy tracking of modifications
    records.sort(key=lambda s: s[1].lower())
    writer.writerows(records)

    with open(dataPathTXT, "w") as f:
        # Sorting ensure easy tracking of modifications
        symbols.sort(key=lambda s: s[0].lower())
        f.writelines(symbols)

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
    tickers = df['Symbol'].tolist()
    tickers = [ticker.upper().replace(".", "") for ticker in tickers]
    return tickers

def getHistoricalData(tickers):
    candleDataDir = "E:\MarketData\STOCK"
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



if __name__ == "__main__":
    datadir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'bin'))
    dataPathCSV = os.path.join(datadir, 'stockTickers.csv')
    dataPathTXT = os.path.join(datadir, 'stockTickers.txt')

    if not os.path.isdir("scripts/tmp"):
        os.makedirs("scripts/tmp")

    source = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    cache = os.path.join("scripts/tmp", "List_of_S%26P_500_companies.html")
    
    # Get tickers and other info and save to csv file
    getCSV()
    # Get list of tickers from csv file
    tickers = getTickers()
    # Get historical data for tickers
    getHistoricalData(tickers)



