import pandas as pd
import os


def get_tickers_list(path, scalemarketcap_number):
    # Get tickers names from "quandl_tickers" and put it in "mega_tickers" file
    if os.path.isfile('data/quandl_tickers.csv'):
        data = pd.read_csv('data/quandl_tickers.csv')
    else:
        data = pd.read_csv('../../data/quandl_tickers.csv')
    tickers_list = []
    if scalemarketcap_number != 8:   # 8 are ETFs
        # get category list
        capit_category_list = sorted(list(set(data['scalemarketcap'].dropna())))[scalemarketcap_number - 1:]
        # sort stocks
        domestic_columns = [cname for cname in data.category.dropna().unique() if 'Domestic Common Stock' in cname]
        data = data[(data.isdelisted == 'N') &
                    (data.exchange != 'OTC') &
                    (data.category.isin(domestic_columns) == True)]

        for category in capit_category_list:
            tickers = data.ticker[data['scalemarketcap'] == f'{category}']
            tickers_list = tickers_list + list(tickers)
        if not os.path.exists(path):
            os.makedirs(path)
        tickers_list = sorted(list(set(tickers_list)))
    else:
        data = data[(data.isdelisted == 'N') & (data.exchange != 'OTC') & (data.category == 'ETF')]
        tickers_list = data['ticker'].to_list()
        tickers_list.remove('PRN')

    tickers_list = [ticker.replace('.', '-') for ticker in tickers_list]
    return tickers_list
