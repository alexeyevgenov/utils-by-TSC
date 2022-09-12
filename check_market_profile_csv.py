import os
import pandas as pd
import datetime
from src.utils.volume_preparation import market_profile_calculate
from src.utils.read_load_stocks import read_data


def check_market_profile(volume_profile_window, tickers_list):
    if not os.path.isdir('.output/market_profile'):
        os.mkdir('.output/market_profile')
    for ticker in tickers_list:
        ohlc_data_filename = f'data/stocks/{ticker}.csv'
        market_profile_filename = f'.output/market_profile/market_profile_values_{ticker}_{volume_profile_window}.csv'
        ohlc_modif_date = os.path.getmtime(ohlc_data_filename)
        if not os.path.exists(ohlc_data_filename) or datetime.datetime.fromtimestamp(ohlc_modif_date).strftime(
                '%Y-%m-%d') != datetime.datetime.now().strftime('%Y-%m-%d'):
            read_data([ticker], f'data/stocks/')
        data = pd.read_csv(ohlc_data_filename)
        data.columns = data.columns.str.lower()
        data['date'] = pd.to_datetime(data['date']).dt.date
        data = data.set_index('date')
        # check if no data
        mp_modif_date = os.path.getmtime(market_profile_filename)
        if not os.path.exists(market_profile_filename) or datetime.datetime.fromtimestamp(mp_modif_date).strftime(
                '%Y-%m-%d') != datetime.datetime.now().strftime('%Y-%m-%d'):
            # print(f'Market profile calculations')
            market_profile_calculate(data, market_profile_filename)
            print(f'Market profile saved for {ticker} in {market_profile_filename}')


def check_market_profile_new(data, market_profile_filename):
    if not os.path.isdir('.output/market_profile'):
        os.mkdir('.output/market_profile')
    if not os.path.exists(market_profile_filename):
        market_profile_calculate(data, market_profile_filename)