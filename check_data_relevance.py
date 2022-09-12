import os
import pandas as pd
import datetime
from pickle import dump

from src.lib.search_candles_pattern import market_profile_calculate, train_rsi_patterns, train_pattern_new, \
    train_pattern_old, train_rsi_patterns_old, train_pattern_new2, train_close_pct_patterns
from src.utils.read_load_stocks import read_data


def check_market_profile_old(tickers_list, volume_profile_window):
    print('\nUpdate market profile files')
    # check if dir exists
    if not os.path.isdir('.output/market_profile'):
        os.makedirs('.output/market_profile')

    for ticker in tickers_list:
        market_profile_filename = f'.output/market_profile/market_profile_values_{ticker}_' \
                                  f'{volume_profile_window}_window.csv'
        # check if market profile file exists
        if not os.path.exists(market_profile_filename):
            print(f'Calculation market profile for {ticker}')
            data = pd.read_csv(f'data/stocks/{ticker}.csv')
            data.columns = data.columns.str.lower()
            market_profile_df = market_profile_calculate(data=data, volume_profile_window=volume_profile_window)
            market_profile_df.to_csv(market_profile_filename, index=False)
        # rewrite new data in exist file
        mp_data = pd.read_csv(market_profile_filename)
        if mp_data.shape[0] < 100:
            print(f'No data for {ticker} for market profile calculations')
            continue
        mp_end_date = mp_data.date.iloc[-1]
        ohlc = pd.read_csv(f'data/stocks/{ticker}.csv')
        ohlc.columns = ohlc.columns.str.lower()
        missed_dates = ohlc.date[ohlc.date > mp_end_date]
        if len(missed_dates) != 0:
            print(f'Calculation market profile for {ticker}')
            start_index = min(ohlc[ohlc.date.isin(missed_dates)].index) - volume_profile_window
            market_profile_df = market_profile_calculate(data=ohlc[start_index:].reset_index(),
                                                         volume_profile_window=volume_profile_window)
            market_profile_df.to_csv(market_profile_filename, mode='a', header=False, index=False)
    print('All market profile files created')


def check_market_profile_new(tickers_list, volume_profile_window, data_folder, output_folder, mute):
    if not mute:
        print('\n - Update MARKET_PROFILE base - ')
    # check if dir exists
    if not os.path.isdir(f'{output_folder}market_profile'):
        os.makedirs(f'{output_folder}market_profile')

    for ticker in tickers_list:
        market_profile_filename = f'{output_folder}/market_profile/market_profile_values_{ticker}_' \
                                  f'{volume_profile_window}_window.csv'
        # check if market profile file exists
        if not os.path.exists(market_profile_filename):
            if not mute:
                print(f'Calculation market profile for {ticker}')
            data = pd.read_csv(f'{data_folder}{ticker}.csv')
            data.columns = data.columns.str.lower()
            market_profile_df = market_profile_calculate(data=data, volume_profile_window=volume_profile_window)
            market_profile_df.to_csv(market_profile_filename, index=False)
        # rewrite new data in exist file
        mp_data = pd.read_csv(market_profile_filename)
        if mp_data.shape[0] < 100:
            if not mute:
                print(f'No data for {ticker} for market profile calculations')
                if ticker in tickers_list:
                    tickers_list.remove(ticker)
            continue
        mp_end_date = mp_data.date.iloc[-1]
        ohlc = pd.read_csv(f'{data_folder}{ticker}.csv')
        ohlc.columns = ohlc.columns.str.lower()
        missed_dates = ohlc.date[ohlc.date > mp_end_date]
        if len(missed_dates) != 0:
            if not mute:
                print(f'Calculation market profile for {ticker}')
            start_index = min(ohlc[ohlc.date.isin(missed_dates)].index) - volume_profile_window
            market_profile_df = market_profile_calculate(data=ohlc[start_index:].reset_index(),
                                                         volume_profile_window=volume_profile_window)
            market_profile_df.to_csv(market_profile_filename, mode='a', header=False, index=False)
    if not mute:
        print('Base updated')

    return tickers_list


def check_patterns_old(tickers_list, scan_data, stocks_path, candles_range, interval, volume_profile_window,
                       error_threshold, pct_similarity, rsi_period, today_date, recalculate):
    from src.lib.search_candles_pattern import market_profile_patterns_search_old

    patterns = 0
    print('\nCheck patterns files')

    if not os.path.isdir(f'.output/{scan_data}/patterns/{today_date}'):
        os.makedirs(f'.output/{scan_data}/patterns/{today_date}')

    for ticker in tickers_list:
        for n_candles in candles_range:
            # for price_change_interval in interval:
            patterns_filename = f'.output/{scan_data}/patterns/{today_date}/' \
                                f'patterns_{ticker}_{scan_data}_{n_candles}_candles.pkl'
            targets_filename = f'.output/{scan_data}/patterns/{today_date}/' \
                               f'targets_{ticker}_{scan_data}_{n_candles}_candles.csv'
            market_profile_filename = f'.output/market_profile/market_profile_values_{ticker}_' \
                                      f'{volume_profile_window}_window.csv'
            if not os.path.exists(patterns_filename) or not os.path.exists(targets_filename) or recalculate == 1:
                data = pd.read_csv(f'{stocks_path}{ticker}.csv').dropna()
                data = data.drop_duplicates()
                data.columns = data.columns.str.lower()
                data['date'] = pd.to_datetime(data['date']).dt.date
                data = data.set_index('date')
                if data.shape[0] < 100:
                    print(f'No OHLC data for {ticker}')
                    continue
                prepared_data = data[['open', 'high', 'low', 'close', 'volume']]

                if scan_data == 'ohlc_n_volume':
                    patterns = train_pattern_old(data=prepared_data, n_candles=n_candles, scan_data=scan_data,
                                                 error_threshold=error_threshold, ticker=ticker)
                elif scan_data == 'market_profile':
                    patterns = market_profile_patterns_search_old(data=prepared_data, n_candles=n_candles,
                                                                  ticker=ticker, scan_data=scan_data,
                                                                  error_threshold=error_threshold,
                                                                  market_profile_filename=market_profile_filename)
                elif scan_data == 'ohlc':
                    prepared_data = data[['open', 'high', 'low', 'close']]
                    patterns = train_pattern_old(data=prepared_data, n_candles=n_candles, ticker=ticker,
                                                 error_threshold=error_threshold, scan_data=scan_data)
                elif 'rsi' in scan_data:
                    prepared_data = data[['open', 'high', 'low', 'close']]
                    patterns = train_rsi_patterns_old(data=prepared_data, n_candles=n_candles, ticker=ticker,
                                                      similarity=pct_similarity, scan_data=scan_data,
                                                      rsi_period=rsi_period)
                dump(patterns, open(patterns_filename, 'wb'))

    print('All patterns files found')


def check_patterns(tickers_list, scan_data, stocks_path, candles_range, interval, volume_profile_window,
                   error_threshold, pct_similarity, rsi_period, today_date, recalculate, output_folder, mute):
    from src.lib.search_candles_pattern import market_profile_patterns_search, market_profile_patterns_search_new

    patterns = 0
    if not mute:
        print(f'\n - Check {scan_data.upper()} patterns files - ')

    if not os.path.isdir(f'{output_folder}{scan_data}/patterns/{today_date}'):
        os.makedirs(f'{output_folder}{scan_data}/patterns/{today_date}')

    for ticker in tickers_list:
        for n_candles in candles_range:
            # for price_change_interval in interval:
            patterns_filename = f'{output_folder}{scan_data}/patterns/{today_date}/' \
                                f'patterns_{ticker}_{scan_data}_{n_candles}_candles.pkl'
            targets_filename = f'{output_folder}{scan_data}/patterns/{today_date}/' \
                               f'targets_{ticker}_{scan_data}_{n_candles}_candles.csv'
            market_profile_filename = f'{output_folder}market_profile/market_profile_values_{ticker}_' \
                                      f'{volume_profile_window}_window.csv'
            if not os.path.exists(patterns_filename) or not os.path.exists(targets_filename) or recalculate == 1:
                data = pd.read_csv(f'{stocks_path}{ticker}.csv').dropna()
                data = data.drop_duplicates()
                data.columns = data.columns.str.lower()
                data['date'] = pd.to_datetime(data['date']).dt.date
                data = data.set_index('date')
                if data.shape[0] < 100:
                    if not mute:
                        print(f'No OHLC data for {ticker}')
                        if ticker in tickers_list:
                            tickers_list.remove(ticker)
                    continue

                data_to_today = data.loc[:(pd.to_datetime(today_date) - datetime.timedelta(days=1)).date()]
                prepared_data_with_volume = data_to_today[['open', 'high', 'low', 'close', 'volume']]
                prepared_data_no_volume = data_to_today[['open', 'high', 'low', 'close']]

                if scan_data == 'ohlc_n_volume':
                    patterns = train_pattern_new(data=prepared_data_with_volume, n_candles=n_candles,
                                                 scan_data=scan_data,
                                                 error_threshold=error_threshold, ticker=ticker, today_date=today_date,
                                                 output_folder=output_folder, mute=mute)
                elif scan_data == 'market_profile':
                    patterns = market_profile_patterns_search_new(data=prepared_data_with_volume, n_candles=n_candles,
                                                                  ticker=ticker, scan_data=scan_data,
                                                                  error_threshold=error_threshold,
                                                                  market_profile_filename=market_profile_filename,
                                                                  today_date=today_date, output_folder=output_folder,
                                                                  mute=mute)
                elif scan_data == 'ohlc':
                    patterns = train_pattern_new(data=prepared_data_no_volume, n_candles=n_candles, ticker=ticker,
                                                 error_threshold=error_threshold, scan_data=scan_data,
                                                 today_date=today_date, output_folder=output_folder, mute=mute)
                elif 'rsi' in scan_data:
                    patterns = train_rsi_patterns(data=prepared_data_no_volume, n_candles=n_candles, ticker=ticker,
                                                  similarity=pct_similarity, scan_data=scan_data,
                                                  rsi_period=rsi_period, today_date=today_date,
                                                  output_folder=output_folder, mute=mute)
                dump(patterns, open(patterns_filename, 'wb'))

    if not mute:
        print('All patterns files found')

    return tickers_list


def check_patterns_new(ticker, scan_data, n_candles, stocks_path, error_threshold, pct_similarity, rsi_period,
                       last_pattern_date, output_folder, mute, pattern_save_path, grid_height,
                       bottom_grid_level, upper_grid_level):
    from src.lib.search_candles_pattern import market_profile_patterns_search, market_profile_patterns_search_new

    data = pd.read_csv(f'{stocks_path}{ticker}.csv').dropna()
    data = data.drop_duplicates()
    data.columns = data.columns.str.lower()
    data['date'] = pd.to_datetime(data['date']).dt.date
    data = data.set_index('date')
    if data.shape[0] < 100:
        if not mute:
            print(f'No OHLC data for {ticker}')
        return

    data_to_today = data.loc[:last_pattern_date]
    prepared_data_with_volume = data_to_today[['open', 'high', 'low', 'close', 'volume']]
    prepared_data_no_volume = data_to_today[['open', 'high', 'low', 'close']]

    if scan_data == 'ohlc_n_volume':
        patterns = train_pattern_new(data=prepared_data_with_volume, n_candles=n_candles,
                                     scan_data=scan_data,
                                     error_threshold=error_threshold, ticker=ticker, today_date=last_pattern_date,
                                     output_folder=output_folder, mute=mute)
    elif scan_data == 'market_profile':
        patterns = market_profile_patterns_search_new(data=prepared_data_with_volume, n_candles=n_candles,
                                                      ticker=ticker, scan_data=scan_data,
                                                      error_threshold=error_threshold,
                                                      # market_profile_filename=market_profile_filename,
                                                      today_date=last_pattern_date, output_folder=output_folder,
                                                      mute=mute)
    elif scan_data == 'ohlc':
        train_pattern_new2(data=prepared_data_no_volume, n_candles=n_candles, ticker=ticker,
                           error_threshold=error_threshold, scan_data=scan_data,
                           last_pattern_date=last_pattern_date, mute=mute,
                           save_path=pattern_save_path)
    elif 'rsi' in scan_data:
        patterns = train_rsi_patterns(data=prepared_data_no_volume, n_candles=n_candles, ticker=ticker,
                                      similarity=pct_similarity, scan_data=scan_data,
                                      rsi_period=rsi_period, today_date=last_pattern_date,
                                      output_folder=output_folder, mute=mute)
    elif scan_data == 'close_pct':
        train_close_pct_patterns(scan_data=scan_data, data=prepared_data_no_volume, n_candles=n_candles, ticker=ticker,
                                 similarity_threshold=pct_similarity, last_pattern_date=last_pattern_date, mute=mute,
                                 save_path=pattern_save_path, height=grid_height, bottom_grid_level=bottom_grid_level,
                                 upper_grid_level=upper_grid_level)

    # if not mute:
    #     print('All patterns files found')

    # return tickers_list


def check_market_profile(volume_profile_window, tickers_list, recalculate, output_folder):
    if not os.path.isdir(f'{output_folder}market_profile'):
        os.makedirs(f'{output_folder}market_profile')
    for ticker in tickers_list:
        ohlc_data_filename = f'data/stocks/{ticker}.csv'
        market_profile_filename = f'{output_folder}market_profile/market_profile_values_{ticker}_{volume_profile_window}.csv'
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
                '%Y-%m-%d') != datetime.datetime.now().strftime('%Y-%m-%d') or recalculate == 1:
            # print(f'Market profile calculations')
            market_profile_calculate(data=data, volume_profile_window=volume_profile_window)
            print(f'Market profile saved for {ticker} in {market_profile_filename}')
