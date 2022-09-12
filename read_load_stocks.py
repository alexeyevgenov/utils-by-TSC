import yfinance as yf
from finam import Exporter, Market
import os
import datetime
import time
from dateutil.relativedelta import relativedelta
import pandas as pd
from json.decoder import JSONDecodeError

def date_in_columns(data):
    return 'date' in map(str.lower, data.columns)


def ohlc_format(df):
    df.columns = [(el.replace('<', '').replace('>', '')).capitalize() for el in df.columns]
    df['Date'] = ([str(el) for el in df['Date']] + df['Time']).apply(lambda x: pd.to_datetime(x,
                                                                                              format='%Y%m%d%H:%M:%S'))
    df['Volume'] = df['Vol']
    df = df.set_index('Date')
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def finam_historical_data(period, symbol, start):
    if start is None:
        timedelta = relativedelta(years=10)
        period_interval, abbrev = ['month', 'mo'] if 'mo' in period else (['day', 'd'] if 'd' in period else (
            ['year', 'y'] if 'y' in period else 'incorrect'))

        num = int(period.replace(abbrev, ''))

        if period_interval == 'day':
            timedelta = datetime.timedelta(days=num)
        elif period_interval == 'month':
            timedelta = relativedelta(months=num)
        elif period_interval == 'year':
            timedelta = relativedelta(years=num)
        else:
            print(f'There is a problem with value of period of downloading historical data: {period}')

        previous_date = datetime.datetime.today().date() - datetime.timedelta(days=1)
        start_date = previous_date - timedelta
    else:
        start_date = start
        previous_date = None
    print(f'Downloading bars...')
    if symbol not in ['RTSI', 'IMOEX']:
        try:
            ohlc = Exporter().download(id_=Exporter().lookup(code=symbol, market=Market.SHARES).index[0],
                                       market=Market.SHARES,
                                       start_date=start_date,
                                       end_date=previous_date,
                                       delay=0)
        except Exception:
            return pd.DataFrame()
    else:
        try:
            ohlc = Exporter().download(id_=Exporter().lookup(code=symbol, market=Market.INDEXES).index[0],
                                       market=Market.INDEXES,
                                       start_date=start_date,
                                       end_date=previous_date,
                                       delay=0)
        except Exception:
            return pd.DataFrame()
    return ohlc_format(ohlc)


def read_data_old(tickers, path, period='10y', source='yahoo'):
    dictionary = {}
    for symbol in tickers:
        file_name = f'{path}{symbol}.csv'
        if not os.path.exists(file_name):
            print(symbol)
            try:
                if source == 'yahoo':
                    data = yf.download(tickers=symbol, period=period)
                else:
                    data = finam_historical_data(period, symbol, None)
                data.to_csv(file_name)
            except:
                pass
        if os.path.exists(file_name):
            date = os.path.getmtime(file_name)
            if datetime.datetime.fromtimestamp(date).strftime('%Y-%m-%d') == datetime.datetime.now().strftime(
                    '%Y-%m-%d'):
                d = pd.read_csv(file_name)
            else:
                print(f'{symbol}')
                if not pd.read_csv(file_name).empty:
                    old_data_last_day = pd.to_datetime(pd.read_csv(file_name)['Date'].iloc[-1])
                else:
                    print(f'No data for {symbol}')
                    tickers.remove(symbol)
                    continue
                if source == 'yahoo':
                    # d = yf.download(tickers=symbol, period=period)
                    d = yf.download(tickers=symbol, start=old_data_last_day)
                else:
                    d = finam_historical_data(None, symbol, old_data_last_day.date())
                if d.empty:
                    tickers.remove(symbol)
                    return tickers
                d = d.reset_index()
                last_date = d['Date'].iloc[-1]
                if (datetime.datetime.today() - last_date).days < 2:
                    d['Date'] = d['Date'].dt.date
                    d = d[d['Date'] > old_data_last_day]
                    d.to_csv(file_name, mode='a', index=False, header=False)
                else:
                    print(f'No ohlc data for {symbol} for {(datetime.datetime.today() - last_date).days} days')
                    tickers.remove(symbol)
            if d.shape[0] > 0:
                d = d.dropna()
                d.columns = d.columns.str.lower()
                dictionary[symbol] = d
    return tickers


def read_data(tickers: list, path: str, period='10y', source='yahoo'):
    for symbol in tickers:
        # print(symbol)
        file_name = f'{path}{symbol}.csv'

        # download data if there is no it in storage
        if not os.path.exists(file_name):
            print(symbol)
            try:
                if source == 'yahoo':
                    data = yf.download(tickers=symbol, period=period)
                else:
                    data = finam_historical_data(period, symbol, None)
            except Exception as ex:
                print(f'It is failed to load {symbol}. {ex}')
                continue
        # read data from storage if it exists. Remove empty files
        else:
            data = pd.read_csv(file_name)
            if data.empty:
                os.remove(file_name)
                tickers.remove(symbol)
                continue

            # if file exists, not empty and uploaded today - continue
            date = os.path.getmtime(file_name)
            if datetime.datetime.fromtimestamp(date).strftime('%Y-%m-%d') == datetime.datetime.now().strftime(
                    '%Y-%m-%d'):
                continue

            # if file uploaded not today
            data['Date'] = pd.to_datetime(data['Date']).dt.date

            # download data from sources from last date in storage
            old_data_last_day = pd.to_datetime(pd.read_csv(file_name)['Date'].iloc[-1])
            if source == 'yahoo':
                counter = 0
                while True:
                    try:
                        print(f'downloading {symbol} {counter}')
                        new_data = yf.download(tickers=symbol, start=old_data_last_day, threads=False)
                        break
                    except JSONDecodeError as ex:
                        counter += 1
                        if counter > 3:
                            raise ex
                        else:
                            time.sleep(2)

            else:
                new_data = finam_historical_data(None, symbol, old_data_last_day.date())

            # add missed days to current data
            if not new_data.empty:
                new_data = new_data.reset_index()
                new_data['Date'] = new_data['Date'].dt.date
                data_to_add = new_data[new_data['Date'] > old_data_last_day]
                data = data.append(data_to_add)

            # check if data missed more than 20 days
            last_date = data['Date'].iloc[-1]
            count_of_missed_days = (datetime.datetime.today().date() - last_date).days
            if count_of_missed_days > 20:
                print(f'No ohlc data for {symbol} for {count_of_missed_days} days')
                tickers.remove(symbol)
                if os.path.isfile(file_name):
                    os.remove(file_name)
                continue
        if not date_in_columns(data):
            data = data.reset_index()
            if not date_in_columns(data):
                print(f'No Date in columns of DataFrame for {symbol}')
                continue
        data.to_csv(file_name, index=False)

    return tickers
