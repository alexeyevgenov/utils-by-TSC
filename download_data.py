from ib_insync import *
import pandas as pd
import datetime


def download_historical_data(contract, days=4100):
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=20)

    last_date = datetime.datetime.today()
    df_all = pd.DataFrame()
    date = last_date - datetime.timedelta(days=days)
    date = date + datetime.timedelta((5 - date.weekday()) % 7)
    while date < last_date:
        end_date_time = date.strftime("%Y%m%d 00:00:00")
        bars = ib.reqHistoricalData(
            contract, endDateTime=end_date_time, durationStr='1 W', keepUpToDate=False,
            barSizeSetting='1 day', whatToShow='MIDPOINT', useRTH=True)

        # convert to pandas dataframe:
        df = util.df(bars)
        df_all = pd.concat([df_all, df], axis=0, sort=False)

        print(f"Loaded {df.shape[0]} bars before {end_date_time}")

        date = date + datetime.timedelta(days=7)

    df_all = df_all.drop(columns=['average', 'barCount'])
    df_all.to_csv(f'../../data/{currency.lower()}.csv', index=False)


if __name__ == '__main__':
    currency = 'EURUSD'  # 'EURUSD'   'GBPUSD'   'USDJPY'
    download_historical_data(Forex(currency))