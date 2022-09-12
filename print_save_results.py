import pandas as pd
import numpy as np
import os
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from src.utils.tickers_from_quandl_tickers_csv import get_tickers_list
from src.utils.read_load_stocks import read_data

bootstrap = Bootstrap()


def results_to_csv(result_list, result_folder, result_prefix):
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    df_result = pd.DataFrame(result_list, columns=['n_candles', 'ticker', 'patterns_count',
                                                   'close_open_mean', 'close_open_std',
                                                   'close_open_positive_count', 'close_open_positive_mean',
                                                   'close_open_negative_count', 'close_open_negative_mean',
                                                   'price_change_interval'])
    # save results with previous result files storing
    files_list = [filename for filename in os.listdir(f'{result_folder}') if filename.startswith(f"{result_prefix}")]
    if len(files_list) != 0:
        number = max([int(p[2]) for p in [x.split('_') for x in files_list]])
    else:
        number = 0

    df_result.to_csv(f'{result_folder}{result_prefix}{number + 1}_.csv', index=False)


def up_or_down(a, b):
    if np.mean(a) > np.mean(b):
        predict = 'up'
    elif np.mean(a) < np.mean(b):
        predict = 'down'
    else:
        predict = 'nowhere'
    return predict


def get_direction(x):  # todo: check it
    return 'up' if ((x.positive_pct > x.negative_pct) &
                    (x.mean_target > 0)) else (
        'down' if ((x.positive_pct < x.negative_pct) & (x.mean_target < 0)) else (
            'stand' if x.mean_target == 0 else 'incorr'))


def print_result_v3(result_all, interval_range, csv_path, patterns_details):
    try:
        result_all = result_all[result_all['price_change_interval'].isin(interval_range)]
        if result_all.empty:
            print('No symbols for date.')
            return
        tickers_list = np.unique(result_all['ticker'])
        # print('{:<10}'.format('Ticker'), '{:<16}'.format('Patterns count'), '{:^35}'.format('Interval 3'),
        #       '{:^35}'.format('Interval 4'), '{:^35}'.format('Interval 5'))
        # note = 'dominance(%) / price change(%)'
        # print(' ' * 10, ' ' * 16, '{:^35}'.format(note), '{:^35}'.format(note), '{:^35}'.format(note))
        # print('\n')
        save_list = []
        for ticker in tickers_list:
            patterns_count = result_all[result_all['ticker'] == ticker]['patterns_count'].iloc[0]
            to_print = '{:<10}'.format(ticker) + '{:^16}'.format(patterns_count)
            save_row = [ticker, patterns_count]
            for interval in interval_range:
                filtered_df = result_all[(result_all['ticker'] == ticker) &
                                         (result_all['price_change_interval'] == interval)]
                if not filtered_df.empty:
                    try:
                        direction = 'rise' if filtered_df['action'].item() == 'buy' else 'fall' if \
                            filtered_df['action'].item() == 'sell' else 'stay'
                        dominance = filtered_df['dominance'].item()
                        pct_change = round(filtered_df['pct_change'].item() * 100, 2)
                        to_print += '{:^35}'.format(str(dominance) + '%' + ' / ' + str(pct_change) + '%' + ' ('
                                                    + direction + ')')
                        # save_row.append(str(dominance) + '%' + ' / ' + str(pct_change) + '%' + ' (' + str(direction) + ')')
                        save_row = save_row + [dominance, pct_change]
                    except Exception as ex:
                        print(ex)
                else:
                    save_row = save_row + ['-', '-']
                    to_print += ' ' * 35
            save_list.append(save_row)
        adding_columns = ['dominance', 'price_change']
        created_columns = [col_name + '_' + str(el) for el, col_name in zip(sorted(list(range(3, 6)) * 2),
                                                                            adding_columns * 3)]
        saved_df = pd.DataFrame(save_list, columns=['ticker', 'patterns_count'] + created_columns)
        saved_df.to_csv(csv_path + '/flask_table.csv', index=False)
        # print(to_print)
        print_html(csv_path + '/flask_table.csv', patterns_details)
    except Exception as ex:
        print(f'{ex} error by result printing')


def print_html(path, patterns_details):
    app = Flask(__name__, template_folder='../../flask_print/app/templates')

    bootstrap.init_app(app)

    @app.route('/')
    def html_table():
        df = pd.read_csv(path)
        return render_template(template_name_or_list='simple.html', table_data=df,
                               titles=df.columns.values)

    @app.route('/details', methods=['GET', 'POST'])
    def details():
        ticker, column = request.args.get('key').replace(' ', '').split(',')
        if column == 'patterns_count':
            df_to_output = patterns_details[patterns_details['ticker'] == ticker][
                ['date', 'interval_3', 'interval_4', 'interval_5']]
            return render_template(template_name_or_list='simple.html', table_data=df_to_output,
                                   titles=df_to_output.columns.values)
        return ''
    app.run(debug=True)


def print_result_v2_old(result_all, tickers_list, interval):
    # check if interval is setted
    if type(interval) is int:
        interval = [interval]
    else:
        interval = range(1, 6)

    df_result = pd.DataFrame(result_all, columns=['ticker', 'n_candles', 'price_change_interval', 'patterns_count',
                                                  'close_open_mean', 'close_open_std',
                                                  'close_open_positive_count', 'close_open_positive_mean',
                                                  'close_open_negative_count', 'close_open_negative_mean',
                                                  'scan_data'])

    # ----------------------------------------------------------------------------------------
    # Output 1
    print('\n\nCHART 1. Price change sorted by scan data and price change\n')
    print('{:<10}'.format('Ticker'), '{:^21}'.format('Scan method'),
          '{:^19}'.format('Interval 1'), '{:^19}'.format('Interval 2'), '{:^19}'.format('Interval 3'),
          '{:^19}'.format('Interval 4'), '{:^19}'.format('Interval 5'))

    df_sorted1 = pd.DataFrame()
    for ticker in tickers_list:
        for scan_data in ['ohlc', 'ohlc_n_volume', 'market_profile', 'rsi', 'rsi_14']:
            for price_change_interval in interval:
                sorted_data = df_result[(df_result.ticker == ticker) &
                                        (df_result.scan_data == scan_data) &
                                        (df_result.price_change_interval == price_change_interval) &
                                        (df_result.patterns_count >= 5)]

                patterns_count = sum(sorted_data.patterns_count)
                mean_target = sorted_data['close_open_mean'].mean()
                positive_target_count = sum(sorted_data.close_open_positive_count)
                negative_target_count = sum(sorted_data.close_open_negative_count)
                pos_neg_ratio = [round(positive_target_count / (positive_target_count + negative_target_count) * 100),
                                 round(negative_target_count / (positive_target_count + negative_target_count) * 100
                                       )] if (positive_target_count + negative_target_count) != 0 else [0, 0]

                df_sorted1 = df_sorted1.append([[ticker, scan_data, price_change_interval, patterns_count,
                                                 pos_neg_ratio[0], pos_neg_ratio[1], mean_target]])

    df_sorted1.columns = ['ticker', 'scan_data', 'price_change_interval', 'patterns_count', 'positive_pct',
                          'negative_pct', 'mean_target']

    df_sorted1 = df_sorted1.reset_index().drop(columns=['index'])
    df_sorted1['direction'] = df_sorted1.apply(get_direction, axis=1)
    df_sorted1['max_pct'] = df_sorted1.apply(lambda x: max(x.positive_pct, x.negative_pct), axis=1)
    df_sorted1 = df_sorted1.drop(df_sorted1[df_sorted1.max_pct < 51].index)
    df_sorted1['target_pct'] = round(df_sorted1.mean_target * 100, 2)
    # drop columns with incoordinated positive/negative percent and the sign of mean value of target
    prepared_df = df_sorted1[df_sorted1.direction != 'incorr']

    df_chart1 = pd.DataFrame()
    for ticker in tickers_list:
        print()
        for scan_data in ['ohlc', 'ohlc_n_volume', 'market_profile', 'rsi', 'rsi_14']:
            to_print = '{:<10}'.format(ticker) + '{:^21}'.format(scan_data)
            for price_change_interval in interval:
                sorted_chart_df = prepared_df[(prepared_df.ticker == ticker) &
                                              (prepared_df.scan_data == scan_data) &
                                              (prepared_df.price_change_interval == price_change_interval)]
                if sorted_chart_df.empty:
                    to_print += '{:^20}'.format('    ---  ')
                    continue

                max_pct = sorted_chart_df.max_pct.item()
                target_pct = sorted_chart_df.target_pct.item()

                to_print = to_print + '     {:>3}'.format(max_pct) + '% / ' + '{:<5.2f}'.format(target_pct) + '%  '
            print(to_print)

    # ----------------------------------------------------------------------------------------
    # Output 2
    print('\n\nCHART 2. Price change sorted by price change interval from 1 to 5 days\n')
    print('{:<10}'.format('Ticker'), '{:^32}'.format('price move for 1 day'),
          '{:^32}'.format('price move for 2 days'), '{:^32}'.format('price move for 3 days'),
          '{:^32}'.format('price move for 4 days'), '{:^32}'.format('price move for 5 days'))

    for ticker in tickers_list:
        print()
        to_print = '{:<10}'.format(ticker)
        for price_change_interval in interval:
            sorted_chart_df = prepared_df[(prepared_df.ticker == ticker) &
                                          (prepared_df.price_change_interval == price_change_interval)]
            # if only one type of scan_data or different directions are predicted
            if sorted_chart_df.empty or (sorted_chart_df.shape[0] == 1) or \
                    (np.unique(sorted_chart_df.direction).size != 1) or (
                    np.unique(sorted_chart_df.scan_data).size <= 1):
                to_print += '{:^32}'.format('     ---  ')
                continue

            max_pct = round(sorted_chart_df.max_pct.mean())
            target_pct = (round(sorted_chart_df.mean_target.mean() * 100, 2))
            direction = sorted_chart_df.direction.any()

            to_print = to_print + '{:>5}'.format(direction) + '     {:>3}'.format(max_pct) + '% / ' + \
                       '{:<5.2f}'.format(target_pct) + '%         '
        print(to_print)


def print_result_v2(result_all, tickers_list, interval, dominance, min_patterns_count, n_candles):
    if type(interval) is int:
        interval = [interval]
    else:
        interval = range(1, 6)

    df_result = pd.DataFrame(result_all, columns=['ticker', 'n_candles', 'price_change_interval', 'patterns_count',
                                                  'close_open_mean', 'close_open_std',
                                                  'close_open_positive_count', 'close_open_positive_mean',
                                                  'close_open_negative_count', 'close_open_negative_mean',
                                                  'close_open_nulls_count', 'scan_data'])

    # ----------------------------------------------------------------------------------------
    # Output 1
    print('\n\nCHART 1. Price change sorted by scan data and price change\n')
    print('{:<10}'.format('Ticker'), '{:^21}'.format('Scan method'),
          '{:^19}'.format('Interval 1'), '{:^19}'.format('Interval 2'), '{:^19}'.format('Interval 3'),
          '{:^19}'.format('Interval 4'), '{:^19}'.format('Interval 5'))

    df_sorted1 = pd.DataFrame()
    for ticker in tickers_list:
        for scan_data in ['ohlc', 'ohlc_n_volume', 'market_profile', 'rsi', 'rsi_14']:
            for price_change_interval in interval:
                sorted_data = df_result[(df_result.ticker == ticker) &
                                        (df_result.scan_data == scan_data) &
                                        (df_result.price_change_interval == price_change_interval) &
                                        (df_result.patterns_count >= min_patterns_count)]
                if sorted_data.empty:
                    continue
                sorted_data['pos_or_neg_dominance'] = sorted_data.apply(
                    lambda x: 1 if (x['close_open_positive_count'] > (
                            x['close_open_negative_count'] + x['close_open_nulls_count']))
                    else -1 if (x['close_open_negative_count'] > (
                            x['close_open_positive_count'] + x['close_open_nulls_count']))
                    else 0, axis=1)

                # if len(sorted_data['pos_or_neg_dominance'].value_counts().to_list()) > 1:
                # print(sorted_data['pos_or_neg_dominance'])
                #     print('count of dominance types more than one "print_save_result.py" line 179')

                dominance_type = (sorted_data['pos_or_neg_dominance'].value_counts() /
                                  sorted_data['pos_or_neg_dominance'].count()).idxmax() if (
                                                                                                   (sorted_data[
                                                                                                        'pos_or_neg_dominance'].value_counts() /
                                                                                                    sorted_data[
                                                                                                        'pos_or_neg_dominance'].count()).max() * 100) > dominance else '-'

                # if dominance_type == '-':
                #     continue

                # if len(np.unique(sorted_data['pos_or_neg_dominance'])) > 1:
                #     print('Here')

                # sorted_data = sorted_data[sorted_data['pos_or_neg_dominance'] == dominance_type]  # todo: ???

                missed_candles = set(n_candles).difference(np.unique(sorted_data['n_candles']))

                mean_target = sorted_data['close_open_positive_mean'].mean() if dominance_type == 1 else (
                    sorted_data['close_open_negative_mean'].mean() if dominance_type == -1 else 0)
                sorted_data['pos_neg_ratio'] = sorted_data.apply(
                    lambda x: [round(x['close_open_positive_count'] / x['patterns_count'] * 100),
                               round(x['close_open_negative_count'] / x['patterns_count'] * 100),
                               round(x['close_open_nulls_count'] / x['patterns_count'] * 100)]
                    if x['patterns_count'] != 0 else [0, 0, 0], axis=1)

                ratio_with_missed_candles = sorted_data['pos_neg_ratio']
                for _ in missed_candles:
                    ratio_with_missed_candles = ratio_with_missed_candles.append(pd.Series([[33.3, 33.3, 33.3]]))
                pos_neg_ratio = np.mean(np.stack(ratio_with_missed_candles.values), axis=0)

                df_sorted1 = df_sorted1.append([[ticker, scan_data, price_change_interval,
                                                 sum(sorted_data['patterns_count']), pos_neg_ratio[0],
                                                 pos_neg_ratio[1], pos_neg_ratio[2], mean_target]])

    df_sorted1.columns = ['ticker', 'scan_data', 'price_change_interval', 'patterns_count', 'positive_pct',
                          'negative_pct', 'null_pct', 'mean_target']

    df_sorted1 = df_sorted1.reset_index().drop(columns=['index'])
    df_sorted1['max_pct'] = df_sorted1.apply(lambda x: max(x.positive_pct, x.negative_pct, x.null_pct), axis=1)
    df_sorted1 = df_sorted1.drop(df_sorted1[df_sorted1.max_pct < dominance].index)
    df_sorted1['max_col_name'] = df_sorted1[['positive_pct', 'negative_pct', 'null_pct']].idxmax(axis=1)
    df_sorted1['direction'] = df_sorted1['max_col_name'].map({'positive_pct': 'up',
                                                              'negative_pct': 'down',
                                                              'null_pct': 'stand'})
    df_sorted1['target_pct'] = round(df_sorted1.mean_target * 100, 2)

    for ticker in tickers_list:
        print()
        for scan_data in ['ohlc', 'ohlc_n_volume', 'market_profile', 'rsi', 'rsi_14']:
            to_print = '{:<10}'.format(ticker) + '{:^21}'.format(scan_data)
            for price_change_interval in interval:
                sorted_chart_df = df_sorted1[(df_sorted1.ticker == ticker) &
                                             (df_sorted1.scan_data == scan_data) &
                                             (df_sorted1.price_change_interval == price_change_interval)]
                if sorted_chart_df.empty:
                    to_print += '{:^20}'.format('    ---  ')
                    continue

                max_pct = sorted_chart_df.max_pct.item()
                target_pct = sorted_chart_df.target_pct.item()

                if target_pct != 0:
                    to_print = to_print + '     {:>3.0f}'.format(max_pct) + '% / ' + '{:<5.2f}'.format(
                        target_pct) + '%  '
            print(to_print)

    # ----------------------------------------------------------------------------------------
    # Output 2
    print('\n\nCHART 2. Price change sorted by price change interval from 1 to 5 days\n')
    print('{:<10}'.format('Ticker'), '{:^32}'.format('price move for 1 day'),
          '{:^32}'.format('price move for 2 days'), '{:^32}'.format('price move for 3 days'),
          '{:^32}'.format('price move for 4 days'), '{:^32}'.format('price move for 5 days'))

    for ticker in tickers_list:
        print()
        to_print = '{:<10}'.format(ticker)
        for price_change_interval in interval:
            sorted_chart_df = df_sorted1[(df_sorted1.ticker == ticker) &
                                         (df_sorted1.price_change_interval == price_change_interval)]
            # if only one type of scan_data or different directions are predicted
            if sorted_chart_df.empty or (sorted_chart_df.shape[0] == 1) or \
                    (np.unique(sorted_chart_df.direction).size != 1) or (
                    np.unique(sorted_chart_df.scan_data).size <= 1):
                to_print += '{:^32}'.format('     ---  ')
                continue

            max_pct = round(sorted_chart_df.max_pct.mean())
            target_pct = (round(sorted_chart_df.mean_target.mean() * 100, 2))
            direction = sorted_chart_df.direction.any()

            if target_pct != 0:
                to_print = to_print + '{:>5}'.format(direction) + '     {:>3}'.format(max_pct) + '% / ' + \
                           '{:<5.2f}'.format(target_pct) + '%         '
            else:
                to_print = to_print + ' {:>5}'.format(direction) + '         {:>3}'.format(max_pct) + '%             '
        print(to_print)

    # -----------------------------------------------------------------------------------------
    # Output 3
    # print('\n\nCHART 3. Recomendations:\n')
    # for ticker in tickers_list:
    #     sorted_chart_df = prepared_df[(prepared_df.ticker == ticker)]
    #     if sorted_chart_df.empty or (sorted_chart_df.shape[0] == 1) or \
    #             (np.unique(sorted_chart_df.direction).size != 1) or (np.unique(sorted_chart_df.scan_data).size <= 1):
    #         print(f'{ticker} - nothing interesting')
    #         continue
    #
    #     max_pct = round(sorted_chart_df.max_pct.mean())
    #     target_pct = (round(sorted_chart_df.mean_target.mean() * 100, 2))
    #     direction = sorted_chart_df.direction.any()
    #
    #     print(f'{ticker} - {direction} {max_pct}%. Average price moves '
    #           f'{direction} {target_pct}%')

    # if direction != 'incorr':
    #     if ((direction == 'up') and (target_mean > 0)) or ((direction == 'down') and (target_mean < 0)) or (
    #             (direction == 'stand') and (target_mean == 0)):
    #         print(f'{ticker} - {direction} {max(positive_pct_mean, negative_pct_mean)}%. Average price moves '
    #               f'{direction} {target_mean}%')
    # else:
    #     print(f'{ticker} - nothing interesting')


def print_result(result_all, stocks_path, scalemarketcap_number, interval):
    # check if interval is setted
    if type(interval) is int:
        interval = [interval]
    else:
        interval = range(1, 6)

    df_result = pd.DataFrame(result_all, columns=['ticker', 'n_candles', 'price_change_interval', 'patterns_count',
                                                  'close_open_mean', 'close_open_std',
                                                  'close_open_positive_count', 'close_open_positive_mean',
                                                  'close_open_negative_count', 'close_open_negative_mean',
                                                  'scan_data'])

    if scalemarketcap_number == 7:
        tickers_list = sorted(['SPY', 'QQQ', 'XLF', 'XLU', 'XLE', 'JETS', 'VNQ', 'IWM', 'VXX', 'TLT'])
        read_data(tickers=tickers_list, path=stocks_path)
    else:
        tickers_list = get_tickers_list(path=stocks_path, scalemarketcap_number=scalemarketcap_number)
        # Download all selected tickers
        read_data(tickers=tickers_list, path=stocks_path)

    print('{:^14}'.format('  \nTicker'), '{:^21}'.format('Scan method'), '{:^21}'.format('Median price change,%'),
          '{:^21}'.format('Mean price change,%'), '{:^16}'.format('Pos Neg ratio'))

    values_dict = {}

    for ticker in tickers_list:
        print('\n')
        pos_list = []
        neg_list = []
        median_list = []
        mean_list = []

        for scan_data in ['ohlc', 'ohlc_n_volume', 'market_profile']:
            selected_ticker = df_result[(df_result.ticker == ticker) &
                                        (df_result.scan_data == scan_data)]['close_open_mean']
            if selected_ticker.empty:
                print('{:^10}'.format(ticker), '{:^21}'.format(scan_data),
                      '{:^21.2f}'.format(np.nan), '{:^21.2f}'.format(np.nan), '{:^16}'.format(f'0% / 0%'))
                continue

            median = selected_ticker.median()
            mean = selected_ticker.mean()
            positive_targets = [el for el in selected_ticker if el > 0]
            positive_count = len(positive_targets)
            positive_mean = np.mean(positive_targets)
            negative_targets = [el for el in selected_ticker if el < 0]
            negative_count = len(negative_targets)
            pos_neg_ratio = [round(positive_count / (positive_count + negative_count) * 100),
                             round(negative_count / (positive_count + negative_count) * 100)]

            pos_list.append(pos_neg_ratio[0])
            neg_list.append(pos_neg_ratio[1])
            median_list.append(median)
            mean_list.append(mean)

            print('{:^10}'.format(ticker), '{:^21}'.format(scan_data),
                  '{:^21.2f}'.format(median * 100),
                  '{:^21.2f}'.format(mean * 100),
                  '{:^16}'.format(f'{pos_neg_ratio[0]}% / {pos_neg_ratio[1]}%'))

            # check results consistency
        sum_percents = sum([0.333 * el for el in max(pos_list, neg_list)])
        if min(max(pos_list, neg_list)) > 50 and sum_percents > 50:
            if pos_list > neg_list:
                predict = 'up'
            else:
                predict = 'down'
            values_dict[f'{ticker}'] = [sum_percents, np.mean(median_list), np.mean(mean_list), predict]

    print('\nPrediction:')
    for ticker in tickers_list:
        if ticker not in values_dict.keys():
            print(f'{ticker} - nothing interesting')
            continue
        weighted_percent = values_dict[f'{ticker}'][0]
        average_median = round(values_dict[f'{ticker}'][1] * 100, 2)
        average_mean = round(values_dict[f'{ticker}'][2] * 100, 2)
        predict = values_dict[f'{ticker}'][3]
        print(f'{ticker} - {predict} {round(weighted_percent)}%. Average price moves {predict} {average_median}%')
