import pandas as pd
import numpy as np
import os
import pickle
from src.utils.pattern_class import Patterns
from src.utils.pattern_class import PriceChange, PatternType


class SearchPatterns:
    def __init__(self, data, pattern_type, n_candles, symbol, price_change_range, price_change_type,
                 similarity_threshold, mute, save_path, grid_height=None, bottom_grid_level=None,
                 upper_grid_level=None):
        self.ohlc_data = data
        self.pattern_type = pattern_type
        self.n_candles = n_candles
        self.symbol = symbol
        self.price_change_range = price_change_range
        self.price_change_type = price_change_type
        self.similarity_threshold = similarity_threshold
        self.mute = mute
        self.save_path = save_path
        self.grid_height = grid_height
        self.bottom_grid_level = bottom_grid_level
        self.upper_grid_level = upper_grid_level
        self.price_change_interval_max = max(price_change_range)
        self.ohlc_calc_df = None
        self.prepared_signal_name = 'close_pct_change'
        self.pattern_class = Patterns(height=self.grid_height, width=self.n_candles,
                                      similarity=self.similarity_threshold,
                                      signal=self.prepared_signal_name)
        self.price_changes = None
        self.list_of_closes = None
        self.patterns_df = pd.DataFrame()

    def start_search_patterns(self):
        if os.path.exists(self.save_path):
            self.ohlc_data, self.patterns_df = pickle.load(open(self.save_path, 'rb'))
            return
        self.prepare_data()
        self.get_patterns()
        self.get_price_changes()
        self.get_similar_patterns()
        #
        # if not self.mute:
        #     ohlc_data_filtered = self.ohlc_data[self.ohlc_data['mse/similarity'] >= self.similarity_threshold]
        #     print(
        #         f"{ohlc_data_filtered.shape[0]} {self.pattern_type} patterns found in {self.symbol} for
        #         {self.n_candles} candles")

    def prepare_data(self):
        # Take n_candles pattern
        if self.pattern_type == PatternType.CLOSE_PCT:
            self.ohlc_data['previous_close'] = self.ohlc_data['close'].shift(1)
            self.ohlc_data[self.prepared_signal_name] = np.log(
                self.ohlc_data['close'] / self.ohlc_data['previous_close'])

    def get_patterns(self):
        # Get codes of patterns
        self.ohlc_data = self.pattern_class.prepare_close_pct_patterns_new(self.ohlc_data, self.bottom_grid_level,
                                                                           self.upper_grid_level)

    def get_price_changes(self):
        ohlc_calc_df = self.ohlc_data.copy()
        # add price changes to df
        list_of_values = []
        ohlc_calc_df['close'].rolling(self.price_change_interval_max).apply(
            lambda x: list_of_values.append(x.values) or 0,
            raw=False)
        ohlc_calc_df.loc[:len(list_of_values) - 1, 'list_of_closes'] = pd.Series(
            list_of_values).values  # todo: check why "-1"
        ohlc_calc_df['price_change_pct'] = ohlc_calc_df.apply(
            lambda x: (x['list_of_closes'] - x['open']) / x['open'], axis=1)

        # shift aligns last pattern candle and price change
        self.price_changes = ohlc_calc_df['price_change_pct'].shift(-1)
        self.list_of_closes = ohlc_calc_df['list_of_closes'].shift(-1)

        self.ohlc_data = self.ohlc_data.join(self.price_changes).join(self.list_of_closes)
        self.ohlc_data['n_candles'] = self.n_candles
        self.ohlc_data['pattern_type'] = self.pattern_type
        self.ohlc_data['ticker'] = self.symbol

        self.ohlc_data = self.ohlc_data.dropna().reset_index(drop=True)

    def get_similar_patterns(self):
        # self.ohlc_data = self.ohlc_data[:100]
        indexes_list = list(self.ohlc_data.index)
        while len(indexes_list) != 0:
            idx = indexes_list.pop(0)
            print(f'{idx} from {len(indexes_list)}')
            prototype = self.ohlc_data.loc[idx]
            patterns_list = [[prototype['str_codes'], prototype['codes'],
                              prototype['str_codes'], prototype['date'], 100, True]]
            weights = self.pattern_class.weights_matrix(prototype['codes'])
            jdx_list = indexes_list.copy()

            while len(jdx_list) != 0:
                jdx = jdx_list.pop(0)
                candidate = self.ohlc_data.loc[jdx]
                similarity = self.pattern_class.get_similarity(weights, candidate['codes'])
                if similarity >= self.similarity_threshold:
                    patterns_list.append(
                        [prototype['str_codes'], candidate['codes'],
                         candidate['str_codes'], candidate['date'], similarity, False])
                    indexes_list.remove(jdx)
            self.patterns_df = self.patterns_df.append(pd.concat([pd.DataFrame([el]) for el in patterns_list],
                                                                 ignore_index=True), ignore_index=True)
        self.patterns_df.columns = ['prototype', 'pattern', 'pattern_str', 'date', 'similarity', 'is_prototype']

        pickle.dump((self.ohlc_data, self.patterns_df), open(self.save_path, 'wb'))
