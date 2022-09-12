import numpy as np
import pandas as pd
import datetime
from collections import namedtuple
from pickle import dump, load
import os


class GridParams:
    close_pct_grid_height = 11
    bottom_grid_level = -0.05
    upper_grid_level = 0.05


class PriceChange:
    CLOSE_OPEN = 'close_open'
    HIGH_LOW = 'high_low'


class PatternType:
    CLOSE_PCT = 'close_pct'
    valid_types = [CLOSE_PCT]


class Patterns:
    def __init__(self, height, width, similarity, signal):
        self.signal = signal
        self.width, self.height = width, height
        self.similarity = similarity

    def get_code(self, data_piece):
        lowest_close = min(data_piece)
        highest_close = max(data_piece)
        pic = np.array(round(
            ((self.height - 1) * (data_piece - lowest_close) / (highest_close - lowest_close + 1e-10))).astype('int'))
        return pic

    def get_code_close_pct(self, data_piece, bottom_limit, upper_limit):
        lowest_close = bottom_limit
        highest_close = upper_limit
        clipped_data = data_piece.clip(lowest_close, highest_close)
        high_low = highest_close - lowest_close
        if high_low == 0:
            pic = np.zeros((1, self.width))
        else:
            pic = np.array((round(
                ((self.height - 1) * (clipped_data - lowest_close) / (highest_close - lowest_close)))).astype('int'))
        return pic

    def weights_matrix(self, pic):
        weights = np.zeros((self.height, self.width))
        d_value = 2 * self.height / ((self.height - pic) * (self.height - pic + 1) + (pic - 1) * pic)
        for c in range(self.width):
            for j in range(self.height):
                try:
                    weights[j, c] = 1 - abs(pic[c] - j) * d_value[c]
                except Exception as ex:
                    print(ex)
        return weights

    def get_similarity(self, prototype_weights_matrix, candidate_pic):
        sum = 0
        for i, j in zip(candidate_pic, range(self.width)):
            sum += (prototype_weights_matrix[i, j])
        similarity = 100 * sum / self.width
        return similarity

    def prepare_patterns(self, data):
        list_of_values = []
        data[self.signal].rolling(self.width).apply(lambda x: list_of_values.append(x.values) or 0, raw=False)
        data.loc[(self.width - 1):, 'set_of_prices'] = pd.Series(list_of_values).values
        data.loc[(self.width - 1):, 'codes'] = data['set_of_prices'].dropna().apply(
            lambda x: self.get_code(pd.Series(x)))
        for i in range(1, 6):
            data.loc[1:, f'price_change_{i}'] = ((data['close'].shift(-i+1) - data['open'])/data['open']).shift(-1)
        return data

    def prepare_close_pct_patterns(self, data, bottom_grid_level, upper_grid_level):
        list_of_values = []
        data[self.signal].rolling(self.width).apply(lambda x: list_of_values.append(x.values) or 0, raw=False)
        data.loc[(self.width - 1):, 'list_of_close_pct'] = pd.Series(list_of_values).values
        data.loc[(self.width - 1):, 'codes'] = data['list_of_close_pct'].dropna().apply(
            lambda x: self.get_code_close_pct(pd.Series(x), bottom_grid_level, upper_grid_level))
        data['str_codes'] = data['codes'].dropna().apply(lambda x: ''.join(str(el)+'-' for el in x)[:-1])
        return data

    def prepare_close_pct_patterns_new(self, data, bottom_grid_level, upper_grid_level):
        list_of_values = []
        data[self.signal].rolling(self.width).apply(lambda x: list_of_values.append(x.values) or 0, raw=False)
        data.loc[self.width:, 'list_of_close_pct'] = pd.Series(list_of_values).values
        data.loc[self.width:, 'codes'] = data['list_of_close_pct'].dropna().apply(
            lambda x: self.get_code_close_pct(pd.Series(x), bottom_grid_level, upper_grid_level))
        data['str_codes'] = data['codes'].dropna().apply(lambda x: ''.join(str(el)+'-' for el in x)[:-1])
        return data
