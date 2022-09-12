import numpy as np
from market_profile import MarketProfile


def volume_profile_quantile(data, volume_profile_window):
    # using quantiles for value profile definition
    data['volume_category'] = np.nan
    for idx in range(data.shape[0] - volume_profile_window):
        data_piece = data.volume[idx: idx + volume_profile_window]
        q1 = data_piece.quantile(0.25)
        q3 = data_piece.quantile(0.75)
        iqr = q3 - q1
        minimum = q1 - 1.5 * iqr
        maximum = q3 + 1.5 * iqr
        current_index = idx + volume_profile_window
        current_volume = data.volume.iloc[current_index]
        data["volume_category"].iloc[current_index] = 'normal' if q1 <= current_volume <= q3 else \
            ('little' if minimum <= current_volume < q1 else ('big' if q3 < current_volume <= maximum else
                                                              ('outlier_up' if current_volume > maximum
                                                               else 'outlier_down')))
    data = data.dropna()
    return data


def volume_rolling(data):
    # take average value for 10 last days method.
    data['volume_avg'] = data.volume.rolling(11).mean()
    data['volume_change'] = data.volume / data.volume_avg - 1
    data['volume_category'] = data.volume_change.apply(lambda x: 'big' if x >= 1.5 else ('little' if x <= - 1.5
                                                                                         else 'average'))
    return data


def market_profile_lib(data, end_date, volume_profile_window):
    """input data in Market Profile library and get poc, vah, val"""
    end_index = data[data.date == end_date].index.item()
    start_index = end_index - volume_profile_window
    data.columns = data.columns.str.capitalize()
    mp = MarketProfile(data)
    mp_slice = mp[start_index:end_index]
    poc = mp_slice.poc_price
    vah = mp_slice.value_area[1]
    val = mp_slice.value_area[0]
    return poc, vah, val


def market_profile(data):
    """try to create market profile algorithm from trading view"""
    value_area = []
    up_rows = 0
    down_rows = 0
    total_volume = sum(data.volume)
    value_area_limit = total_volume * 0.7
    poc = max(data.volume)
    poc_index = np.argmax(data.volume)
    value_area.append(poc)
    up_indexes = list(data.index[:poc_index])
    up_indexes.reverse()
    down_indexes = list(data.index[poc_index + 1:])
    while ((len(up_indexes) + len(down_indexes)) >= 2) or (sum(value_area) < value_area_limit):
        if len(up_indexes) >= 2:
            up_rows = data[up_indexes[0]:up_indexes[1]]
            del up_indexes[:2]
        if len(down_indexes) >= 2:
            down_rows = data[down_indexes[0]:down_indexes[1]]
            del down_indexes[:2]
        rows_max = max(sum(up_rows.volume), sum(down_rows.volume))
        if sum(up_rows.volume) > sum(down_rows.volume):
            value_area.insert(0, rows_max)
        elif sum(up_rows.volume) < sum(down_rows.volume):
            value_area.append(rows_max)
    vah = value_area[0]
    val = value_area[-1]
    return value_area, poc, vah, val
