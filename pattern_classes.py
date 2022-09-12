from typing import NamedTuple


class Pattern(NamedTuple):
    index: str
    ohlc: list
    target: float
    next_candle_ohlc: list
    samples: list


class Sample(NamedTuple):
    index: str
    ohlc: list
    mse: float
    # target_close_open: float
    # target_high_open: float
    # target_low_open: float


class All(NamedTuple):
    patterns: list


class ChartParamsNew(NamedTuple):
    patterns_count: int
    mean: float
    std: float
    positive_count: int
    positive_mean: float
    negative_count: int
    negative_mean: float
    nulls_count: int
    pos_neg_ratio: list
    price_change: list


class ChartParams(NamedTuple):
    patterns_count: int
    mean: float
    std: float
    positive_count: int
    positive_mean: float
    negative_count: int
    negative_mean: float
    pos_neg_ratio: list
    action: str
    change: str
    price_change: list
