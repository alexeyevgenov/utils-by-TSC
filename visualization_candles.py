import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


# Function for candle plot creation based on Plotly Candlestick Plot
def func_candle_plot(frame):
    fig = go.Figure(data=[go.Candlestick(open=frame['open1'],
                                         high=frame['high1'],
                                         low=frame['low1'],
                                         close=frame['close1'])])
    fig.layout.xaxis.type = 'category'
    fig.layout.xaxis.visible = False
    fig.layout.update(margin=dict(b=0))

    return fig.show()


def print_candles(candles):
    # Visualization
    # Alighn candles along horizontal axis
    candles['open1'] = candles['open'] - candles['open']
    candles['close1'] = candles['close'] - candles['open']
    candles['high1'] = candles['high'] - candles['open']
    candles['low1'] = candles['low'] - candles['open']
    # Plot candles
    plt.show()
    #print(f'\n\n\nCluster {i} with {cluster.shape[0]} candles. Action {candles[i][0]}')
    func_candle_plot(candles)


def print_cluster(dict_differ, cluster):
    up = dict_differ[cluster][1]
    stay = dict_differ[cluster][4] - dict_differ[cluster][1] - dict_differ[cluster][2]
    down = dict_differ[cluster][2]

    index = ['-1', '0', '1']

    bar_df = pd.DataFrame([down,
                           stay,
                           up],
                          index=index)

    plt.rcParams["figure.figsize"] = (10, 7)
    bar_df.plot.bar(rot=0)   #, color=(0.2, 0.4, 0.6))

    print('action', dict_differ[cluster][0])
    print_candles(dict_differ[cluster][3])


def candlestick(df, title):
    fig = go.Figure(data=[go.Candlestick(x=df['date'],
                                         open=df['open'],
                                         high=df['high'],
                                         low=df['low'],
                                         close=df['close'])])
    fig.layout.xaxis.type = 'category'
    fig.update_layout(xaxis_rangeslider_visible=False,
                      title=f'{title}')

    # fig.layout.xaxis.visible = False

    # hide scroll along x label
    fig.layout.update(margin=dict(b=0))

    return fig.show()


def candlestick_with_flats(df, title, flat_coordinates, window):
    fig = go.Figure(data=[go.Candlestick(x=df['date'],
                                         open=df['open'],
                                         high=df['high'],
                                         low=df['low'],
                                         close=df['close'])])
    fig.layout.xaxis.type = 'category'
    fig.update_layout(xaxis_rangeslider_visible=False,
                      title=f'{title}')

    while len(flat_coordinates) != 0:
        end = flat_coordinates.pop(0)
        df_piece = df[end - window: end]

        fig.add_shape(type="line",
                      x0=end - window, y0=df_piece['close'].mean(), x1=end, y1=df_piece['close'].mean(),
                      line=dict(color="Red"))

    # fig.layout.xaxis.visible = False

    # hide scroll along x label
    fig.layout.update(margin=dict(b=0))

    return fig.show()


def candlestick_with_rectangle(df, title, n_candles):
    candles = df[:n_candles]
    fig = go.Figure(data=[go.Candlestick(x=df['date'],
                                         open=df['open'],
                                         high=df['high'],
                                         low=df['low'],
                                         close=df['close'])])
    fig.layout.xaxis.type = 'category'
    fig.update_layout(xaxis_rangeslider_visible=False,
                      title=f'{title}')
    fig.add_shape(type="rect",
                  x0=-0.3, y0=candles.low.min(), x1=n_candles-0.5, y1=candles.high.max(),
                  line=dict(color="RoyalBlue"))

    # fig.layout.xaxis.visible = False

    # hide scroll along x label
    fig.layout.update(margin=dict(b=0))

    return fig.show()


def market_profile_visualization(data, title):
    n_candles = len(data)

    fig = go.Figure()

    fig.update_layout(title=f'{title}')

    text_list = []
    for i in range(n_candles):
        text_list.append(data.date.iloc[i].strftime('%Y-%m-%d'))

    # Set axes ranges
    higher_label = data.iloc[:, 1:].to_numpy().max() + 0.05 * data.iloc[:, 1:].to_numpy().max()
    lower_label = data.iloc[:, 1:].to_numpy().min() - 0.05 * data.iloc[:, 1:].to_numpy().min()

    fig.update_xaxes(range=[data.index[0] - 1, data.index[-1] + 1])
    fig.update_yaxes(range=[lower_label, higher_label])

    # Create scatter trace of text labels
    y_text = data.val.min()
    fig.add_trace(go.Scatter(
        x=list(data.index), y=[y_text-y_text*0.03]*n_candles,
        text=text_list,
        mode="text",
    ))

    # Add shapes
    for ind, values in data.iterrows():
        # VAH
        fig.add_shape(name="VAH", type="line",
                      x0=ind - 0.25, y0=values.vah, x1=ind + 0.25, y1=values.vah,
                      line=dict(color="blue", width=3))
        # POC
        fig.add_shape(name="POC", type="line",
                      x0=ind - 0.25, y0=values.poc, x1=ind + 0.25, y1=values.poc,
                      line=dict(color="lightgreen", width=3))
        # VAL
        fig.add_shape(name="VAL", type="line",
                      x0=ind - 0.25, y0=values.val, x1=ind + 0.25, y1=values.val,
                      line=dict(color="red", width=3))
        fig.update_layout(title=f'{title}')

    fig.update_shapes(dict(xref='x', yref='y'))
    return fig.show()


def market_profile_visualization_with_rectangle(data, title, n_candles):
    candles = data[:n_candles]

    text_list = []
    for i in range(data.shape[0]):
        text_list.append(data.date.iloc[i].strftime('%Y-%m-%d'))

    fig = go.Figure()

    fig.update_layout(title=f'{title}')

    # Set axes ranges
    start_date = data.date.iloc[0]
    end_date = data.date.iloc[-1]
    higher_label = data.iloc[:, 1:].to_numpy().max() + 0.05 * data.iloc[:, 1:].to_numpy().max()
    lower_label = data.iloc[:, 1:].to_numpy().min() - 0.05 * data.iloc[:, 1:].to_numpy().min()

    fig.update_xaxes(range=[data.index[0] - 1, data.index[-1] + 1])
    fig.update_yaxes(range=[lower_label, higher_label])

    # Create scatter trace of text labels
    y_text = data.val.min()
    fig.add_trace(go.Scatter(
        x=list(data.index), y=[y_text - y_text * 0.04] * len(data),
        text=text_list,
        mode="text",
    ))

    # Add shapes
    for ind, values in data.iterrows():
        # VAH
        fig.add_shape(name="VAH", type="line",
                      x0=ind - 0.25, y0=values.vah, x1=ind + 0.25, y1=values.vah,
                      line=dict(color="blue", width=3))

        # POC
        fig.add_shape(name="POC", type="line",
                      x0=ind - 0.25, y0=values.poc, x1=ind + 0.25, y1=values.poc,
                      line=dict(color="lightgreen", width=3))
        # VAL
        fig.add_shape(name="VAL", type="line",
                      x0=ind - 0.25, y0=values.val, x1=ind + 0.25, y1=values.val,
                      line=dict(color="red", width=3))

    high_label = candles.vah.max()
    low_label = candles.val.min()
    fig.add_shape(type="rect",
                  x0=data.index[0] - 0.5, y0=low_label - low_label * 0.02, x1=data.index[n_candles] - 0.5,
                  y1=high_label + low_label * 0.02,
                  line=dict(color="RoyalBlue"))

    fig.update_shapes(dict(xref='x', yref='y'))
    return fig.show()

