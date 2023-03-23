import ccxt
import numpy as np
import datetime
import pandas as pd
import pandas_ta as ta
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import plotly.graph_objects as go
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from dash.dependencies import Input, Output
from sklearn.model_selection import train_test_split
import os


available_currencies = ['BTC/USDT', 'ETH/USDT', 'LTC/USDT', 'XRP/USDT', 'BCH/USDT']

# Funções fetch_ohlcv_data, calculate_indicators, trading_strategy, backtest, buy_and_hold, prepare_data e create_rnn_model

def fetch_ohlcv_data(exchange_id='binance', symbol='BTC/USDT', timeframe='1h', since=None, limit=1000):
    exchange = getattr(ccxt, exchange_id)()
    ohlcv_data = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
    return np.array(ohlcv_data)

def calculate_indicators(ohlcv_data):
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['ema_short'] = ta.ema(df['close'], length=12)
    df['ema_long'] = ta.ema(df['close'], length=26)
    df['rsi'] = ta.rsi(df['close'])
    return df['ema_short'].values, df['ema_long'].values, df['rsi'].values

def trading_strategy(ohlcv_data, ema_short, ema_long, rsi):
    actions = []
    in_position = False

    for i in range(len(ohlcv_data)):
        if ema_short[i] > ema_long[i] and rsi[i] < 30:
            if not in_position:
                actions.append('buy')
                in_position = True
            else:
                actions.append('hold')
        elif ema_short[i] < ema_long[i] and rsi[i] > 70:
            if in_position:
                actions.append('sell')
                in_position = False
            else:
                actions.append('hold')
        else:
            actions.append('hold')

    return actions

def backtest(ohlcv_data, actions, initial_balance=10000):
    balance = initial_balance
    in_position = False
    trades = 0
    successful_trades = 0

    for i in range(len(actions)):
        if actions[i] == 'buy' and not in_position:
            balance -= ohlcv_data[i, 4]
            in_position = True
            trades += 1
        elif actions[i] == 'sell' and in_position:
            balance += ohlcv_data[i, 4]
            in_position = False
            trades += 1
            successful_trades += 1

    if in_position:
        balance += ohlcv_data[-1, 4]

    return balance, trades, successful_trades

def buy_and_hold(ohlcv_data, initial_balance=10000):
    return initial_balance * (ohlcv_data[-1, 4] / ohlcv_data[0, 4])

def prepare_data(ohlcv_data, window_size=60, test_size=0.2):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(ohlcv_data[:, 1:5])

    X = []
    y = []

    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, :])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    return X_train, X_test, y_train, y_test, scaler

def create_rnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50, activation='relu', return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def rnn_trading_strategy(model, ohlcv_data, scaler, weights_path='rnn_model_weights.h5', window_size=60):
    model.load_weights(weights_path)
    scaled_data = scaler.transform(    ohlcv_data[:, 1:5])
    actions = []

    for i in range(window_size, len(scaled_data)):
        X = np.array([scaled_data[i-window_size:i, :]])
        y_pred = model.predict(X)[0, 0]
        y_actual = scaled_data[i, 0]

        if y_pred > y_actual:
            actions.append('buy')
        elif y_pred < y_actual:
            actions.append('sell')
        else:
            actions.append('hold')

    return ['hold'] * window_size + actions

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1('Crypto Trading Dashboard'), className='text-center mb-4')),
    dbc.Row([
        dbc.Col([
            html.H3('Backtest period:'),
            html.P(id='backtest-period', className='lead')
        ], width=4),
        dbc.Col([
            html.H3('Initial balance:'),
            html.P(id='initial-balance', className='lead')
        ], width=4),
        dbc.Col([
            html.H3('Total trades:'),
            html.P(id='total-trades', className='lead')
        ], width=4),
    ], className='mb-4'),
    dbc.Row([
        dbc.Col([
            html.H3('Final balance (Trading strategy):'),
            html.P(id='final-balance-strategy', className='lead')
        ], width=4),
        dbc.Col([
            html.H3('Percentage return (Trading strategy):'),
            html.P(id='percentage-return-strategy', className='lead')
        ], width=4),
        dbc.Col([
            html.H3('Successful trades:'),
            html.P(id='successful-trades', className='lead')
        ], width=4),
    ], className='mb-4'),
     dbc.Row([
        dbc.Col([
            html.H3('Select currency pair:'),
            dcc.Dropdown(
                id='currency-dropdown',
                options=[{'label': currency, 'value': currency} for currency in available_currencies],
                value='BTC/USDT'
            )
        ], width=4),
    ], className='mb-4'),
    dbc.Row([
        dbc.Col([
            html.H3('Final balance (Buy and hold):'),
            html.P(id='final-balance-hold', className='lead')
        ], width=4),
        dbc.Col([
            html.H3('Percentage return (Buy and hold):'),
            html.P(id='percentage-return-hold', className='lead')
        ], width=4),
        dbc.Col([
            html.H3('Success rate:'),
            html.P(id='success-rate', className='lead')
        ], width=4),
    ], className='mb-4'),
    dbc.Row(dbc.Col(dcc.Graph(id='price-chart'), width=12)),
    dbc.Row(dbc.Col(dash_table.DataTable(id='trades-table',
                                         columns=[{'name': 'Timestamp', 'id': 'timestamp'},
                                                  {'name': 'Price', 'id': 'price'},
                                                  {'name': 'Action', 'id': 'action'}],
                                         style_cell={'textAlign': 'left'},
                                         style_header={'backgroundColor': 'rgb(230, 230, 230)',
                                                       'fontWeight': 'bold'},
                                         page_size=10), width=12)),
    dcc.Interval(
        id='interval-component',
        interval=60*60*1000,  # Atualiza a cada hora
        n_intervals=0
    )
], fluid=True)

input_shape = (60, 4)  # Supondo que você esteja usando uma janela de 60 períodos e 4 recursos (OHLC)
model = create_rnn_model(input_shape)
model.load_weights('rnn_model_weights.h5')

ohlcv_data = fetch_ohlcv_data()
_, _, _, _, scaler = prepare_data(ohlcv_data)

@app.callback(
    [Output('backtest-period', 'children'),
     Output('initial-balance', 'children'),
     Output('final-balance-strategy', 'children'),
     Output('percentage-return-strategy', 'children'),
     Output('final-balance-hold', 'children'),
     Output('percentage-return-hold', 'children'),
     Output('total-trades', 'children'),
     Output('successful-trades', 'children'),
     Output('success-rate', 'children'),
     Output('price-chart', 'figure'),
     Output('trades-table', 'data')],
     [Input('interval-component', 'n_intervals'),
     Input('currency-dropdown', 'value')]
)

def update_dashboard(n, selected_currency):
    ohlcv_data = fetch_ohlcv_data(symbol=selected_currency)
    ema_short, ema_long, rsi = calculate_indicators(ohlcv_data)
    actions = rnn_trading_strategy(model, ohlcv_data, scaler)
    final_balance, trades, successful_trades = backtest(ohlcv_data, actions)
    buy_and_hold_balance = buy_and_hold(ohlcv_data, 10000)

    start_time = datetime.datetime.fromtimestamp(ohlcv_data[0, 0] / 1000).strftime('%Y-%m-%d %H:%M:%S')
    end_time = datetime.datetime.fromtimestamp(ohlcv_data[-1, 0] / 1000).strftime('%Y-%m-%d %H:%M:%S')
    initial_balance = 10000
    percentage_return = (final_balance / initial_balance - 1) * 100
    buy_and_hold_return = (buy_and_hold_balance / initial_balance - 1) * 100
    success_rate = (successful_trades / trades) * 100

    price_chart = go.Figure()
    price_chart.add_trace(go.Candlestick(x=[datetime.datetime.fromtimestamp(ts / 1000) for ts in ohlcv_data[:, 0]],
                                         open=ohlcv_data[:, 1], high=ohlcv_data[:, 2], low=ohlcv_data[:, 3], close=ohlcv_data[:, 4],
                                         name='Price'))
    price_chart.add_trace(go.Scatter(x=[datetime.datetime.fromtimestamp(ts / 1000) for ts in ohlcv_data[:, 0]], y=ema_short, mode='lines', name='EMA Short'))
    price_chart.add_trace(go.Scatter(x=[datetime.datetime.fromtimestamp(ts / 1000) for ts in ohlcv_data[:, 0]], y=ema_long, mode='lines', name='EMA Long'))
    price_chart.update_layout(xaxis_rangeslider_visible=False)

    buy_signals = np.where(np.array(actions) == 'buy')[0]
    sell_signals = np.where(np.array(actions) == 'sell')[0]

    price_chart.add_trace(go.Scatter(x=[datetime.datetime.fromtimestamp(ohlcv_data[i, 0] / 1000) for i in buy_signals],
                                     y=ohlcv_data[buy_signals, 4], mode='markers', name='Buy', marker=dict(color='green', size=8)))
    price_chart.add_trace(go.Scatter(x=[datetime.datetime.fromtimestamp(ohlcv_data[i, 0] / 1000) for i in sell_signals],
                                     y=ohlcv_data[sell_signals, 4], mode='markers', name='Sell', marker=dict(color='red', size=8)))

    trades_data = []
    for i in range(len(actions)):
        if actions[i] == 'buy' or actions[i] == 'sell':
            trades_data.append({
                'timestamp': datetime.datetime.fromtimestamp(ohlcv_data[i, 0] / 1000),
                'price': ohlcv_data[i, 4],
                'action': actions[i]
            })

    return (f"{start_time} - {end_time}",
            f"{initial_balance} USDT",
            f"{final_balance:.2f} USDT",
            f"{percentage_return:.2f}%",
            f"{buy_and_hold_balance:.2f} USDT",
            f"{buy_and_hold_return:.2f}%",
            f"{trades}",
            f"{successful_trades}",
            f"{success_rate:.2f}%",
            price_chart,
            trades_data)

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=os.environ['PORT'])