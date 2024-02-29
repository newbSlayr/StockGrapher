import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Define the stock symbol and timeframe
symbol = 'GBPJPY=X'
start_date = pd.Timestamp.today() - pd.Timedelta(days=3)
end_date = pd.Timestamp.today()

# Get intraday data using yfinance
data = yf.download(symbol, start=start_date, end=end_date, interval='1m')

# Clean up the data
data.drop(columns=['Adj Close'], inplace=True)
data.columns = [col.lower() for col in data.columns]

# Define EMA Calculation
def calculate_EMA(data, window):
    return data.ewm(span=window, adjust=False).mean()

# Define RSI Calculation
def calculate_RSI(data, window):
    delta = data.diff()
    up_days = delta.clip(lower=0)
    down_days = -1 * delta.clip(upper=0)
    RS = up_days.rolling(window).mean() / down_days.rolling(window).mean()
    return 100 - (100 / (1 + RS))


# Define Stochastic RSI Calculation
def calculate_stochastic_rsi(data, window=14):
    rsi = calculate_RSI(data, window)
    min_rsi = rsi.rolling(window=window).min()
    max_rsi = rsi.rolling(window=window).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi)
    return stoch_rsi


# Calculate 200 EMA
data['200 EMA'] = calculate_EMA(data['close'], 200)

# Calculate RSI and Stochastic RSI
data['RSI'] = calculate_RSI(data['close'], 14)
data['Stoch_RSI'] = calculate_stochastic_rsi(data['close'], 14)


# Create the figure
fig = go.Figure()

# Add Candlestick plot
fig.add_trace(go.Candlestick(x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'))

# Add 200 EMA trace
fig.add_trace(go.Scatter(x=data.index, y=data['200 EMA'], line=dict(color='white', width=1.5), name='200 EMA'))

# Add RSI plot
fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', yaxis='y2'))

# Add Stochastic RSI plot
fig.add_trace(go.Scatter(x=data.index, y=data['Stoch_RSI'], name='Stoch RSI', yaxis='y3'))

# Update layout
fig.update_layout(title=f'{symbol} Stock Price',
                xaxis=dict(domain=[0, 1], rangeslider=dict(visible=False)),
                yaxis=dict(domain=[0.3, 1], title='Price'),
                yaxis2=dict(domain=[0.2, 0.3], title='RSI', fixedrange=True),
                yaxis3=dict(domain=[0, 0.2], title='Stoch RSI', fixedrange=True),
                template='plotly_dark')

# Show the plot
fig.show()
