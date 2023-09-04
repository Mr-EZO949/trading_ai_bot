import alpaca_trade_api as tradeapi
import yfinance as yf
import pandas as pd
import talib
import time
import datetime

import logging

# Set up logging
logging.basicConfig(filename='bot_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set your Alpaca API credentials
APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'
APCA_API_KEY_ID = 'PKKITUPUAUDE42VCHB2I'
APCA_API_SECRET_KEY = 'izpr9lb8TfqxFHdET7N5HvXqOzjxtZj9A0tTWfbQ'



# Initialize the Alpaca API client
api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, api_version='v2')

# Define different trading strategies
def simple_moving_average_strategy(data, period1, period2):
    if len(data) < max(period1, period2):
        return 'hold'  # Not enough data for calculation
    
    sma1 = data['Close'].rolling(window=period1).mean()
    sma2 = data['Close'].rolling(window=period2).mean()

    if sma1.iloc[-1] > sma2.iloc[-1]:
        return 'buy'
    elif sma1.iloc[-1] < sma2.iloc[-1]:
        return 'sell'
    else:
        return 'hold'

def macd_strategy(data, short_period=12, long_period=26, signal_period=9):
    ema_short = data['Close'].ewm(span=short_period).mean()
    ema_long = data['Close'].ewm(span=long_period).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal_period).mean()
    
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        return 'buy'
    elif macd_line.iloc[-1] < signal_line.iloc[-1]:
        return 'sell'
    else:
        return 'hold'


def bollinger_bands_strategy(data, window=20, num_std_dev=2):
    sma = data['Close'].rolling(window=window).mean()
    std_dev = data['Close'].rolling(window=window).std()
    
    upper_band = sma + num_std_dev * std_dev
    lower_band = sma - num_std_dev * std_dev
    
    if data['Close'].iloc[-1] > upper_band.iloc[-1]:
        return 'sell'
    elif data['Close'].iloc[-1] < lower_band.iloc[-1]:
        return 'buy'
    else:
        return 'hold'



def calculate_rsi(data, window):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def advanced_rsi_strategy(data, rsi_window, rsi_overbought, rsi_oversold):
    rsi = calculate_rsi(data['Close'], rsi_window)

    if rsi.iloc[-1] > rsi_overbought:
        return 'sell'
    elif rsi.iloc[-1] < rsi_oversold:
        return 'buy'
    else:
        return 'hold'

# Combine strategies based on market conditions
def combined_strategy(data, conditions):
    if conditions['trend_up'] and conditions['oversold']:
        print("sma strategy")
        return simple_moving_average_strategy(data, 20, 50)
    elif conditions['volatility_high']:
        print("advanced rsi strategy")
        return advanced_rsi_strategy(data, rsi_window=14, rsi_overbought=70, rsi_oversold=30)
    elif conditions['bollinger_bands_signal']:
        print("bollinger bands strategy")
        return bollinger_bands_strategy(data)
    elif conditions['macd_signal']:
        print("macd strategy")
        return macd_strategy(data)
    else:
        return None


# def rsi_strategy(data, rsi_threshold):
#     delta = data['Close'].diff()
#     gain = delta.where(delta > 0, 0)
#     loss = -delta.where(delta < 0, 0)

#     avg_gain = gain.rolling(window=7).mean()
#     avg_loss = loss.rolling(window=7).mean()

#     rs = avg_gain / avg_loss
#     rsi = 100 - (100 / (1 + rs))

#     if rsi.iloc[-1] > rsi_threshold:
#         return 'sell'
#     elif rsi.iloc[-1] < 100 - rsi_threshold:
#         return 'buy'
#     else:
#         return 'hold'

# # Combine strategies based on market conditions
# def combined_strategy(data, conditions):
#     if conditions['trend_up'] and conditions['oversold']:
#         print("sma strategy")
#         return simple_moving_average_strategy(data, 20, 50)
#     elif conditions['volatility_high']:
#         print("rsi_strategy")
#         return rsi_strategy(data, 70)
#     else:

#         return None

def get_historical_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Main trading loop
def trading_loop(symbol, start_date, end_date):
    while True:
        current_time = datetime.datetime.now()
        
        # Break the loop if the current time is after the specified end date
        # if current_time > datetime.strptime(end_date):
        #     break

        # Fetch historical data
        data = get_historical_data(symbol, start_date, end_date)
        
        # Analyze market conditions
        conditions = analyze_market_conditions(data)
        if conditions is None: continue
        
        # Choose strategy based on conditions
        selected_strategy = combined_strategy(data, conditions)
        print(selected_strategy)
        
        if selected_strategy:
            # Execute the selected strategy
            execute_strategy(selected_strategy)

        # close_trades(api, symbol)
        
        time.sleep(60)   # Wait for a minute before making the next decision

def analyze_market_conditions(data):
    # Calculate SMA for short and long periods
    sma_short = data['Close'].rolling(window=20).mean()
    sma_long = data['Close'].rolling(window=50).mean()
    
    # Calculate RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # Inside the analyze_market_conditions function
    atr = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
    
    # Calculate Bollinger Bands
    middle_band = data['Close'].rolling(window=20).mean()
    std_dev = data['Close'].rolling(window=20).std()
    upper_band = middle_band + 2 * std_dev
    lower_band = middle_band - 2 * std_dev
    
    # Calculate MACD
    macd, signal_line, _ = talib.MACD(data['Close'])
    
    # Implement logic to analyze market conditions based on indicators
    conditions = {
        'trend_up': sma_short.iloc[-1] > sma_long.iloc[-1],
        'oversold': rsi.iloc[-1] < 30,
        'volatility_high': atr.iloc[-1] > 1.0,
        'bollinger_bands_signal': data['Close'].iloc[-1] < lower_band.iloc[-1],
        'macd_signal': macd.iloc[-1] > signal_line.iloc[-1]
    }
    return conditions



def calculate_optimal_quantity(api, symbol, risk_percentage=0.02):
    account = api.get_account()
    available_capital = float(account.buying_power)
    current_price = api.get_latest_trade(symbol=symbol).price
    optimal_quantity = int((available_capital * risk_percentage) / current_price)
    return optimal_quantity

def close_trades(api, symbol):
    positions = api.list_positions()
    for position in positions:
        if position.symbol == symbol:
            current_price = api.get_latest_trade(symbol=symbol).price
            if position.side == 'long' and float(current_price) > float(position.avg_entry_price):
                try:
                    api.submit_order(
                        symbol=symbol,
                        qty=position.qty,
                        side='sell',
                        type='market',
                        time_in_force='gtc'
                    )
                    logging.info(f'Sell order executed to close position: {position.qty} shares of {symbol}')
                except Exception as e:
                    logging.error(f"Error executing sell order to close position: {e}")

def execute_strategy(strategy):
    global api  # You might need to adjust this depending on your code structure
    
    # Define trade parameters
    symbol = 'NVDA'
    print("Trading", symbol)
    quantity = calculate_optimal_quantity(api, symbol)
    
    if strategy == 'buy':
        try:
            api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            logging.info(f'Buy order executed: {quantity} shares of {symbol}')
        except Exception as e:
            logging.error(f"Error executing buy order: {e}")
    
    elif strategy == 'sell':
        try:
            api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            logging.info(f'Sell order executed: {quantity} shares of {symbol}')
        except Exception as e:
            logging.error(f"Error executing sell order: {e}")

if __name__ == '__main__':
    symbol = 'NVDA'  # Stock symbol to trade

    start_date = '2023-01-01'
    end_date = '2023-08-31'


    trading_loop(symbol, start_date, end_date)
