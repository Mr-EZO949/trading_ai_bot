import alpaca_trade_api as tradeapi
import yfinance as yf
import pandas as pd
import talib
import time
import datetime
import requests.exceptions
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

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
        print("Trading:", symbol)
        current_time = datetime.datetime.now()
        
        # Break the loop if the current time is after the specified end date
        # if current_time > datetime.strptime(end_date):
        #     break

        # Fetch historical data
        data = get_historical_data(symbol, start_date, end_date)
        
        # Define parameters for technical indicators
        short_period = 20
        long_period = 50
        macd_params = {'short_period': 12, 'long_period': 26, 'signal_period': 9}
        bollinger_params = {'window': 20, 'num_std_dev': 2}
        rsi_params = {'rsi_window': 14, 'rsi_overbought': 70, 'rsi_oversold': 30}

        # Choose strategy based on multiple indicators
        selected_strategy = multiple_indicators_strategy(data, short_period, long_period, macd_params, bollinger_params, rsi_params)
        print("\n")
        print("Selected strategy:", selected_strategy)
        
        if selected_strategy == 'buy':
            # Execute the buy strategy
            execute_buy_order(symbol)

        elif selected_strategy == 'sell':
            # Execute the sell strategy
            execute_sell_order(symbol)
        
        # Implement additional logic for hold or other strategies
        
        # close_trades(api, symbol)
        
        time.sleep(60)   # Wait for a minute before making the next decision

def get_fundamental_data(symbol):
    try:
        # Check if the symbol is valid (exists in Yahoo Finance)
        stock = yf.Ticker(symbol)
        info = stock.info  # Fetch stock information to check if it's valid
        
        if info is None:
            print(f"Invalid symbol: {symbol}")
            return None, None, None, None

        # Get the latest financial data
        financials = stock.financials
        if not financials.empty:
            # Example: Check if 'PriceToEarningsRatio' exists and use it
            if 'PriceToEarningsRatio' in financials.columns:
                pe_ratio = financials['PriceToEarningsRatio'].iloc[-1]
            else:
                print(f"P/E ratio not found for {symbol}")
                pe_ratio = None

        # Get the latest balance sheet data
        balance_sheet = stock.balance_sheet
        if not balance_sheet.empty:
            # Extract the most recent P/B ratio
            if 'PriceToBookRatio' in balance_sheet.columns:
                pb_ratio = balance_sheet['PriceToBookRatio'].iloc[-1]
            else:
                print(f"P/B ratio not found for {symbol}")
                pb_ratio = None

            # Extract Debt-to-Equity (D/E) ratio
            de_ratio = balance_sheet['Total Debt'] / balance_sheet['Total Equity']

            # Extract Dividend Yield
            dividend_yield = financials['Forward Dividend & Yield'].iloc[-1]

        return pe_ratio, pb_ratio, de_ratio, dividend_yield

    except requests.exceptions.RequestException as conn_error:
        print(f"Connection error while fetching data for {symbol}: {conn_error}")
        return None, None, None, None
    except Exception as e:
        print(f"Error fetching fundamental data for {symbol}: {e.__class__.__name__} - {str(e)}")
        return None, None, None, None
def fundamental_analysis(pe_ratio, pb_ratio, de_ratio, dividend_yield, custom_rules=None):
    # Define default analysis rules
    default_rules = {
        'PE_Ratio_Buy_Threshold': 15,
        'PB_Ratio_Buy_Threshold': 1,
        'DE_Ratio_Sell_Threshold': 2,
        'Dividend_Yield_Buy_Threshold': 0.03
    }

    # Merge custom rules with default rules
    analysis_rules = {**default_rules, **(custom_rules or {})}

    # Evaluate buy/sell/hold signals based on fundamental metrics
    signals = []
    if pe_ratio is not None:
        if pe_ratio < analysis_rules['PE_Ratio_Buy_Threshold']:
            signals.append('Buy')
        else:
            signals.append('Hold')
    else:
        signals.append('Hold')

    if pb_ratio is not None:
        if pb_ratio < analysis_rules['PB_Ratio_Buy_Threshold']:
            signals.append('Buy')
        else:
            signals.append('Hold')
    else:
        signals.append('Hold')

    if de_ratio is not None:
        if de_ratio > analysis_rules['DE_Ratio_Sell_Threshold']:
            signals.append('Sell')
        else:
            signals.append('Hold')
    else:
        signals.append('Hold')

    if dividend_yield is not None:
        if dividend_yield > analysis_rules['Dividend_Yield_Buy_Threshold']:
            signals.append('Buy')
        else:
            signals.append('Hold')
    else:
        signals.append('Hold')

    # The final signal is determined by a majority vote among the signals
    if signals.count('Buy') > signals.count('Sell'):
        return 'Buy'
    elif signals.count('Sell') > signals.count('Buy'):
        return 'Sell'
    else:
        return 'Hold'

def execute_sell_order(symbol):
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

def execute_buy_order(symbol):

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


# Import the necessary libraries and set up logging and API credentials as before...

# Define different trading strategies as before...

# Combine strategies based on market conditions
def multiple_indicators_strategy(data, short_period, long_period, macd_params, bollinger_params, rsi_params):
    # Initialize counters for buy and sell signals
    buy_signals = 0
    hold_signals = 0
    sell_signals = 0
    
    # Check signals from each indicator and count buy and sell signals
    sma_strategy = simple_moving_average_strategy(data, short_period, long_period)
    if sma_strategy == 'buy':
        print("sma_strategy: buy", end='\t')
        buy_signals += 1
    elif sma_strategy == 'sell':
        print("sma_strategy: sell", end='\t')
        sell_signals += 1
    else:
        hold_signals += 1
        print("sma_strategy: hold", end='\t')
    
    macd_strat = macd_strategy(data, **macd_params)
    if macd_strat == 'buy':
        buy_signals += 1
        print("macd_strat: buy", end='\t')
    elif macd_strat == 'sell':
        print("macd_strat: sell", end='\t')
        sell_signals += 1
    else:
        hold_signals += 1
        print("macd_strat: hold", end='\t')
    
    bb_strat = bollinger_bands_strategy(data, **bollinger_params)
    if bb_strat == 'buy':
        print("bb_strat: buy", end='\t')
        buy_signals += 1
    elif bb_strat == 'sell':
        print("bb_strat: sell", end='\t')
        sell_signals += 1
    else:
        hold_signals += 1
        print("bb_strat: hold", end='\t')
    
    rsi_strat = advanced_rsi_strategy(data, **rsi_params)
    if rsi_strat == 'buy':
        print("rsi_strat: buy", end='\t')
        buy_signals += 1
    elif rsi_strat == 'sell':
        print("rsi_strat: sell", end='\t')
        sell_signals += 1
    else:
        hold_signals += 1
        print("rsi_strat: hold", end='\t')


    # pe_ratio, pb_ratio, de_ratio, dividend_yield = get_fundamental_data(symbol)
    # custom_rules = {
    #     'PE_Ratio_Buy_Threshold': 12,  # Custom threshold for P/E ratio
    #     'Dividend_Yield_Buy_Threshold': 0.035  # Custom threshold for dividend yield
    # }
    # recommendation = fundamental_analysis(pe_ratio, pb_ratio, de_ratio, dividend_yield, custom_rules)
    # if recommendation == 'Buy':
    #     print("fundamental analysis: buy", end='\t')
    #     buy_signals += 1
    # elif recommendation == 'Sell':
    #     print("fundamental analysis: sell", end='\t')
    #     sell_signals += 1
    # else:
    #     hold_signals += 1
    #     print("fundamental analysis: hold", end='\t')
    # # Decide based on the majority vote
    if buy_signals > sell_signals and buy_signals > hold_signals:
        return 'buy'
    elif sell_signals > buy_signals and sell_signals > hold_signals:
        return 'sell'
    else:
        return 'hold'



# Rest of your code remains the same...



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


if __name__ == '__main__':
    symbol = 'LBRT'  # Stock symbol to trade

    start_date = '2023-01-01'
    end_date = '2023-08-31'


    trading_loop(symbol, start_date, end_date)
