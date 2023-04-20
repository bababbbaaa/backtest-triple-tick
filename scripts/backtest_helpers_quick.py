from functools import partial
from multiprocessing import freeze_support
import os
import sys
import timeit
import numpy as np
import pandas as pd
import glob
from itertools import product
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

leverage = 8
loss_percent = 0
atr_multiplier = 1.5
period_days = 30

# abspath of '../public'
public_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'public')) + '/'

def simulate_trades(result, data):
    balance = 5000.0
    ticks_elapsed = lower_bound = target = stop = 0
    wins = losses = draws = 0
    position = entry_price = 0
    same_tick_counter = first_trade_timestamp = 0
    entry_counter = stop_triggered = max_loss_rate = 0
    prev_balance = prev_position = prev_ticks_elapsed = 0
    ma_trend = 0

    atr_tick_multiplier = result['conditions']['atr_tick_multiplier']
    same_tick = result['conditions']['same_tick']
    back_size = result['conditions']['back_size']
    profit_percent = result['conditions']['profit_percent']
    same_holding = result['conditions']['same_holding']
    divide = result['conditions']['divide']
    trade_tick = result['conditions']['trade_tick']

    print("current conditions", profit_percent, atr_tick_multiplier, trade_tick, same_tick, same_holding, divide, back_size)

    for _, row in data.iterrows():
        [timestamp, open, high, low, close, atr, ema] = [row['timestamp'], row['open'], row['high'], row['low'], row['close'], row['atr'], row['ema']]
        tick_size = atr_tick_multiplier * atr
        
        if ema > close:
            ma_trend += 1
        else:
            ma_trend = 0
        
        if ticks_elapsed == 0:
            lower_bound = open

        if high >= target and position > 0:
            wins += 1
            fee = (position * target) * 0.0004
            profit = position * (target - entry_price) - fee
            balance += profit
            position = 0
            entry_price = 0
            target = 0
            stop = 0
            ticks_elapsed = 0
            entry_counter = 0
            same_tick_counter = 0

        elif (low <= stop and position > 0) or (stop_triggered == 1 and position > 0 and prev_position > 0):
            stop_triggered = 0
            if stop == 0:
                stop = open
            fee = (position * stop) * 0.0004
            loss = position * (stop - entry_price) - fee
            balance += loss
            position = 0
            entry_price = 0
            target = 0
            stop = 0
            ticks_elapsed = 0
            entry_counter = 0
            same_tick_counter = 0

            if prev_balance < balance:
                draws += 1
            else:
                losses += 1

        if balance > 0:
            if lower_bound - close >= tick_size:
                same_tick_counter = 0
                ticks_elapsed += 1
                lower_bound = close

                if ticks_elapsed >= trade_tick and entry_counter < divide: # and (entry_count == 0 and row['rsi'] < 120) or (entry_count > 0)):
                    if ma_trend > 20 and entry_counter == 0:
                        ticks_elapsed = 0
                    else:
                        if first_trade_timestamp == 0:
                            first_trade_timestamp = timestamp
                        entry_counter += 1
                        newly_size = balance / close * leverage * (1/4.0)
                        entry_price = ((entry_price * position) + (close * newly_size)) / (position + newly_size)
                        fee = newly_size * entry_price * 0.0004
                        balance -= fee
                        position += newly_size
                        target = entry_price + atr * atr_multiplier * profit_percent
                        if loss_percent > 0:
                            stop = entry_price - atr * atr_multiplier * loss_percent

            if ticks_elapsed > 0:
                if lower_bound - close <= tick_size * -1 * back_size and position == 0:
                    ticks_elapsed = 0
                    same_tick_counter = 0
                if ticks_elapsed == prev_ticks_elapsed if prev_ticks_elapsed else False:
                    same_tick_counter += 1
                    if same_tick_counter == same_tick and position == 0:
                        ticks_elapsed = 0
                        same_tick_counter = 0
                    if same_tick_counter == same_holding and position > 0 and prev_position > 0:
                        stop_triggered = 1
                else:
                    same_tick_counter = 0
                    
        if balance <= 100:
            break

        prev_balance = balance
        prev_position = position
        prev_ticks_elapsed = ticks_elapsed

    pnl = (balance - 5000.0)/5000.0 * 100
    if wins + losses == 0:
        win_rate = 0
    else:
        win_rate = wins / (wins + losses) * 100
    # print pnl, win_rate with 2 decimal places and win, lose, draw
    print(f"PNL: {pnl}%, Win Rate: {win_rate}%, Win: {wins}, Lose: {losses}, Draw: {draws}")
    result['result'] = {'win_rate': win_rate, 'pnl': pnl}
    return result

def get_param_combinations(step_size):
    profit_percents = np.arange(0.45, 0.81, step_size)
    atr_tick_multipliers = np.arange(0.5, 0.96, step_size)
    trade_ticks = np.arange(3, 4, 1)
    same_ticks = np.arange(2, 13, step_size * 10)
    same_holdings = np.arange(20, 21, 1)
    divides = np.arange(2, 5, 2)
    back_sizes = np.arange(0.5, 0.96, step_size)
    
    return list(product(profit_percents, atr_tick_multipliers, trade_ticks, same_ticks, same_holdings, divides, back_sizes))

def backtest(params, data):
    profit_percent, atr_tick_multiplier, trade_tick, same_tick, same_holding, divide, back_size = params
    result = {'conditions': {'profit_percent': profit_percent, 
                            'atr_tick_multiplier': atr_tick_multiplier,
                            'trade_tick': trade_tick,
                            'same_tick': same_tick,
                            'same_holding': same_holding,
                            'divide': divide,
                            'back_size': back_size}}
    result = simulate_trades(result, data)
    return result

def refine_search_space(best_conditions):
    step_size = 0.05
    profit_percents = np.arange(best_conditions['profit_percent'] - step_size, best_conditions['profit_percent'] + step_size + 0.01, step_size)
    atr_tick_multipliers = np.arange(best_conditions['atr_tick_multiplier'] - step_size, best_conditions['atr_tick_multiplier'] + step_size + 0.01, step_size)
    back_sizes = np.arange(best_conditions['back_size'] - step_size, best_conditions['back_size'] + step_size + 0.01, step_size)
    same_ticks = np.arange(best_conditions['same_tick'] - step_size * 20, best_conditions['same_tick'] + step_size * 20 + 0.01, step_size * 20)
    
    return list(product(profit_percents, atr_tick_multipliers, [best_conditions['trade_tick']], same_ticks, [best_conditions['same_holding']], [best_conditions['divide']], back_sizes))

def get_df():
    # load result.csv if exists, else create new one
    if glob.glob(public_path + 'result.csv'):
        print('## Load result.csv')
        df = pd.read_csv(public_path + 'result.csv')
    else:
        df = pd.DataFrame(columns=[
            'symbol',
            'pnl',
            'win_rate',
            'profit_percent',
            'atr_tick_multiplier',
            'trade_tick',
            'same_tick',
            'same_holding',
            'divide',
            'back_size',
            'backtesting_datetime',
            'data_first_time',
            'data_last_time',
        ]) 
    return df

def get_symbols(symbols, df):
    if symbols is None or symbols == 'all':
        filenames = glob.glob(public_path + 'klines/*_5m.csv')
        symbols = [filename.split('/')[-1].split('_')[0] for filename in filenames]
    elif symbols == 'exists':
        symbols = df['symbol'].unique().tolist()
    else:
        symbols = symbols.split(',') if isinstance(symbols, str) else symbols
    print("target symbols: {}", symbols)
    return symbols

def convert_time(time):
    if time is not None:
        time = pd.to_datetime(time).timestamp()
    return time

def get_klines_data(symbol, first_time = None, last_time = None):
    data = pd.read_csv(public_path + 'klines/{}_5m.csv'.format(symbol), parse_dates=['timestamp'])
    data = data[['timestamp', 'open', 'high', 'low', 'close', 'atr', 'rsi', 'ema']]

    # data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    if first_time is not None and last_time is not None:
        data = data.loc[(data['timestamp'] >= first_time) & (data['timestamp'] <= last_time)]
    return data

def convert_df_to_results(df, symbol):
    best_results = df.loc[df['symbol'] == symbol].sort_values(by='pnl', ascending=False).iloc[0].to_dict()
    best_results['conditions'] = {
        'profit_percent': best_results['profit_percent'],
        'atr_tick_multiplier': best_results['atr_tick_multiplier'],
        'trade_tick': best_results['trade_tick'],
        'same_tick': best_results['same_tick'],
        'same_holding': best_results['same_holding'],
        'divide': best_results['divide'],
        'back_size': best_results['back_size'],
    }
    best_results['result'] = { 'pnl': best_results['pnl'], 'win_rate': best_results['win_rate'] }
    return best_results

def get_data_in_range(data, first_time, last_time):
    # data["timestamp"] = string("1580515200")
    # first_time = Timestamp("2020-01-01 00:00:00")
    # last_time = Timestamp("2020-01-31 00:00:00")
    # so, "return data.loc[(data['timestamp'] >= first_time) & (data['timestamp'] <= last_time)]" will occur error.
    return data.loc[(pd.to_datetime(data['timestamp'], unit="s") >= first_time) & (pd.to_datetime(data['timestamp'], unit="s") <= last_time)]

# result will be like this:
# {"symbol": "BTCUSDT",
#  "pnl_sum": 0.0,
#  "win_rate_avg": 0.0,
#  "periods": [
#    {"period_from": "2021-01-01 00:00:00", "period_to": "2021-01-15 00:00:00", "pnl": 0.0, "win_rate": 0.0, "conditions": {"profit_percent": 0.1, "atr_tick_multiplier": 1.0, "trade_tick": 0.5, "same_tick": 0.5, "same_holding": 0.5, "divide": 0.5, "back_size": 0.5}},
#    {"period_from": "2021-01-15 00:00:00", "period_to": "2021-01-30 00:00:00", "pnl": 0.0, "win_rate": 0.0, "conditions": {"profit_percent": 0.1, "atr_tick_multiplier": 1.0, "trade_tick": 0.5, "same_tick": 0.5, "same_holding": 0.5, "divide": 0.5, "back_size": 0.5}},
#  ],
# }
def run_backtest(symbols = None, step_size = 0.2, first_time = None, last_time = None):
    df = get_df()
    symbols = get_symbols(symbols, df)
    
    first_time = pd.to_datetime(first_time).timestamp() if first_time is not None else None
    
    all_result = []
    for symbol in symbols:
        print('## Symbol: {} ({}/{})'.format(symbol, symbols.index(symbol) + 1, len(symbols)))

        data = get_klines_data(symbol)
        
        if data.empty:
            print('## Skip Backtesting... data is empty')
            continue
        
        data_first_time, data_last_time = pd.to_datetime(data['timestamp'].iloc[0], unit="s"), pd.to_datetime(data['timestamp'].iloc[-1], unit="s")

        print("data_first_time: {}, first_time: {}".format(data_first_time, first_time))
        
        first_time = pd.to_datetime(first_time, unit="s") if first_time is not None else None
        
        if first_time is not None and data_first_time > first_time:
            print('## Skip Backtesting... data_first_time is later than first_time, {} > {}'.format(data_first_time, first_time))
            continue
        
        print('we are testing {} {} {}'.format(symbol, data_first_time, data_last_time))


        print('## Start Backtesting...')

        results = []
        best_results = None
        
        if not symbol in df['symbol'].values:
            continue

        symbol_result = { 'symbol': symbol, 'pnl_sum': 0.0, 'win_rate_avg': 0.0, 'periods': [] }
        next_period_from = first_time
        while True:
            period_from = next_period_from
            if data_last_time < period_from + pd.Timedelta(days=period_days*2):
                break
            period_to = period_from + pd.Timedelta(days=period_days*2)

            data_in_range = get_data_in_range(data, period_from, period_to)
            print('## Period: {} ~ {}'.format(period_from, period_to))
            
            # convert for result['conditions'] each results to list of tuple
            results = convert_df_to_results(df, symbol)
            param_combi = [tuple(results['conditions'].values())]
            
            
            with ProcessPoolExecutor() as executor:
                backtest_with_data = partial(backtest, data=data_in_range)
                final_results = list(executor.map(backtest_with_data, param_combi))
            # Select best pnl final result
            final_result = sorted(final_results, key=lambda x: x['result']['pnl'], reverse=True)[0]
            if final_result['result']['pnl'] < 0:
                break
            result = final_result
            new_result = {
                'symbol': symbol,
                'pnl': result['result']['pnl'],
                'win_rate': result['result']['win_rate'],
                'profit_percent': result['conditions']['profit_percent'],
                'atr_tick_multiplier': result['conditions']['atr_tick_multiplier'],
                'trade_tick': result['conditions']['trade_tick'],
                'same_tick': result['conditions']['same_tick'],
                'same_holding': result['conditions']['same_holding'],
                'divide': result['conditions']['divide'],
                'back_size': result['conditions']['back_size'],
                'backtesting_datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_first_time': pd.to_datetime(data_in_range['timestamp'].iloc[0], unit="s").strftime('%Y-%m-%d %H:%M:%S'),
                'data_last_time': pd.to_datetime(data_in_range['timestamp'].iloc[-1], unit="s").strftime('%Y-%m-%d %H:%M:%S'),
            }
            # round float values to 2 decimal places
            for key, value in new_result.items():
                if isinstance(value, float):
                    new_result[key] = round(value, 2)
            print('## Save backtesting results to result.csv: {}'.format(new_result))
            df = pd.concat([df, pd.DataFrame([new_result])], ignore_index=True)
            # sort df by symbol to A-Z and pnl to high to low
            df = df.sort_values(by=['symbol', 'pnl'], ascending=[True, False])
            df.to_csv(public_path + 'result.csv', index=False)
            symbol_result['periods'].append({
                "period_from": period_from,
                "period_to": period_to,
                "pnl": final_result['result']['pnl'],
                "win_rate": final_result['result']['win_rate'],
                "conditions": final_result['conditions'],
            })
            print("## Step 3: Best PNL conditions: {}".format(final_result))
            next_period_from = period_to
        
        if len(symbol_result['periods']) == 0:
            continue
        
        symbol_result['pnl_sum'] = sum([period['pnl'] for period in symbol_result['periods']])
        symbol_result['win_rate_avg'] = sum([period['win_rate'] for period in symbol_result['periods']]) / len(symbol_result['periods'])
        
        print("symbol_result: {}".format(symbol_result))
    
        
