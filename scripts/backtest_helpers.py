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
                            'back_size': back_size, 
                            'atr_multiplier': atr_multiplier,
                            'leverage': leverage}}
    result = simulate_trades(result, data)
    return result

def refine_search_space(best_conditions):
    step_size = 0.05
    profit_percents = np.arange(best_conditions['profit_percent'] - step_size, best_conditions['profit_percent'] + step_size + 0.01, step_size)
    atr_tick_multipliers = np.arange(best_conditions['atr_tick_multiplier'] - step_size, best_conditions['atr_tick_multiplier'] + step_size + 0.01, step_size)
    back_sizes = np.arange(best_conditions['back_size'] - step_size, best_conditions['back_size'] + step_size + 0.01, step_size)
    same_ticks = np.arange(best_conditions['same_tick'] - step_size * 20, best_conditions['same_tick'] + step_size * 20 + 0.01, step_size * 20)
    
    return list(product(profit_percents, atr_tick_multipliers, [best_conditions['trade_tick']], same_ticks, [best_conditions['same_holding']], [best_conditions['divide']], back_sizes))

def run_backtest(symbols = None, step_size = 0.2, first_time = None, last_time = None):
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
    if symbols is None or symbols == 'all':
        filenames = glob.glob(public_path + 'klines/*_5m.csv')
        symbols = [filename.split('/')[-1].split('_')[0] for filename in filenames]
    else:
        symbols = symbols.split(',') if isinstance(symbols, str) else symbols
    print("target symbols: {}", symbols)

    if first_time is not None and last_time is not None:
        # convert first_time and last_time from '2020-01-01' to '1580512000'
        first_time = pd.to_datetime(first_time).strftime('%Y-%m-%d %H:%M:%S')
        last_time = pd.to_datetime(last_time).strftime('%Y-%m-%d %H:%M:%S')
    for symbol in symbols:
        print('## Symbol: {} ({}/{})'.format(symbol, symbols.index(symbol) + 1, len(symbols)))

        data = pd.read_csv(public_path + 'klines/{}_5m.csv'.format(symbol), parse_dates=['timestamp'])
        data = data[['timestamp', 'open', 'high', 'low', 'close', 'atr', 'rsi', 'ema']]

        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
        if first_time is not None and last_time is not None:
            data = data.loc[(data['timestamp'] >= first_time) & (data['timestamp'] <= last_time)]
        data_first_time, data_last_time = data['timestamp'].iloc[0], data['timestamp'].iloc[-1]
        data_first_time = pd.to_datetime(data_first_time).strftime('%Y-%m-%d %H:%M:%S')
        data_last_time = pd.to_datetime(data_last_time).strftime('%Y-%m-%d %H:%M:%S')
        print('we are testing {} {} {}'.format(symbol, data_first_time, data_last_time))

        same_data_df = df.loc[(df['symbol'] == symbol) & (df['data_first_time'] == data_first_time) & (df['data_last_time'] == data_last_time)]
        if len(same_data_df) > 2:
            print('## Skip Backtesting... already exists {}', same_data_df)
            continue

        results = []

        print('## Start Backtesting...')

        best_results = None  # Add this line to initialize the variable
    
        # Check to see if there is data of symbol for a win rate of 90% or higher.
        
        # If df contains data for symbol and the win rate is 90% or higher, then refine the search space to move forward to the next step.
        if symbol in df['symbol'].values and df.loc[df['symbol'] == symbol]['win_rate'].max() >= 90 and df.loc[df['symbol'] == symbol]['profit_percent'].min() >= 0.45:
            # skip step 1 and define best_results as the best result of df if the pnl is greater than 80
            best_results = df.loc[df['symbol'] == symbol].sort_values(by='pnl', ascending=False).iloc[0].to_dict()
            if best_results['pnl'] < 80:
                best_results = None
            else:
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
            print('## Step 1: Skip Backtesting... already exists {}', best_results)
        else:
            # Step 1: Perform a rough search
            param_combinations = get_param_combinations(step_size)
            total_step = len(param_combinations)
            print('## Step 1: Perform a rough search, total_step is {}'.format(total_step))

            with ProcessPoolExecutor() as executor:
                backtest_with_data = partial(backtest, data=data)
                results = list(executor.map(backtest_with_data, param_combinations))

            # Select the best PNL conditions among those with a win rate of 90% or more.
            best_results = sorted([result for result in results if result['result']['win_rate'] >= 90 and result['result']['pnl'] >= 80], key=lambda x: x['result']['pnl'], reverse=True)[0]
            print('## Step 1: Best PNL conditions: {}'.format(best_results))
        
        # check if step 1 is processed and if there is a condition with win_rate >= 90% && pnl >= 80
        if best_results is not None:
            # Step 2: Perform a more detailed search around the best condition found in Step 1
            refined_param_combinations = refine_search_space({
                'profit_percent': best_results['conditions']['profit_percent'],
                'atr_tick_multiplier': best_results['conditions']['atr_tick_multiplier'],
                'trade_tick': best_results['conditions']['trade_tick'],
                'same_tick': best_results['conditions']['same_tick'],
                'same_holding': best_results['conditions']['same_holding'],
                'divide': best_results['conditions']['divide'],
                'back_size': best_results['conditions']['back_size'],
            })
        else:
            print('## No condition with win_rate >= 90%, pnl >= 80 found in Step 1. Skipping Step 2.')
            continue

        # Execute backtesting for refined parameters
        total_step = len(refined_param_combinations)
        print('## Step 2: Perform a more detailed search, total_step is {}'.format(total_step))

        with ProcessPoolExecutor() as executor:
            backtest_with_data = partial(backtest, data=data)
            results += list(executor.map(backtest_with_data, refined_param_combinations))
            
        # Select the best 3 PNL conditions among those with a win rate of 90% or more.
        results = sorted([result for result in results if result['result']['win_rate'] >= 90], key=lambda x: x['result']['pnl'], reverse=True)[:3]
        
        # Save backtesting results to result.csv
        for result in results:
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
                'data_first_time': first_time,
                'data_last_time': last_time,
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
