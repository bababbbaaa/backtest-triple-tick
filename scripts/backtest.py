from functools import partial
from multiprocessing import freeze_support
import os
import sys
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

def test_trading(result, data):
    current_capital = 5000.0
    tick_count = lower = target_price = stop_price = 0
    win = lose = draw = 0
    position_size = entried_price = 0
    same_tick_count = first_trade_time = 0
    entry_count = trigger_stop = biggest_loss_rate = 0
    previous_capital = previous_position_size = previous_tick_count = 0
    ma_trend = 0
    atr_tick_multiplier = result['conditions']['atr_tick_multiplier']
    same_tick = result['conditions']['same_tick']
    back_size = result['conditions']['back_size']
    profit_percent = result['conditions']['profit_percent']
    same_holding = result['conditions']['same_holding']
    divide = result['conditions']['divide']
    trade_tick = result['conditions']['trade_tick']
    print("current conditions", atr_tick_multiplier, same_tick, back_size, profit_percent, same_holding, divide)
    for _, row in data.iterrows():
        time = row['timestamp']
        [open, high, low, close, atr] = [row['open'], row['high'], row['low'], row['close'], row['atr']]
        tick_size = atr_tick_multiplier * atr
        
        if row['ema'] > row['close']:
            ma_trend += 1
        else:
            ma_trend = 0
        
        if tick_count == 0:
            lower = open

        if high >= target_price and position_size > 0:
            win += 1
            fee = (position_size * target_price) * 0.0004
            profit = position_size * (target_price - entried_price) - fee
            current_capital += profit
            position_size = 0
            entried_price = 0
            target_price = 0
            stop_price = 0
            tick_count = 0
            entry_count = 0
            same_tick_count = 0

        elif (low <= stop_price and position_size > 0) or (trigger_stop == 1 and position_size > 0 and previous_position_size > 0):
            trigger_stop = 0
            if stop_price == 0:
                stop_price = open
            fee = (position_size * stop_price) * 0.0004
            loss = position_size * (stop_price - entried_price) - fee
            loss_rate = loss # / current_capital
            biggest_loss_rate = loss_rate if loss_rate < biggest_loss_rate else biggest_loss_rate
            current_capital += loss
            position_size = 0
            entried_price = 0
            target_price = 0
            stop_price = 0
            tick_count = 0
            entry_count = 0
            same_tick_count = 0

            if previous_capital < current_capital:
                draw += 1
            else:
                lose += 1

        if current_capital > 0:
            if lower - close >= tick_size:
                same_tick_count = 0
                tick_count += 1
                lower = close

                if tick_count >= trade_tick and entry_count < divide and ((entry_count == 0 and row['rsi'] < 40) or (entry_count > 0)):
                    if ma_trend > 20 and entry_count == 0:
                        tick_count = 0
                    else:
                        if first_trade_time == 0:
                            first_trade_time = time
                        entry_count += 1
                        newly_size = current_capital / close * leverage * (1/4.0)
                        entried_price = ((entried_price * position_size) + (close * newly_size)) / (position_size + newly_size)
                        fee = newly_size * entried_price * 0.0004
                        current_capital -= fee
                        position_size += newly_size
                        target_price = entried_price + atr * atr_multiplier * profit_percent
                        if loss_percent > 0:
                            stop_price = entried_price - atr * atr_multiplier * loss_percent

            if tick_count > 0:
                if lower - close <= tick_size * -1 * back_size and position_size == 0:
                    tick_count = 0
                    same_tick_count = 0
                if tick_count == previous_tick_count if previous_tick_count else False:
                    same_tick_count += 1
                    if same_tick_count == same_tick and position_size == 0:
                        tick_count = 0
                        same_tick_count = 0
                    if same_tick_count == same_holding and position_size > 0 and previous_position_size > 0:
                        trigger_stop = 1
                else:
                    same_tick_count = 0

        previous_capital = current_capital
        previous_position_size = position_size
        previous_tick_count = tick_count

    pnl = (current_capital - 5000.0)/5000.0 * 100
    print(f"Win: {win}, Lose: {lose}, Draw: {draw}")
    print(f"Win Rate: {win / (win + lose) * 100}%")
    print(f"P&L: {pnl}%")
    # print(f"First Trade Time: {first_trade_time}")
    # print(f"Last Trade Time: {time}")
    # print(f"Total Trade Count: {win + lose + draw}")
    # print(f"biggest_loss_rate: {biggest_loss_rate * 100}%")
    result['result'] = {'win': win, 'lose': lose, 'draw': draw, 'win_rate': win / (win + lose) * 100, 'pnl': pnl, 'first_trade_time': first_trade_time, 'last_trade_time': time, 'total_trade_count': win + lose + draw, 'biggest_loss_rate': biggest_loss_rate * 100}
    return result

def get_param_combinations(step_size):
    profit_percents = np.arange(0.3, 0.71, step_size)
    atr_tick_multipliers = np.arange(0.5, 0.91, step_size)
    trade_ticks = np.arange(3, 4, 1)
    same_ticks = np.arange(4, 13, step_size * 20)
    same_holdings = np.arange(20, 21, 1)
    divides = np.arange(4, 5, 1)
    back_sizes = np.arange(0.5, 1.01, step_size)
    
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
    result = test_trading(result, data)
    return result

def refine_search_space(best_conditions):
    step_size = 0.05
    profit_percents = np.arange(best_conditions['profit_percent'] - step_size, best_conditions['profit_percent'] + step_size + 0.01, step_size)
    atr_tick_multipliers = np.arange(best_conditions['atr_tick_multiplier'] - step_size, best_conditions['atr_tick_multiplier'] + step_size + 0.01, step_size)
    back_sizes = np.arange(best_conditions['back_size'] - step_size, best_conditions['back_size'] + step_size + 0.01, step_size)
    same_ticks = np.arange(best_conditions['same_tick'] - step_size * 20, best_conditions['same_tick'] + step_size * 20 + 0.01, step_size * 20)
    
    return list(product(profit_percents, atr_tick_multipliers, [best_conditions['trade_tick']], same_ticks, [best_conditions['same_holding']], [best_conditions['divide']], back_sizes))

def run_backtest(symbols = None, min_max_step = None):
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
    if symbols is None:
        filenames = glob.glob(public_path + 'klines/*_5m.csv')
        symbols = [filename.split('/')[-1].split('_')[0] for filename in filenames]
    else:
        symbols = symbols.split(',') if isinstance(symbols, str) else symbols
    print("target symbols: {}", symbols)
    
    for symbol in symbols:
        print('## Symbol: {}'.format(symbol))

        data = pd.read_csv(public_path + 'klines/{}_5m.csv'.format(symbol), parse_dates=['timestamp'])
        data = data[['timestamp', 'open', 'high', 'low', 'close', 'atr', 'rsi', 'ema']]

        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
        first_time, last_time = data['timestamp'].iloc[0], data['timestamp'].iloc[-1]
        first_time = pd.to_datetime(first_time).strftime('%Y-%m-%d %H:%M:%S')
        last_time = pd.to_datetime(last_time).strftime('%Y-%m-%d %H:%M:%S')
        print('we are testing {} {} {}'.format(symbol, first_time, last_time))
        
        same_data_df = df.loc[(df['symbol'] == symbol) & (df['data_first_time'] == first_time) & (df['data_last_time'] == last_time)]
        if len(same_data_df) > 0:
            print('## Skip Backtesting... already exists {}', same_data_df)
            continue
        
        results = []

        print('## Start Backtesting...')
        
        # Check to see if there is data of symbol for a win rate of 90% or higher.
        if symbol in df['symbol'].values and df.loc[df['symbol'] == symbol]['win_rate'].max() >= 90:
            first_row = df.loc[df['symbol'] == symbol].iloc[0]
            refined_param_combinations = refine_search_space({
                'profit_percent': first_row['profit_percent'],
                'atr_tick_multiplier': first_row['atr_tick_multiplier'],
                'trade_tick': first_row['trade_tick'],
                'same_tick': first_row['same_tick'],
                'same_holding': first_row['same_holding'],
                'divide': first_row['divide'],
                'back_size': first_row['back_size'],
            })
        else:
            # Step 1: Perform a rough search
            param_combinations = get_param_combinations(step_size=0.2)
            total_step = len(param_combinations)
            print('## Step 1: Perform a rough search, total_step is {}'.format(total_step))

            with ProcessPoolExecutor() as executor:
                backtest_with_data = partial(backtest, data=data)
                results = list(executor.map(backtest_with_data, param_combinations))

            # Select the best PNL conditions among those with a win rate of 90% or more.
            best_condition = max([result for result in results if result['result']['win_rate'] >= 90], key=lambda x: x['result']['pnl'])

            print('## best_condition is {}, and P&L is {}'.format(best_condition['conditions'], best_condition['result']['pnl']))
            
            # Step 2: Refine the search around the best conditions
            refined_param_combinations = refine_search_space(best_condition['conditions'])
        total_step = len(refined_param_combinations)
        print('## Step 2: Refine the search around the best conditions, total_step is {}'.format(total_step))

        with ProcessPoolExecutor() as executor:
            backtest_with_data = partial(backtest, data=data)
            refined_results = list(executor.map(backtest_with_data, refined_param_combinations))
        # Select the best PNL conditions among those with a win rate of 90% or more.
           
        refined_best_condition = max([result for result in refined_results if result['result']['win_rate'] >= 90], key=lambda x: x['result']['pnl'])
        results.extend(refined_results)
        
        # check if best_condition is not none
        if 'best_condition' in locals():
            best_condition = max([best_condition, refined_best_condition], key=lambda x: x['result']['pnl'])
        else:
            best_condition = refined_best_condition

        print('## {} symbol best_condition is {}, and P&L is {}'.format(symbol, best_condition['conditions'], best_condition['result']['pnl']))
        
        # sort results by P&L
        results = sorted(results, key=lambda x: x['result']['pnl'], reverse=True)

        # get best 3 data
        best = results[:3]
        # find the data of symbol and if exists, update it, else append new one
        for result in best:
            row = {
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
            }
            
            # if there are 3 data of symbol, remove the one of worst P&L
            pnl = result['result']['pnl']
            df = df.append(row, ignore_index=True)
            
        # order df by pnl
        df = df.sort_values(by=['pnl'], ascending=False)
            
        # Save the dataframe to the CSV file
        df.to_csv(public_path + 'result.csv', index=False)

        print("Best trading condition:")
        print(best_condition)

if __name__ == '__main__':
    freeze_support()
    run_backtest(sys.argv[1])
