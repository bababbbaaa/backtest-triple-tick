from functools import partial
from multiprocessing import freeze_support
import os
import sys
import time
import timeit
import numpy as np
import pandas as pd
import glob
from itertools import product
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# Configuration
leverage = 8
loss_percent = 0
atr_multiplier = 1.5
init_balance = 10000.0

# Public directory path
public_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'public')) + '/'

class TradeState:
    def __init__(self, conditions):
        self.initialize_variables(conditions)

    def initialize_variables(self, conditions):
        self.balance = init_balance
        self.ticks_elapsed = self.lower_bound = self.target = self.stop = 0
        self.wins = self.losses = self.draws = 0
        self.position = self.entry_price = 0
        self.same_tick_counter = 0
        self.entry_counter = self.stop_triggered = 0
        self.prev_balance = self.prev_position = self.prev_ticks_elapsed = 0
        self.ma_trend = 0
        self.atr_tick_multiplier = conditions['atr_tick_multiplier']
        self.same_tick = conditions['same_tick']
        self.back_size = conditions['back_size']
        self.profit_percent = conditions['profit_percent']
        self.same_holding = conditions['same_holding']
        self.divide = conditions['divide']
        self.trade_tick = conditions['trade_tick']

    def set_state(self, row):
        self.open = row['open']
        self.high = row['high']
        self.low = row['low']
        self.close = row['close']
        self.atr = row['atr']
        self.ema = row['ema']
        self.tick_size = self.atr_tick_multiplier * self.atr

    def set_prev_state(self):
        self.prev_balance = self.balance
        self.prev_position = self.position
        self.prev_ticks_elapsed = self.ticks_elapsed

    def update_ma_trend(self):
        self.ma_trend = self.ma_trend + 1 if self.ema > self.close else 0

    def reset_variables(self):
        self.position = 0
        self.entry_price = 0
        self.target = 0
        self.stop = 0
        self.ticks_elapsed = 0
        self.entry_counter = 0
        self.same_tick_counter = 0
    def execute_win(self):
        self.wins += 1
        fee = (self.position * self.target) * 0.0004
        profit = self.position * (self.target - self.entry_price) - fee
        self.balance += profit
        self.reset_variables()

    def execute_loss_or_draw(self):
        self.stop_triggered = 0
        if self.stop == 0:
            self.stop = self.open
        fee = (self.position * self.stop) * 0.0004
        loss = self.position * (self.stop - self.entry_price) - fee
        self.balance += loss

        if self.prev_balance < self.balance:
            self.draws += 1
        else:
            self.losses += 1
        self.reset_variables()

    def update_ticks_elapsed(self):
        if self.lower_bound - self.close >= self.tick_size:
            self.same_tick_counter = 0
            self.ticks_elapsed += 1
            self.lower_bound = self.close

            if self.ticks_elapsed >= self.trade_tick and self.entry_counter < self.divide:
                if self.ma_trend > 20 and self.entry_counter == 0:
                    self.ticks_elapsed = 0
                else:
                    self.update_entry_variables()

    def update_entry_variables(self):
        self.entry_counter += 1
        newly_size = self.balance / self.close * leverage * (1/4.0)
        self.entry_price = ((self.entry_price * self.position) + (self.close * newly_size)) / (self.position + newly_size)
        fee = newly_size * self.entry_price * 0.0004
        self.balance -= fee
        self.position += newly_size
        self.target = self.entry_price + self.atr * self.atr_tick_multiplier * self.profit_percent

    def update_same_tick_counter(self):
        if self.ticks_elapsed > 0:
            if self.lower_bound - self.close <= self.tick_size * -1 * self.back_size and self.position == 0:
                self.ticks_elapsed = 0
                self.same_tick_counter = 0
            if self.ticks_elapsed == self.prev_ticks_elapsed if self.prev_ticks_elapsed else False:
                self.same_tick_counter += 1
                self.check_same_tick_conditions()
            else:
                self.same_tick_counter = 0

    def check_same_tick_conditions(self):
        if self.same_tick_counter == self.same_tick and self.position == 0:
            self.ticks_elapsed = 0
            self.same_tick_counter = 0
        if self.same_tick_counter == self.same_holding and self.position > 0 and self.prev_position > 0:
            self.stop_triggered = 1

    def calculate_result(self, result):
        print("Wins: ", self.wins, " Losses: ", self.losses, " Draws: ", self.draws, " Balance: ", self.balance)
        pnl = (self.balance - init_balance)/init_balance * 100
        win_rate = self.wins / (self.wins + self.losses) * 100 if self.wins + self.losses > 0 else 0
        result['result'] = {'win_rate': win_rate, 'pnl': pnl}
        return result
    
def simulate_trades(result, data):
    state = TradeState(result['conditions'])

    for _, row in data.iterrows():
        state.set_state(row)
        state.update_ma_trend()

        if state.ticks_elapsed == 0:
            state.lower_bound = state.open

        if state.high >= state.target and state.position > 0:
            state.execute_win()
        elif (state.low <= state.stop and state.position > 0) or (state.stop_triggered == 1 and state.position > 0 and state.prev_position > 0):
            state.execute_loss_or_draw()
        if state.balance > 0:
            state.update_ticks_elapsed()
            state.update_same_tick_counter()

        state.set_prev_state()

        if state.balance <= 100:
            break

    result = state.calculate_result(result)
    # Tab 간격으로 result['result']와 result['conditions']을 출력한다.
    print("PNL: ", result['result']['pnl'], "\tWIN_RATE: ", result['result']['win_rate'], "\tCONDITIONS: ", result['conditions'].values())
    return result

def get_param_combinations(step_size):
    profit_percents = np.arange(0.5, 0.91, step_size) # if step_size is 0.2, len(profit_percents) = 3 (0.5, 0.7, 0.9)
    atr_tick_multipliers = np.arange(0.5, 0.91, step_size) # if step_size is 0.2, len(atr_tick_multipliers) = 3 (0.5, 0.7, 0.9)
    trade_ticks = np.arange(3, 4, 1)
    same_ticks = np.arange(2, 13, step_size * 20) # if step_size is 0.2, len(same_ticks) = 3 (2, 6, 10)
    same_holdings = np.arange(20, 21, 5) # len(same_holdings) = 2 (15, 20)
    divides = np.arange(2, 5, 2) # len(divides) = 2 (2, 4)
    back_sizes = np.arange(0.5, 1.1, step_size) # if step_size is 0.2, len(back_sizes) = 3 (0.5, 0.7, 0.9)
    # all combinations of the above is 3 * 3 * 1 * 3 * 2 * 2 * 3 = 324
    
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
    elif symbols == 'exists':
        symbols = df['symbol'].unique().tolist()
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
        
        # if data_first_time is later than first_time, then skip
        if first_time is not None and data_first_time > first_time:
            print('## Skip Backtesting... data_first_time is not same as first_time, {} > {}'.format(data_first_time, first_time))
            continue
        else: 
            print('## data_first_time is same as first_time, {} == {}'.format(data_first_time, first_time))
        
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
        if False and symbol in df['symbol'].values and df.loc[df['symbol'] == symbol]['win_rate'].max() >= 90 and df.loc[df['symbol'] == symbol]['profit_percent'].min() >= 0.45:
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
                'data_first_time': data_first_time,
                'data_last_time': data_last_time,
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
