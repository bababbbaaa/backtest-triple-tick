# fetch result.csv
# result csv is like this
# symbol,pnl,win_rate,profit_percent,atr_tick_multiplier,trade_tick,same_tick,same_holding,divide,back_size,backtesting_datetime,data_first_time,data_last_time
# ACHUSDT,1042.14,90.86,0.75,0.6,3,4.0,20,4,0.8,2023-04-09 10:05:44,2023-02-08 00:00:00,2023-04-10 00:00:00
# ACHUSDT,992.4,90.86,0.75,0.6,3,4.0,20,4,0.75,2023-04-09 10:05:44,2023-02-08 00:00:00,2023-04-10 00:00:00
# ACHUSDT,876.11,90.76,0.75,0.6,3,4.0,20,4,0.7,2023-04-09 10:05:44,2023-02-08 00:00:00,2023-04-10 00:00:00
# ANTUSDT,76.16,94.34,0.75,0.7,3,2.0,20,4,0.85,2023-04-09 00:33:48,2023-02-08 00:00:00,2023-04-08 14:35:00
# ANTUSDT,76.16,94.34,0.75,0.7,3,2.0,20,4,0.9,2023-04-09 00:33:48,2023-02-08 00:00:00,2023-04-08 14:35:00
# ANTUSDT,76.16,94.34,0.75,0.7,3,2.0,20,4,0.95,2023-04-09 00:33:48,2023-02-08 00:00:00,2023-04-08 14:35:00
# ARUSDT,209.79,92.0,0.75,0.5,3,3.0,20,4,0.5,2023-04-09 09:56:51,2023-02-08 00:00:00,2023-04-10 00:00:00

import pandas as pd


df = pd.read_csv('../public/result.csv')

print("min, max, avg of each column")
for col in df.columns:
    if col == 'symbol' or col == 'backtesting_datetime' or col == 'data_first_time' or col == 'data_last_time':
        continue
    print(col, df[col].min(), df[col].max(), df[col].mean())