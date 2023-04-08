# modify_result.csv
# Load result.csv
import os
import pandas as pd
df = pd.read_csv('./public/result.csv')
# add new columns (data_first_time, data_last_time) to df if not exists
if 'data_first_time' not in df.columns:
  df['data_first_time'] = None
if 'data_last_time' not in df.columns:
  df['data_last_time'] = None
# loop for each row of df
for index, row in df.iterrows():
    # check the file of symbol in public/klines/{symbol}_5m.csv exists
    kline_file = './public/klines/{}_5m.csv'.format(row['symbol'])
    # if exists, load the time of first and last row
    if os.path.exists(kline_file):
        kline_df = pd.read_csv(kline_file)
        kline_df['timestamp'] = pd.to_datetime(kline_df['timestamp'], unit='s')
        first_time = kline_df['timestamp'].iloc[0]
        last_time = kline_df['timestamp'].iloc[-1]
        df.at[index, 'data_first_time'] = first_time
        df.at[index, 'data_last_time'] = last_time

# round the value of each columns to 2 decimal places if it is float. Otherwise, keep it as it is
df = df.round(2)

# save df to result.csv
df.to_csv('./public/result.csv', index=False)
