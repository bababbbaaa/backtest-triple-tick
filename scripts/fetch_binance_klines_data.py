import ccxt.async_support as ccxt
import asyncio
import time
import os
import pandas as pd
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator

interval = '5m'
limit = 1000


# abspath of '../public'
public_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'public')) + '/'

binance_futures = ccxt.binance({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future'
    }
})

async def fetch_klines(symbol, since):
    klines = await binance_futures.fetch_ohlcv(symbol, interval, limit=limit, since=since)
    # sleep 0.2 second to avoid rate limit
    return klines

async def fetch_popular_symbols():
    # Fetch Top 10 symbols by price change in the last 24 hours
    symbols = await binance_futures.fetch_tickers()
    symbols = sorted(symbols.items(), key=lambda x: x[1]['percentage'], reverse=True)
    print("Popular symbols:")
    for symbol in symbols[:10]:
        print(symbol[0], symbol[1]['percentage'])
    symbols = [x[0] for x in symbols]
    symbols = [s.replace('/USDT:', '') for s in symbols if s.endswith('USDT')][:10]
    return symbols

def concat_to(df, klines):
    klines = [[int(x[0]/1000), x[1], x[2], x[3], x[4], x[5]] for x in klines]
    # use pandas.concat instead of df.append
    df = pd.concat([df, pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])])
    df = df.drop_duplicates(subset=['timestamp'])
    return df

async def main():
    symbols = await fetch_popular_symbols()
    current_symbols = [x.split('_')[0] for x in os.listdir(public_path + 'klines') if x.endswith('.csv')]
    if len(current_symbols) == 0:
        print("No existing data, fetching all symbols")
        return
    # combine symbols and current_symbols and remove duplicates
    symbols = list(set(symbols + current_symbols))
    processed_symbol_count = 0
    print(symbols)
    for symbol in symbols:
        csv_file = public_path + 'klines/{}_{}.csv'.format(symbol, interval)
        print("Loading", csv_file)
        if os.path.exists(csv_file):
            print("File exists, loading from file")
            df = pd.read_csv(csv_file)
            df = df.sort_values(by=['timestamp'], ascending=True)
            print("Loaded", len(df), "rows", "first:", pd.to_datetime(df['timestamp'].iloc[0], unit='s'), "last:", pd.to_datetime(df['timestamp'].iloc[-1], unit='s'))
            last_timestamp = df['timestamp'].iloc[-1] * 1000
            while time.time() * 1000 - last_timestamp >= 5 * 60 * 1000:
                print("File exists, but not up to date, loading from binance. Last timestamp:", pd.to_datetime(last_timestamp, unit='ms'), "Now:", pd.to_datetime(time.time(), unit='s'))
                # Fetch klines from last_timestamp to now
                klines = await fetch_klines(symbol, last_timestamp)
                print("Fetched", len(klines), "klines", "first klines timestamp:", pd.to_datetime(klines[0][0], unit='ms'), "last klines timestamp:", pd.to_datetime(klines[-1][0], unit='ms'))
                df = concat_to(df, klines)
                df = df.sort_values(by=['timestamp'], ascending=True)
                last_timestamp = df['timestamp'].iloc[-1] * 1000
        else:
            print("File not exists, loading from binance")
            df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            last_timestamp = None
        df = df.sort_values(by=['timestamp'], ascending=False)
        while len(df) < 50000:
            if len(df) > 0:
                print("current df length:", len(df), "first timestamp:", pd.to_datetime(df['timestamp'].iloc[0], unit='s'), "last timestamp:", pd.to_datetime(df['timestamp'].iloc[-1], unit='s'))
            new_since = last_timestamp - (limit * 300 * 1000) if last_timestamp else None
            klines = await fetch_klines(symbol, new_since)
            print("Fetched", len(klines), "klines", "first klines timestamp:", klines[0][0], "last klines timestamp:", klines[-1][0])
            df = concat_to(df, klines)
            df = df.sort_values(by=['timestamp'], ascending=False)
            if last_timestamp == klines[0][0]:
                break
            last_timestamp = klines[0][0]

        print("Done loading", csv_file)
        df = df.sort_values(by=['timestamp'], ascending=True)
        
        # Calculate RSI (source = close, len = 14)
        rsi_indicator = RSIIndicator(close=df["close"], window=14)
        df["rsi"] = rsi_indicator.rsi()

        # Calculate ATR (len = 14)
        atr_indicator = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
        df["atr"] = atr_indicator.average_true_range()
        
        # Calculate EMA (len = 20)
        df["ema"] = df["close"].ewm(span=20, adjust=False).mean()
        df.to_csv(csv_file, index=False)
        # print top 3 rows and bottom 3 rows
        print(df.head(3))
        print(df.tail(3))
        processed_symbol_count += 1
    await binance_futures.close()

asyncio.run(main())
