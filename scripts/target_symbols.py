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
# "계속적으로 가격 변동이 큰 심볼들"의 의미를 갖는 함수 이름을 작성합니다. -> symbols_with_high_price_changes
async def symbols_with_high_price_changes():
    # Binance Futures의 모든 ticker를 가져오고, 일별 가격 변동률을 최근 30일까지 반복하여 계산합니다.
    # e.g. 2021-01-01 ~ 2021-01-30
    # 2021-01-01 ~ 2021-01-02: 10%, 2021-01-02 ~ 2021-01-03: 20%, ..., 2021-01-29 ~ 2021-01-30: 100% -> sum: 1550%
    symbols = await binance_futures.fetch_tickers()
    
    # USDT와 paired된 모든 심볼을 가져옵니다.
    symbols = [x[0] for x in symbols.items() if x[0].endswith('USDT')]
    
    symbol_changes = {}
    for symbol in symbols:
        # 30일간의 4h 데이터를 가져옵니다.
        klines = await binance_futures.fetch_ohlcv(symbol, '4h', limit=1000)
        # klines가 1000개 미만인 경우, 30일간의 데이터가 없다는 뜻이므로, 다음 심볼로 넘어갑니다.
        if len(klines) < 1000:
            continue
        # 일봉 데이터를 pandas DataFrame으로 변환합니다.
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # 일봉 데이터의 가격 변동률을 계산하고, 절대값을 가져옵니다. (pct_change가 하는 일을 간단하게 설명하면, 1일차의 종가가 1000, 2일차의 종가가 1100, 3일차의 종가가 1200이라면, df['close'].pct_change()는 [0, 0.1, 0.1]을 반환합니다.)
        changes = df['close'].pct_change().abs()
        # 일별 가격 변동률의 평균을 계산합니다. (symbol_changes[symbol] = [0, 0.1, 0.2, 0.3, 0.4]라면, symbol_changes[symbol].mean()은 0.2를 반환합니다.)
        symbol_changes[symbol] = changes.mean()
        print(symbol, symbol_changes[symbol])
    # 가격 변동률이 높은 순서대로 정렬합니다.
    symbol_changes = sorted(symbol_changes.items(), key=lambda x: x[1], reverse=True)
    
    # 가격 변동률이 높은 상위 20개의 심볼과 가격 변동률을 출력합니다.
    return symbol_changes[:20]

async def main():
    symbols = await symbols_with_high_price_changes()
    print("Symbols with high price changes:")
    for symbol in symbols:
        print(symbol[0], symbol[1])
    await binance_futures.close()

asyncio.run(main())
