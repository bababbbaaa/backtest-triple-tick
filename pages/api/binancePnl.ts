// This code calculates realized PnL for Binance futures trading pairs over the past month.
// It uses the Binance API to fetch trading pairs and trades, and calculates the realized PnL
// for each trading pair by summing the realized PnL for each trade.

import { NextApiRequest, NextApiResponse } from "next";
import Binance from "node-binance-api";

// Initialize Binance instance with API key and secret
const binance = new Binance().options({
  APIKEY: process.env.API_KEY,
  APISECRET: process.env.API_SECRET,
});

// Define symbolPnl type
type symbolPnl = {
  symbol: string;
  pnl: number;
  trade_count: number[];
  profit_loss: number[];
};

// Next.js API handler function
export default async function handler(req: NextApiRequest, res: NextApiResponse<symbolPnl[]>) {
  const symbols = [
    "QTUMUSDT",
    "CELRUSDT",
    "OPUSDT",
    "MASKUSDT",
    "SANDUSDT",
    "COMPUSDT",
    "SNXUSDT",
    "ARUSDT",
    "GALAUSDT",
    "FLOWUSDT",
  ];
  console.log(`Fetching trades for ${symbols.length} symbols...: ${symbols}`);

  // Calculate time range from 2023-03-18 to now (in milliseconds)

  // Initialize array for PnL data
  const pnls: symbolPnl[] = [];

  // Loop through symbols and calculate realized PnL for each
  for (const symbol of symbols) {
    console.log(`Fetching trades for symbol ${symbol}...`);
    let startTime = new Date("2023-03-18").getTime();
    const endTime = new Date().getTime();
    console.log(`Fetching trades between ${new Date(startTime)} and ${new Date(endTime)}`);
    try {
      let trades: any = [];
      const limit = 1000;
      while (true) {
        // Fetch trades for symbol
        const tradesResponse = await binance.futuresUserTrades(symbol, { limit, startTime });
        if (!tradesResponse || tradesResponse.length === 0) {
          break;
        }
        console.log(
          `call futuresUserTrades(${symbol}, { limit: ${limit}, startTime: ${startTime} }) -> ${tradesResponse.length} trades`
        );
        // Calculate trades[0].time and trades[trades.length - 1].time (in milliseconds) and print to console
        console.log(
          `Fetched. ${new Date(tradesResponse[0].time)} ~ ${new Date(tradesResponse[tradesResponse.length - 1].time)}`
        );
        // conver tradesResponse.realizedPnl to number
        for (const trade of tradesResponse) {
          trade.realizedPnl = Number(trade.realizedPnl);
        }
        // Add tradesResponse to trades
        trades = [...trades, ...tradesResponse];

        // Add the following condition to check if the tradesResponse array is empty before accessing its elements.
        if (tradesResponse && tradesResponse.length > 0) {
          if (tradesResponse[0].time === startTime && tradesResponse[tradesResponse.length - 1].time === startTime) {
            break;
          }
          // if end of trades data is not 'now', fetch more trades from the end of the last batch as the start time
          startTime = tradesResponse[tradesResponse.length - 1].time;
        }
      }
      console.log(`Fetched ${trades.length} trades for symbol ${symbol}.`);
      const orderPnls: { orderId: any; pnl: any }[] = [];
      if (Array.isArray(trades) && trades.length > 0) {
        for (const trade of trades) {
          const orderPnl = orderPnls.find((orderPnl) => orderPnl.orderId === trade.orderId);
          if (orderPnl) {
            orderPnl.pnl += trade.realizedPnl;
          } else {
            orderPnls.push({ orderId: trade.orderId, pnl: trade.realizedPnl });
          }
        }
      }

      // orderPnls의 pnl을 모두 더해서 pnls에 추가. pnl이 0보다 크면 trade_count[0]에 1을 더하고, 0보다 작으면 trade_count[1]에 1을 더한다.
      const pnl = orderPnls.reduce((acc, cur) => acc + cur.pnl, 0);
      const trade_count = [0, 0];
      const profit_loss = [0, 0];
      for (const orderPnl of orderPnls) {
        if (orderPnl.pnl > 0) {
          trade_count[0] += 1;
          profit_loss[0] += orderPnl.pnl;
        } else {
          trade_count[1] += 1;
          profit_loss[1] += orderPnl.pnl;
        }
      }
      pnls.push({ symbol, pnl, trade_count, profit_loss });
    } catch (error: any) {
      console.error(`Error fetching trades for symbol ${symbol}:`, error);
    }
  }

  // Send PnL data as JSON response
  res.status(200).json(pnls);
}
