import type { NextPage } from "next";
import { useState, useEffect, SetStateAction } from "react";
import BacktestButton from "../components/BacktestButton";
import DataTable from "../components/DataTable";

const Home: NextPage = () => {
  const [data, setData] = useState<any[]>([]);
  const [symbol, setSymbol] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const handleSymbolChange = (event: { target: { value: SetStateAction<string> } }) => {
    setSymbol(event.target.value);
  };

  // symbols from the API

  const [symbols, setSymbols] = useState([]);

  useEffect(() => {
    fetch("/api/symbols")
      .then((response) => response.json())
      .then((data) => setSymbols(data));
  }, []);

  useEffect(() => {
    fetch("/api/test-results")
      .then((response) => response.json())
      .then((data) => setData(data));
  }, []);

  const runBacktest = (symbol: string) => {
    fetch("/api/run-backtest", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ symbol }),
    }).then((response) => response.json());
  };

  const runFetchChartData = () => {
    fetch("/api/run-fetch-chart-data").then((response) => response.json());
  };

  return (
    <div className="container mx-auto my-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-4">Run Backtest</h1>
        <h2 className="text-xl font-bold mb-4">Fetch Chart Data to get the latest data</h2>
        <button
          className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
          onClick={runFetchChartData}
        >
          Run Fetch Chart Data
        </button>
        <div className="mb-4">
          <h2 className="text-xl font-bold mb-4">Select a symbol to run a backtest</h2>
          <select value={symbol} onChange={handleSymbolChange}>
            {/* Add options for each symbol here */}
            {symbols.map((symbol) => (
              <option key={symbol} value={symbol}>
                {symbol}
              </option>
            ))}
          </select>
          <h2 className="text-xl font-bold mb-4">Select backtest period(YYYY-MM-DD ~ YYYY-MM-DD) by datetime</h2>
          {/* Add date picker here */}
          <input type="text" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
          <input type="text" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
          <h2 className="text-xl font-bold mb-4">Run Backtest</h2>
          <BacktestButton onClick={() => runBacktest(symbol)} />
        </div>
      </div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-4">Backtest Results</h1>
        <div className="shadow overflow-hidden border-b border-gray-200 sm:rounded-lg">
          <DataTable data={data} />
        </div>
      </div>
    </div>
  );
};

export default Home;
