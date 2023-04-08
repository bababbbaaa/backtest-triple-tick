import { spawn } from "child_process";
import schedule from "node-schedule";

const fetchChartData = () => {
  const process = spawn("python", ["./scripts/fetch_chart_data.py"]);
  process.stdout.on("data", (data) => console.log(data.toString()));
  process.stderr.on("data", (data) => console.error(data.toString()));
};

const backtest = () => {
  const process = spawn("python", ["./scripts/backtest.py"]);
  process.stdout.on("data", (data) => console.log(data.toString()));
  process.stderr.on("data", (data) => console.error(data.toString()));
};

schedule.scheduleJob("0 9 * * *", () => {
  fetchChartData();
  backtest();
});

export { fetchChartData, backtest };
