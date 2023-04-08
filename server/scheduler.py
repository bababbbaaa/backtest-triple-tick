import json
import subprocess
import sys
import schedule
import time

def fetch_chart_data():
    print("Fetching chart data")
    filename = "./scripts/fetch_binance_klines_data.py"
    process = subprocess.Popen(
        ["python", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        text=True,  # This sets 'universal_newlines' to True in Python 3.6 and earlier
    )

    # Stream output in real-time
    with process.stdout:
        for line in iter(process.stdout.readline, ""):
            sys.stdout.write(line)
    with process.stderr:
        for line in iter(process.stderr.readline, ""):
            sys.stderr.write(line)

    process.wait()
    

def backtest(symbol):
    print("Backtesting")
    filename = "./scripts/backtest.py"
    process = subprocess.Popen(
        ["python", filename, symbol],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        text=True,  # This sets 'universal_newlines' to True in Python 3.6 and earlier
    )

    # Stream output in real-time
    with process.stdout:
        for line in iter(process.stdout.readline, ""):
            sys.stdout.write(line)
    with process.stderr:
        for line in iter(process.stderr.readline, ""):
            sys.stderr.write(line)

    process.wait()

def schedule_daily_job():
    
    schedule.every().day.at("09:00").do(fetch_chart_data)
    schedule.every().day.at("09:00").do(backtest, None)

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    schedule_daily_job()
