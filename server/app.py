from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from .scheduler import fetch_chart_data, backtest
import csv
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/test-results")
async def get_test_results():
    file_path = "public/result.csv"
    data = []
    with open(file_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Round all float values by 2 digits
            for key, value in row.items():
                try:
                    row[key] = round(float(value), 2)
                except:
                    pass
            data.append(row)
    return JSONResponse(content=data)

class BacktestPayload(BaseModel):
    symbol: str

@app.post("/run-backtest")
async def run_backtest_endpoint(payload: BacktestPayload):
    backtest(payload.symbol)
    return {"status": "success"}

@app.get("/symbols")
def get_symbols():
    # Get all symbols if public/klines/{symbol}_5m.csv exists
    symbols = []
    if os.path.exists("public/klines"):
        for file in os.listdir("public/klines"):
            if file.endswith("_5m.csv"):
                symbols.append(file.split("_")[0])
    return JSONResponse(content=symbols)

@app.get("/run-fetch-chart-data")
def run_fetch_chart_data():
    fetch_chart_data()
    return {"status": "success"}
