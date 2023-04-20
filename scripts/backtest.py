import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from backtest_helpers import run_backtest

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please provide both 'symbols' and 'min_max_step' arguments.")
        sys.exit(1)

    symbols = sys.argv[1]
    
    # parse step_size from string to float
    step_size = float(sys.argv[2]) if len(sys.argv) > 2 else None
    first_time = sys.argv[3] if len(sys.argv) > 3 else None
    last_time = sys.argv[4] if len(sys.argv) > 4 else None

    run_backtest(symbols=symbols, step_size=step_size, first_time=first_time, last_time=last_time)
