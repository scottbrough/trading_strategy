#!/bin/bash
# Script to run backtesting with proper Python path

# Set the Python path to include the current directory
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Default parameters
STRATEGY="momentum_strategy"
SYMBOLS="BTC/USD"
TIMEFRAME="1h"
START_DATE="2023-01-01"
END_DATE="2023-12-31"
INITIAL_CAPITAL=10000
PLOT=true

# Run the backtest
python -m src.scripts.backtest \
  --strategy ${STRATEGY} \
  --symbols ${SYMBOLS} \
  --timeframe ${TIMEFRAME} \
  --start-date ${START_DATE} \
  --end-date ${END_DATE} \
  --initial-capital ${INITIAL_CAPITAL} \
  --config config/momentum_config.yaml \
  --plot

# Check if backtest was successful
if [ $? -eq 0 ]; then
  echo "Backtest completed successfully!"
  echo "Results should be saved in the results directory."
else
  echo "Backtest failed. Check the error messages above."
fi