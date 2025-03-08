#!/bin/bash
# start_trading.sh - Script to properly start trading system in background

# Configuration
PROJECT_DIR="$(pwd)"
LOG_DIR="$PROJECT_DIR/logs"
MODE="paper"  # paper, backtest, or full

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Kill any existing processes
if [ -f "$PROJECT_DIR/process_ids.txt" ]; then
    echo "Stopping existing processes..."
    python -m src.scripts.shutdown
    sleep 3
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "Starting trading system in $MODE mode..."

# Start data streaming component in background
echo "Starting data stream..."
python -m src.data.stream > "$LOG_DIR/stream.log" 2>&1 &
STREAM_PID=$!
echo "Stream process started with PID: $STREAM_PID"

# Wait for stream to initialize
sleep 2

# Start trading system in background
echo "Starting trading system..."
python -m src.scripts.run --$MODE --strategy momentum_strategy --config config/momentum_strategy.yaml > "$LOG_DIR/trading.log" 2>&1 &
TRADING_PID=$!
echo "Trading process started with PID: $TRADING_PID"

# Wait for trading system to initialize
sleep 2

# Start dashboard in background
echo "Starting monitoring dashboard..."
python -m src.monitoring.dashboard --host 0.0.0.0 > "$LOG_DIR/dashboard.log" 2>&1 &
DASHBOARD_PID=$!
echo "Dashboard process started with PID: $DASHBOARD_PID"

# Save process IDs for later management
echo "stream_pid=$STREAM_PID" > "$PROJECT_DIR/process_ids.txt"
echo "trading_pid=$TRADING_PID" >> "$PROJECT_DIR/process_ids.txt" 
echo "dashboard_pid=$DASHBOARD_PID" >> "$PROJECT_DIR/process_ids.txt"

echo "Trading system deployment complete!"
echo "Dashboard should be available at: http://localhost:8050"
echo "To stop the system: python -m src.scripts.shutdown"
echo "To view logs: tail -f $LOG_DIR/*.log"