#!/usr/bin/env python
"""
Script to download historical OHLCV data from Kraken and store it in the database.
"""

import os
import sys
from pathlib import Path
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import database
from src.data.database import db
from src.core.logger import log_manager

logger = log_manager.get_logger(__name__)

def fetch_historical_data(exchange, symbol, timeframe, since, limit=1000):
    """
    Fetch historical OHLCV data from exchange
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading pair (e.g., 'BTC/USD')
        timeframe: Candle timeframe (e.g., '1h')
        since: Start timestamp in milliseconds
        limit: Maximum number of candles to fetch
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        # Fetch OHLCV data
        candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        
        # Convert to DataFrame
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

def main():
    # Configure parameters
    symbol = 'BTC/USD'
    timeframe = '1h'
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # Initialize exchange
    exchange = ccxt.kraken({
        'enableRateLimit': True,  # required by the Manual
    })
    
    logger.info(f"Downloading historical data for {symbol} {timeframe} from {start_date} to {end_date}")
    
    # Convert start_date to milliseconds
    since = int(start_date.timestamp() * 1000)
    end_timestamp = int(end_date.timestamp() * 1000)
    
    # Calculate maximum number of candles
    timeframe_ms = {
        '1m': 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000
    }
    
    # Get timeframe in milliseconds
    tf_ms = timeframe_ms.get(timeframe, 60 * 60 * 1000)  # Default to 1h
    
    # Calculate number of candles needed (time range / timeframe)
    period_ms = end_timestamp - since
    candles_needed = period_ms // tf_ms
    
    logger.info(f"Need to fetch approximately {candles_needed} candles")
    
    # Initialize variables for pagination
    current_since = since
    all_data = pd.DataFrame()
    
    # Maximum candles per request
    limit = 1000
    
    # Fetch data in batches
    while current_since < end_timestamp:
        # Fetch a batch of candles
        logger.info(f"Fetching data from {datetime.fromtimestamp(current_since/1000)}")
        df = fetch_historical_data(exchange, symbol, timeframe, current_since, limit)
        
        if df.empty:
            logger.warning("No data returned, moving to next batch")
            # Move to next batch even if no data
            current_since += limit * tf_ms
            continue
        
        # Append to all data
        all_data = pd.concat([all_data, df])
        
        # Update since for next batch (use last timestamp + timeframe)
        if len(df) > 0:
            last_timestamp = df.index[-1].timestamp() * 1000
            current_since = int(last_timestamp + tf_ms)
        else:
            # If no data returned, move to next batch
            current_since += limit * tf_ms
        
        # Remove duplicates
        all_data = all_data[~all_data.index.duplicated(keep='first')]
        
        logger.info(f"Total candles fetched so far: {len(all_data)}")
        
        # Rate limit to avoid overloading the API
        time.sleep(exchange.rateLimit / 1000)  # Convert milliseconds to seconds
    
    # Filter data to exactly match the requested date range
    all_data = all_data[(all_data.index >= start_date) & (all_data.index <= end_date)]
    
    # Store data in database
    if not all_data.empty:
        logger.info(f"Storing {len(all_data)} candles in database")
        db.store_ohlcv(symbol, timeframe, all_data)
        logger.info("Data storage complete")
    else:
        logger.error("No data to store")

if __name__ == "__main__":
    main()