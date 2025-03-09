#!/usr/bin/env python
"""
Simple script to directly download and store 2023 BTC/USD historical data from KuCoin.
"""

import sys
from pathlib import Path
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import database
from src.data.database import db
from src.core.logger import log_manager

logger = log_manager.get_logger(__name__)

def main():
    """Download and store historical data from KuCoin"""
    # Set up parameters
    symbol = 'BTC/USD'  # Target symbol for database
    kucoin_symbol = 'BTC/USDT'  # Symbol to use with KuCoin
    timeframe = '1h'
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31, 23, 59, 59)
    
    try:
        logger.info(f"Initializing KuCoin exchange")
        exchange = ccxt.kucoin({
            'enableRateLimit': True,
        })
        
        # Load markets to confirm symbol availability
        markets = exchange.load_markets()
        if kucoin_symbol not in markets:
            logger.error(f"Symbol {kucoin_symbol} not found on KuCoin")
            return False
        
        logger.info(f"Downloading data for {kucoin_symbol} ({timeframe}) from {start_date} to {end_date}")
        
        # Download data in chunks to respect exchange limits
        all_data = []
        
        # Convert start/end dates to milliseconds
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)
        current_ts = start_ts
        
        # Max candles per request for KuCoin
        limit = 1500
        
        # Calculate how many hours in the entire period
        total_hours = int((end_ts - start_ts) / (3600 * 1000)) + 1
        logger.info(f"Need to fetch approximately {total_hours} hours of data")
        
        # Calculate approx. number of batches needed
        num_batches = (total_hours + limit - 1) // limit
        logger.info(f"Will download in approximately {num_batches} batches")
        
        # Track progress
        batch_number = 0
        candles_collected = 0
        
        # Download in batches
        while current_ts < end_ts:
            batch_number += 1
            logger.info(f"Downloading batch {batch_number}/{num_batches} from {datetime.fromtimestamp(current_ts/1000)}")
            
            try:
                # Download candles
                candles = exchange.fetch_ohlcv(kucoin_symbol, timeframe, current_ts, limit)
                
                if not candles or len(candles) == 0:
                    logger.warning(f"No data returned for batch {batch_number}")
                    # Move forward in time even if no data returned
                    current_ts += 3600 * 1000 * limit
                    continue
                
                logger.info(f"Received {len(candles)} candles")
                
                # Log the time range received
                first_ts = candles[0][0]
                last_ts = candles[-1][0]
                logger.info(f"Data from {datetime.fromtimestamp(first_ts/1000)} to {datetime.fromtimestamp(last_ts/1000)}")
                
                # Add candles to our collection
                all_data.extend(candles)
                candles_collected += len(candles)
                
                # Update current timestamp for next batch
                current_ts = last_ts + 3600 * 1000  # Move to next hour after last received candle
                
                logger.info(f"Total candles collected: {candles_collected}")
                
                # Sleep to respect rate limits
                time.sleep(exchange.rateLimit / 1000 * 1.5)  # Add 50% extra wait time
                
            except Exception as e:
                logger.error(f"Error fetching batch {batch_number}: {str(e)}")
                time.sleep(10)  # Longer sleep on error
                continue
        
        # Convert to DataFrame
        if not all_data:
            logger.error("No data collected!")
            return False
            
        logger.info(f"Creating DataFrame from {len(all_data)} candles")
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Remove any duplicates
        old_len = len(df)
        df = df[~df.index.duplicated(keep='first')]
        logger.info(f"Removed {old_len - len(df)} duplicate entries")
        
        # Filter to only include 2023 data
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        logger.info(f"After filtering for 2023: {len(df)} candles")
        
        # Save to CSV for backup
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        csv_file = data_dir / f"BTC_USD_1h_2023.csv"
        df.to_csv(csv_file)
        logger.info(f"Saved data to CSV: {csv_file}")
        
        # Log data range
        if not df.empty:
            logger.info(f"Data spans from {df.index.min()} to {df.index.max()}")
            logger.info(f"Number of hours: {len(df)}")
            
            # Store in database
            logger.info(f"Storing {len(df)} candles in database for {symbol} {timeframe}")
            db.store_ohlcv(symbol, timeframe, df)
            logger.info("Database storage complete")
            
            return True
        else:
            logger.error("No data to store after filtering")
            return False
            
    except Exception as e:
        logger.error(f"Error downloading historical data: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("Historical data download completed successfully")
    else:
        logger.error("Failed to download historical data")