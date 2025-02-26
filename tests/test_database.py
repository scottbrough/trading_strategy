"""
Tests for database operations.
"""

import unittest
import pandas as pd
from datetime import datetime, timedelta
from src.data.database import db, OHLCV
from src.core.exceptions import DatabaseError

class TestDatabase(unittest.TestCase):
    def setUp(self):
        """Set up test database"""
        # Use a test database configuration or mock
        pass
        
    def test_store_and_retrieve_trade(self):
        """Test storing and retrieving trades"""
        # Create sample trade
        test_trade = {
            'symbol': 'TEST/USD',
            'side': 'buy',
            'entry_price': 100.0,
            'exit_price': 110.0,
            'amount': 1.0,
            'entry_time': datetime.now() - timedelta(hours=1),
            'exit_time': datetime.now(),
            'pnl': 10.0,
            'status': 'closed',
            'strategy': 'test_strategy'
        }
        
        # Store trade
        db.store_trade(test_trade)
        
        # Retrieve trades
        trades = db.get_trades(symbol='TEST/USD')
        
        # Verify
        self.assertGreater(len(trades), 0)
        self.assertEqual(trades.iloc[-1]['symbol'], 'TEST/USD')
        
    # tests/test_database.py
def test_store_and_retrieve_ohlcv(self):
    """Test storing and retrieving OHLCV data"""
    # Clear previous data for this symbol and timeframe
    try:
        with db.get_session() as session:
            session.query(OHLCV).filter(
                OHLCV.symbol == 'TEST/USD',
                OHLCV.timeframe == '1d'
            ).delete()
            session.commit()
    except:
        pass
    
    # Create sample data with specific timestamps
    now = datetime.now()
    dates = pd.date_range(start=now - timedelta(days=5), periods=5, freq='1d')
    data = {
        'open': [100, 102, 101, 103, 105],
        'high': [105, 107, 104, 108, 110],
        'low': [98, 100, 99, 101, 103],
        'close': [102, 101, 103, 105, 107],
        'volume': [1000, 1100, 900, 1200, 1300]
    }
    df = pd.DataFrame(data, index=dates)
    
    # Store data
    db.store_ohlcv('TEST/USD', '1d', df)
    
    # Retrieve data - use exact same date range
    retrieved = db.get_ohlcv('TEST/USD', '1d', dates[0], dates[-1])
    
    # Verify
    self.assertEqual(len(retrieved), 5)
    self.assertAlmostEqual(retrieved['close'].iloc[-1], 107.0)