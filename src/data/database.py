"""
Database integration module using SQLAlchemy for the trading system.
Handles all database operations and provides connection pooling.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Boolean, Index, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError
from contextlib import contextmanager
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from typing import Generator, List, Dict, Any, Optional, Tuple

from ..core.config import config
from ..core.logger import log_manager
from ..core.exceptions import DatabaseError

logger = log_manager.get_logger(__name__)
Base = declarative_base()

class Trade(Base):
    """Trade record table"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    side = Column(String(4), nullable=False)  # buy/sell
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    amount = Column(Float, nullable=False)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime)
    pnl = Column(Float)
    status = Column(String(10), nullable=False)  # open/closed
    strategy = Column(String(50))
    parameters = Column(JSON)
    
    # Add indexes for better query performance
    __table_args__ = (
        Index('ix_trades_symbol', 'symbol'),
        Index('ix_trades_entry_time', 'entry_time'),
        Index('ix_trades_status', 'status'),
    )

class OHLCV(Base):
    """Price and volume data table"""
    __tablename__ = 'ohlcv'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    timeframe = Column(String(3), nullable=False)  # 1m, 5m, 1h, etc.
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    # Add composite index for efficient queries
    __table_args__ = (
        Index('ix_ohlcv_symbol_tf_ts', 'symbol', 'timeframe', 'timestamp'),
    )

class Performance(Base):
    """Performance metrics table"""
    __tablename__ = 'performance'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    equity = Column(Float, nullable=False)
    realized_pnl = Column(Float, nullable=False)
    unrealized_pnl = Column(Float)
    total_trades = Column(Integer)
    win_rate = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    volatility = Column(Float)
    
    # Add index for time-series queries
    __table_args__ = (
        Index('ix_performance_timestamp', 'timestamp'),
    )

def create_db_engine(db_config):
    """Create SQLAlchemy engine with optimized connection pooling"""
    db_url = f"postgresql://{db_config.user}:{db_config.password}@{db_config.host}:{db_config.port}/{db_config.name}"
    
    return create_engine(
        db_url,
        poolclass=QueuePool,
        pool_size=10,  # Number of connections to keep open
        max_overflow=20,  # Maximum overflow beyond pool_size
        pool_timeout=30,  # Timeout to get a connection from the pool
        pool_recycle=1800,  # Recycle connections after 30 minutes
        pool_pre_ping=True,  # Check connection validity before using
        connect_args={"connect_timeout": 10}  # Connection timeout
    )

class DatabaseManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.setup_database()
    
    def setup_database(self):
        """Initialize database connection and create tables"""
        try:
            self.engine = create_db_engine(config.db_config)
            
            # Create session factory
            session_factory = sessionmaker(bind=self.engine)
            self.Session = scoped_session(session_factory)
            
            # Create tables
            Base.metadata.create_all(self.engine)
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise DatabaseError("Failed to initialize database", details={'error': str(e)})
    
    @contextmanager
    def get_session(self) -> Generator:
        """Get a database session with automatic cleanup"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise DatabaseError("Database operation failed", details={'error': str(e)})
        finally:
            session.close()
    
    @contextmanager
    def get_session_with_retry(self, max_retries=3, backoff_factor=2) -> Generator:
        """Get a database session with retries and exponential backoff"""
        session = self.Session()
        retries = 0
        
        while True:
            try:
                yield session
                session.commit()
                break
            except OperationalError as e:
                if retries >= max_retries:
                    logger.error(f"Max retries reached for database operation: {str(e)}")
                    session.rollback()
                    raise DatabaseError("Database operation failed after retries", details={'error': str(e)})
                
                retries += 1
                wait_time = backoff_factor ** retries
                logger.warning(f"Database operation failed, retrying in {wait_time}s: {str(e)}")
                time.sleep(wait_time)
                session.rollback()
            except IntegrityError as e:
                session.rollback()
                logger.error(f"Database integrity error: {str(e)}")
                raise DatabaseError("Database integrity error", details={'error': str(e)})
            except Exception as e:
                session.rollback()
                logger.error(f"Database operation failed: {str(e)}")
                raise DatabaseError("Database operation failed", details={'error': str(e)})
            finally:
                if retries >= max_retries:
                    session.close()
    
    def store_trade(self, trade_data: dict) -> None:
        """Store a trade record"""
        try:
            with self.get_session() as session:
                trade = Trade(**trade_data)
                session.add(trade)
                logger.info(f"Stored trade record for {trade_data['symbol']}")
        except Exception as e:
            logger.error(f"Failed to store trade: {str(e)}")
            raise DatabaseError("Failed to store trade", details={'trade_data': trade_data})
    
    def store_ohlcv(self, symbol: str, timeframe: str, data: pd.DataFrame) -> None:
        """Store OHLCV data"""
        try:
            records = []
            for _, row in data.iterrows():
                record = OHLCV(
                    symbol=symbol,
                    timestamp=row.name,
                    timeframe=timeframe,
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume']
                )
                records.append(record)
            
            with self.get_session() as session:
                session.bulk_save_objects(records)
                logger.info(f"Stored {len(records)} OHLCV records for {symbol} {timeframe}")
        except Exception as e:
            logger.error(f"Failed to store OHLCV data: {str(e)}")
            raise DatabaseError("Failed to store OHLCV data")
    
    def bulk_store_ohlcv(self, symbol: str, timeframe: str, data: pd.DataFrame) -> int:
        """
        Efficiently store multiple OHLCV records in bulk
        Returns the number of records stored
        """
        try:
            # Convert DataFrame to list of dictionaries
            records = []
            for idx, row in data.iterrows():
                record = {
                    'symbol': symbol,
                    'timestamp': idx,
                    'timeframe': timeframe,
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                }
                records.append(record)
            
            # Use SQLAlchemy Core for bulk inserts
            if records:
                with self.engine.connect() as conn:
                    result = conn.execute(OHLCV.__table__.insert(), records)
                    conn.commit()
                    
                    count = len(records)
                    logger.info(f"Bulk stored {count} OHLCV records for {symbol} {timeframe}")
                    return count
            return 0
        except Exception as e:
            logger.error(f"Failed to bulk store OHLCV data: {str(e)}")
            raise DatabaseError("Failed to bulk store OHLCV data")
    
    def store_performance(self, metrics: dict) -> None:
        """Store performance metrics"""
        try:
            with self.get_session() as session:
                perf = Performance(
                    timestamp=datetime.utcnow(),
                    **metrics
                )
                session.add(perf)
                logger.info("Stored performance metrics")
        except Exception as e:
            logger.error(f"Failed to store performance metrics: {str(e)}")
            raise DatabaseError("Failed to store performance metrics")
    
    def get_trades(self, symbol: str = None, start_time: datetime = None,
                  end_time: datetime = None, status: str = None) -> pd.DataFrame:
        """Retrieve trade records"""
        try:
            with self.get_session() as session:
                query = session.query(Trade)
                
                if symbol:
                    query = query.filter(Trade.symbol == symbol)
                if start_time:
                    query = query.filter(Trade.entry_time >= start_time)
                if end_time:
                    query = query.filter(Trade.entry_time <= end_time)
                if status:
                    query = query.filter(Trade.status == status)
                
                trades = query.all()
                
                # Convert to DataFrame
                trade_data = []
                for trade in trades:
                    trade_dict = {
                        'symbol': trade.symbol,
                        'side': trade.side,
                        'entry_price': trade.entry_price,
                        'exit_price': trade.exit_price,
                        'amount': trade.amount,
                        'entry_time': trade.entry_time,
                        'exit_time': trade.exit_time,
                        'pnl': trade.pnl,
                        'status': trade.status,
                        'strategy': trade.strategy,
                        'parameters': trade.parameters
                    }
                    trade_data.append(trade_dict)
                
                return pd.DataFrame(trade_data)
        except Exception as e:
            logger.error(f"Failed to retrieve trades: {str(e)}")
            raise DatabaseError("Failed to retrieve trades")
    
    def get_ohlcv(self, symbol: str, timeframe: str, start_time: datetime,
                  end_time: datetime) -> pd.DataFrame:
        """Retrieve OHLCV data"""
        try:
            with self.get_session() as session:
                query = session.query(OHLCV).filter(
                    OHLCV.symbol == symbol,
                    OHLCV.timeframe == timeframe,
                    OHLCV.timestamp >= start_time,
                    OHLCV.timestamp <= end_time
                ).order_by(OHLCV.timestamp)
                
                records = query.all()
                
                # Convert to DataFrame
                data = {
                    'open': [],
                    'high': [],
                    'low': [],
                    'close': [],
                    'volume': []
                }
                timestamps = []
                
                for record in records:
                    timestamps.append(record.timestamp)
                    data['open'].append(record.open)
                    data['high'].append(record.high)
                    data['low'].append(record.low)
                    data['close'].append(record.close)
                    data['volume'].append(record.volume)
                
                df = pd.DataFrame(data, index=timestamps)
                return df
        except Exception as e:
            logger.error(f"Failed to retrieve OHLCV data: {str(e)}")
            raise DatabaseError("Failed to retrieve OHLCV data")
    
    def get_performance_metrics(self, start_time: datetime = None,
                              end_time: datetime = None) -> pd.DataFrame:
        """Retrieve performance metrics"""
        try:
            with self.get_session() as session:
                query = session.query(Performance)
                
                if start_time:
                    query = query.filter(Performance.timestamp >= start_time)
                if end_time:
                    query = query.filter(Performance.timestamp <= end_time)
                
                records = query.all()
                
                # Convert to DataFrame
                data = []
                for record in records:
                    data.append({
                        'timestamp': record.timestamp,
                        'equity': record.equity,
                        'realized_pnl': record.realized_pnl,
                        'unrealized_pnl': record.unrealized_pnl,
                        'total_trades': record.total_trades,
                        'win_rate': record.win_rate,
                        'sharpe_ratio': record.sharpe_ratio,
                        'sortino_ratio': record.sortino_ratio,
                        'max_drawdown': record.max_drawdown,
                        'volatility': record.volatility
                    })
                
                return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Failed to retrieve performance metrics: {str(e)}")
            raise DatabaseError("Failed to retrieve performance metrics")
            
    def check_database_health(self) -> Dict[str, Any]:
        """Check database connection health and return status"""
        health_data = {
            "status": "healthy",
            "response_time_ms": 0,
            "connection_errors": 0,
            "table_counts": {},
            "last_error": None
        }
        
        try:
            start_time = time.time()
            
            # Test query performance
            with self.get_session() as session:
                # Check each table count
                tables = ['trades', 'ohlcv', 'performance']
                for table in tables:
                    count = session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                    health_data["table_counts"][table] = count
            
            # Calculate response time
            health_data["response_time_ms"] = int((time.time() - start_time) * 1000)
            
            return health_data
        except Exception as e:
            health_data["status"] = "unhealthy"
            health_data["last_error"] = str(e)
            logger.error(f"Database health check failed: {str(e)}")
            return health_data
            
    def purge_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """
        Purge old OHLCV data to manage database size
        Returns count of deleted records by table
        """
        deleted_counts = {}
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        try:
            with self.get_session() as session:
                # Delete old OHLCV data
                result = session.execute(
                    text("DELETE FROM ohlcv WHERE timestamp < :cutoff"),
                    {"cutoff": cutoff_date}
                )
                deleted_counts['ohlcv'] = result.rowcount
                
                # Delete old performance data (optional)
                result = session.execute(
                    text("DELETE FROM performance WHERE timestamp < :cutoff"),
                    {"cutoff": cutoff_date}
                )
                deleted_counts['performance'] = result.rowcount
                
                logger.info(f"Purged {sum(deleted_counts.values())} old records from database")
                return deleted_counts
        except Exception as e:
            logger.error(f"Failed to purge old data: {str(e)}")
            raise DatabaseError("Failed to purge old data", details={'error': str(e)})

    def optimize_database(self) -> bool:
        """
        Run database optimization tasks (VACUUM, ANALYZE)
        """
        try:
            # Create raw connection to run maintenance commands
            connection = self.engine.raw_connection()
            try:
                cursor = connection.cursor()
                
                # VACUUM ANALYZE for all tables
                tables = ['trades', 'ohlcv', 'performance']
                for table in tables:
                    cursor.execute(f"VACUUM ANALYZE {table}")
                
                connection.commit()
                logger.info("Database optimization completed successfully")
                return True
            finally:
                connection.close()
        except Exception as e:
            logger.error(f"Database optimization failed: {str(e)}")
            return False
            
    def get_symbols(self) -> List[str]:
        """Get list of all symbols in the database"""
        try:
            with self.get_session() as session:
                symbols = session.query(OHLCV.symbol).distinct().all()
                return [symbol[0] for symbol in symbols]
        except Exception as e:
            logger.error(f"Failed to retrieve symbols: {str(e)}")
            raise DatabaseError("Failed to retrieve symbols")
            
    def get_timeframes(self) -> List[str]:
        """Get list of all timeframes in the database"""
        try:
            with self.get_session() as session:
                timeframes = session.query(OHLCV.timeframe).distinct().all()
                return [tf[0] for tf in timeframes]
        except Exception as e:
            logger.error(f"Failed to retrieve timeframes: {str(e)}")
            raise DatabaseError("Failed to retrieve timeframes")
            
    def get_data_range(self, symbol: str, timeframe: str) -> Tuple[datetime, datetime]:
        """Get min and max timestamp for a symbol and timeframe"""
        try:
            with self.get_session() as session:
                min_ts = session.query(OHLCV.timestamp).filter(
                    OHLCV.symbol == symbol,
                    OHLCV.timeframe == timeframe
                ).order_by(OHLCV.timestamp.asc()).first()
                
                max_ts = session.query(OHLCV.timestamp).filter(
                    OHLCV.symbol == symbol,
                    OHLCV.timeframe == timeframe
                ).order_by(OHLCV.timestamp.desc()).first()
                
                if min_ts and max_ts:
                    return min_ts[0], max_ts[0]
                return None, None
        except Exception as e:
            logger.error(f"Failed to retrieve data range: {str(e)}")
            raise DatabaseError("Failed to retrieve data range")

# Global database instance
db = DatabaseManager()