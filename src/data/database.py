"""
Database integration module using SQLAlchemy for the trading system.
Handles all database operations and provides connection pooling.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Generator, Any
import pandas as pd
from datetime import datetime
import json

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
            db_config = config.db_config
            db_url = f"postgresql://{db_config.user}:{db_config.password}@{db_config.host}:{db_config.port}/{db_config.name}"
            
            self.engine = create_engine(
                db_url,
                poolclass=QueuePool,
                pool_size=20,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=1800
            )
            
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

# Global database instance
db = DatabaseManager()