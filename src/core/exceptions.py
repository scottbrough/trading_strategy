"""
Custom exception definitions for the trading system.
"""

class TradingSystemError(Exception):
    """Base class for trading system exceptions."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}

class DatabaseError(TradingSystemError):
    """Exception raised for database-related errors."""
    pass

class DataError(TradingSystemError):
    """Exception raised for data processing errors."""
    class DataValidationError(TradingSystemError):
        pass

class DataStreamError(TradingSystemError):
    """Exception raised for errors in data streaming."""
    pass

class StrategyError(TradingSystemError):
    """Exception raised for strategy-related errors."""
    pass

class ExecutionError(TradingSystemError):
    """Exception raised for order execution errors."""
    pass
