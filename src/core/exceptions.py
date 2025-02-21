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

class DataError:
    """Container for data-related errors."""
    
    class DataValidationError(TradingSystemError):
        """Exception raised for data validation errors."""
        pass
    
    class DataFetchError(TradingSystemError):
        """Exception raised for data fetching errors."""
        pass
    
    class DataProcessingError(TradingSystemError):
        """Exception raised for data processing errors."""
        pass

class DataStreamError(TradingSystemError):
    """Exception raised for errors in data streaming."""
    
    class ConnectionError(TradingSystemError):
        """Exception raised for connection issues."""
        pass
    
    class SubscriptionError(TradingSystemError):
        """Exception raised for subscription issues."""
        pass

class StrategyError(TradingSystemError):
    """Exception raised for strategy-related errors."""
    
    class ValidationError(TradingSystemError):
        """Exception raised for strategy validation errors."""
        pass
    
    class OptimizationError(TradingSystemError):
        """Exception raised for optimization errors."""
        pass
    
    class BacktestError(TradingSystemError):
        """Exception raised for backtesting errors."""
        pass

class ExecutionError(TradingSystemError):
    """Exception raised for order execution errors."""
    
    class OrderError(TradingSystemError):
        """Exception raised for order-related errors."""
        pass
    
    class PositionError(TradingSystemError):
        """Exception raised for position-related errors."""
        pass
    
    class LimitError(TradingSystemError):
        """Exception raised for trading limit violations."""
        pass