"""
Base strategy class for the trading system.

All custom trading strategies should inherit from BaseStrategy and implement the generate_signals method.
This class also provides common functionality for backtesting, trade simulation, and real-time processing.
"""

import abc
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import pandas as pd


class BaseStrategy(ABC):
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the strategy with configuration and parameters.
        
        Args:
            config: A dictionary containing configuration parameters.
                    Expected to include a "strategy" key with strategy-specific settings.
        """
        self.config = config
        self.params = config.get("strategy", {})
        self.positions: List[Dict[str, Any]] = []  # Current open positions
        self.trades: List[Dict[str, Any]] = []       # List of executed (or simulated) trades

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on processed market data.
        
        Args:
            data: A DataFrame of processed OHLCV data.
        
        Returns:
            A list of signal dictionaries. Each signal might include fields such as:
            - 'action': 'buy' or 'sell'
            - 'confidence': a numerical value indicating signal strength
            - any other parameters used for trade execution.
        """
        pass

    def backtest(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Run a backtest simulation on historical data.
        
        Args:
            data: A DataFrame containing historical OHLCV data.
        
        Returns:
            A list of simulated trade records.
        """
        signals = self.generate_signals(data)
        for signal in signals:
            trade = self.simulate_trade(signal, data)
            self.trades.append(trade)
        return self.trades

    def simulate_trade(self, signal: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """
        Simulate a trade execution based on the generated signal.
        This basic simulation assumes the trade is executed at the close price of the last bar.
        
        Args:
            signal: A dictionary representing the trade signal.
            data: A DataFrame of historical data.
        
        Returns:
            A dictionary representing the executed trade, with fields for entry price, exit price, pnl, etc.
        """
        entry_price = data['close'].iloc[-1]
        trade = {
            "signal": signal,
            "entry_price": entry_price,
            "exit_price": entry_price,  # For a basic simulation, use the same price
            "pnl": 0.0,               # Profit/Loss to be calculated with a more advanced model
            "status": "closed"
        }
        return trade

    def on_data(self, data: pd.DataFrame) -> None:
        """
        Process incoming market data in real time.
        This method calls generate_signals and, if any signals are produced, passes them to the execution logic.
        
        Args:
            data: A DataFrame with the latest OHLCV data.
        """
        signals = self.generate_signals(data)
        if signals:
            for signal in signals:
                self.execute_trade(signal, data)

    def execute_trade(self, signal: Dict[str, Any], data: pd.DataFrame) -> None:
        """
        Placeholder for trade execution logic.
        In a live system, this method would interface with a broker or execution module.
        Here, it simulates immediate trade execution.
        
        Args:
            signal: A dictionary representing the trading signal.
            data: A DataFrame with the latest market data.
        """
        trade = self.simulate_trade(signal, data)
        self.trades.append(trade)
        print(f"Executed trade: {trade}")
