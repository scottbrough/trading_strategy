"""
Position management module.
Tracks open positions, updates PnL, and manages position lifecycle.
"""

from typing import Dict, List
from core.logger import log_manager

logger = log_manager.get_logger(__name__)

class PositionManager:
    def __init__(self):
        self.open_positions: List[Dict] = []

    def open_position(self, order: Dict) -> None:
        """
        Add a new open position.
        
        Args:
            order: The executed order data.
        """
        position = {
            "symbol": order.get("symbol"),
            "entry_price": order.get("filled_price"),
            "amount": order.get("amount"),
            "side": order.get("side"),
            "status": "open"
        }
        self.open_positions.append(position)
        logger.info(f"Opened position: {position}")

    def close_position(self, position_index: int, exit_price: float) -> Dict:
        """
        Close an open position.
        
        Args:
            position_index: Index of the position in the open_positions list.
            exit_price: The price at which the position is closed.
        
        Returns:
            A dictionary with details of the closed position.
        """
        if position_index < 0 or position_index >= len(self.open_positions):
            logger.error("Invalid position index")
            return {}

        position = self.open_positions.pop(position_index)
        position["exit_price"] = exit_price
        position["status"] = "closed"
        position["pnl"] = (exit_price - position["entry_price"]) * position["amount"] if position["side"] == "buy" else (position["entry_price"] - exit_price) * position["amount"]
        logger.info(f"Closed position: {position}")
        return position
