"""
Advanced risk management system implementing portfolio-level risk controls,
position sizing, and dynamic risk adjustment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from ..core.config import config
from ..core.logger import log_manager
from ..core.exceptions import StrategyError
from ..data.database import db

logger = log_manager.get_logger(__name__)

@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    var_95: float
    cvar_95: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    beta: float
    correlation_matrix: pd.DataFrame
    current_exposure: float
    daily_volatility: float
    kelly_fraction: float

class RiskManager:
    def __init__(self, config: Dict[str, Any]):
        """Initialize risk manager with configuration"""
        self.config = config
        # Check if risk_params exists, if not use the config itself
        self.risk_params = config.get('risk_params', config)
        self.positions = []
        self.daily_stats = {
            'pnl': 0,
            'trades': 0,
            'max_loss': -float('inf'),
            'start_time': datetime.now()
            }
    
    def calculate_position_size(self, 
                              capital: float,
                              price: float,
                              volatility: float,
                              trade_params: Dict[str, Any]) -> float:
        """
        Calculate optimal position size with risk adjustments
        
        Args:
            capital: Available trading capital
            price: Current asset price
            volatility: Current volatility measure
            trade_params: Additional parameters including win rate and profit factor
            
        Returns:
            float: Optimal position size
        """
        try:
            # Base position size using risk parameters
            max_risk = capital * self.risk_params['max_risk_per_trade']
            
            # Kelly Criterion calculation
            if 'win_rate' in trade_params and 'profit_factor' in trade_params:
                kelly_fraction = self.calculate_kelly_fraction(
                    trade_params['win_rate'],
                    trade_params['profit_factor']
                )
                # Use half-Kelly for safety
                base_size = capital * kelly_fraction * 0.5
            else:
                base_size = max_risk
            
            # Volatility adjustment
            vol_factor = 1.0 - (min(volatility, 50) / 100)
            position_size = base_size * vol_factor
            
            # Apply position limits
            max_position = capital * self.risk_params['max_position_size'] / price
            min_position = capital * self.risk_params['min_position_size'] / price
            
            return min(max(position_size, min_position), max_position)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
    
    def calculate_kelly_fraction(self, win_rate: float, profit_factor: float) -> float:
        """Calculate Kelly Criterion optimal fraction"""
        try:
            q = 1 - win_rate  # Probability of loss
            if q == 0:
                return 0.0
            
            return (win_rate / q) * (profit_factor - 1)
            
        except Exception as e:
            logger.error(f"Error calculating Kelly fraction: {str(e)}")
            return 0.0
    
    def check_risk_limits(self, new_position: Dict[str, Any]) -> bool:
        """
        Check if new position violates risk limits
        
        Args:
            new_position: Dictionary containing position details
            
        Returns:
            bool: True if position is within risk limits
        """
        try:
            # Calculate current metrics
            metrics = self.calculate_risk_metrics()
            
            # Check drawdown
            if metrics.current_drawdown > self.risk_params['max_drawdown']:
                logger.warning(f"Max drawdown exceeded: {metrics.current_drawdown:.2%}")
                return False
            
            # Check exposure
            if metrics.current_exposure > self.risk_params['max_position_size']:
                logger.warning(f"Max exposure exceeded: {metrics.current_exposure:.2%}")
                return False
            
            # Check daily loss
            if abs(self.daily_stats['pnl']) > self.risk_params['max_daily_loss']:
                logger.warning(f"Daily loss limit reached: {self.daily_stats['pnl']:.2f}")
                return False
            
            # Check correlation
            if self.check_correlation_risk(new_position):
                logger.warning("Correlation risk too high")
                return False
            
            # Check VaR limits
            if metrics.var_95 > self.risk_params.get('var_limit', 0.02):
                logger.warning(f"VaR limit exceeded: {metrics.var_95:.2%}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {str(e)}")
            return False
    
    def calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            # Get historical returns for calculations
            returns = self.calculate_portfolio_returns()
            
            # Calculate metrics
            var_95 = self.calculate_var(returns, 0.95)
            cvar_95 = self.calculate_cvar(returns, 0.95)
            sharpe = self.calculate_sharpe_ratio(returns)
            sortino = self.calculate_sortino_ratio(returns)
            max_dd = self.calculate_max_drawdown(returns)
            beta = self.calculate_portfolio_beta(returns)
            
            # Current portfolio state
            exposure = sum(p['size'] * p['price'] for p in self.positions)
            daily_vol = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
            
            # Correlation matrix
            corr_matrix = pd.DataFrame()
            if len(self.positions) > 1:
                position_returns = pd.DataFrame({
                    p['symbol']: p.get('returns', []) for p in self.positions
                })
                corr_matrix = position_returns.corr()
            
            metrics = RiskMetrics(
                var_95=var_95,
                cvar_95=cvar_95,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                max_drawdown=max_dd,
                beta=beta,
                correlation_matrix=corr_matrix,
                current_exposure=exposure,
                daily_volatility=daily_vol,
                kelly_fraction=self.calculate_kelly_fraction(0.5, 1.0)  # Default values
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return RiskMetrics(0, 0, 0, 0, 0, 0, pd.DataFrame(), 0, 0, 0)
    
    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        try:
            if len(returns) < 2:
                return 0.0
            return -np.percentile(returns, (1 - confidence) * 100)
        except Exception as e:
            logger.error(f"Error calculating VaR: {str(e)}")
            return 0.0
    
    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        try:
            if len(returns) < 2:
                return 0.0
            var = self.calculate_var(returns, confidence)
            return -returns[returns <= -var].mean()
        except Exception as e:
            logger.error(f"Error calculating CVaR: {str(e)}")
            return 0.0
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max
            return abs(drawdowns.min()) if len(drawdowns) > 0 else 0
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(returns) < 2:
                return 0.0
            excess_returns = returns - (risk_free_rate / 252)
            return np.sqrt(252) * excess_returns.mean() / returns.std()
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        try:
            if len(returns) < 2:
                return 0.0
            excess_returns = returns - (risk_free_rate / 252)
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252)
            return excess_returns.mean() * 252 / downside_std if downside_std != 0 else 0
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {str(e)}")
            return 0.0
    
    def check_correlation_risk(self, new_position: Dict[str, Any]) -> bool:
        """Check if new position creates excessive correlation risk"""
        try:
            if not self.positions:
                return False
            
            correlations = []
            for pos in self.positions:
                if 'returns' in pos and 'returns' in new_position:
                    corr = np.corrcoef(pos['returns'], new_position['returns'])[0,1]
                    correlations.append(abs(corr))
            
            max_corr = max(correlations, default=0)
            return max_corr > self.risk_params.get('max_correlation', 0.7)
            
        except Exception as e:
            logger.error(f"Error checking correlation risk: {str(e)}")
            return True  # Conservative approach
    
    def update_position_sizing(self, position: Dict[str, Any],
                             market_conditions: Dict[str, Any]) -> float:
        """
        Update position size based on market conditions
        
        Args:
            position: Current position details
            market_conditions: Dictionary of market metrics
        
        Returns:
            float: Updated position size
        """
        try:
            # Get base position size
            current_size = position['size']
            
            # Adjust for volatility
            if 'volatility' in market_conditions:
                vol_factor = 1.0 - (min(market_conditions['volatility'], 50) / 100)
                current_size *= vol_factor
            
            # Adjust for trend strength
            if 'trend_strength' in market_conditions:
                trend_factor = 1.0 + (market_conditions['trend_strength'] * 0.2)
                current_size *= trend_factor
            
            # Apply limits
            max_size = position['capital'] * self.risk_params['max_position_size']
            min_size = position['capital'] * self.risk_params['min_position_size']
            
            return min(max(current_size, min_size), max_size)
            
        except Exception as e:
            logger.error(f"Error updating position size: {str(e)}")
            return position['size']
    
    def calculate_portfolio_returns(self) -> pd.Series:
        """Calculate portfolio returns series"""
        try:
            if not self.positions:
                return pd.Series()
            
            # Combine position returns with weights
            weighted_returns = []
            for pos in self.positions:
                if 'returns' in pos:
                    weight = pos['size'] * pos['price'] / sum(p['size'] * p['price'] for p in self.positions)
                    weighted_returns.append(pos['returns'] * weight)
            
            if weighted_returns:
                return pd.concat(weighted_returns, axis=1).sum(axis=1)
            return pd.Series()
            
        except Exception as e:
            logger.error(f"Error calculating portfolio returns: {str(e)}")
            return pd.Series()
    
    def update_daily_stats(self):
        """Reset daily statistics at day change"""
        try:
            current_time = datetime.now()
            if current_time.date() > self.daily_stats['start_time'].date():
                logger.info("Resetting daily statistics")
                self.daily_stats = {
                    'pnl': 0,
                    'trades': 0,
                    'max_loss': -float('inf'),
                    'start_time': current_time
                }
        except Exception as e:
            logger.error(f"Error updating daily stats: {str(e)}")
            
    def adjust_for_market_regime(self, 
                               base_size: float, 
                               market_regime: str,
                               volatility: float) -> float:
        """
        Adjust position size based on market regime
        
        Args:
            base_size: Initial position size
            market_regime: Current market regime ('trending', 'ranging', 'volatile')
            volatility: Current volatility level
            
        Returns:
            float: Adjusted position size
        """
        try:
            # Define regime-based adjustments
            regime_factors = {
                'trending': 1.2,
                'ranging': 0.8,
                'volatile': 0.6
            }
            
            # Get regime adjustment factor
            regime_factor = regime_factors.get(market_regime, 1.0)
            
            # Additional volatility adjustment for high volatility regime
            if market_regime == 'volatile':
                vol_scale = max(0.3, 1.0 - (volatility / self.risk_params['max_volatility']))
                regime_factor *= vol_scale
            
            return base_size * regime_factor
            
        except Exception as e:
            logger.error(f"Error adjusting for market regime: {str(e)}")
            return base_size
        
    def calculate_portfolio_beta(self, returns: pd.Series, 
                          market_returns: pd.Series = None) -> float:
        """Calculate portfolio beta relative to market"""
        try:
            if market_returns is None:
                # Use BTC as proxy for market
                market_returns = self._get_market_returns()
                
            if len(returns) < 2 or len(market_returns) < 2:
                return 1.0  # Default to 1.0 if not enough data
                
            # Align dates
            aligned_returns = pd.DataFrame({
                'portfolio': returns,
                'market': market_returns
            }).dropna()
            
            # Calculate beta using covariance method
            covariance = aligned_returns['portfolio'].cov(aligned_returns['market'])
            market_variance = aligned_returns['market'].var()
            
            beta = covariance / market_variance if market_variance != 0 else 1.0
            return beta
            
        except Exception as e:
            logger.error(f"Error calculating portfolio beta: {str(e)}")
            return 1.0

    def _get_market_returns(self) -> pd.Series:
        """Get market returns (using BTC as proxy)"""
        try:
            # Get the past 90 days of BTC data
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=90)
            
            btc_data = db.get_ohlcv("BTC/USD", "1d", start_date, end_date)
            
            if btc_data.empty:
                return pd.Series()
                
            return btc_data['close'].pct_change().dropna()
            
        except Exception as e:
            logger.error(f"Error getting market returns: {str(e)}")
            return pd.Series()

    def _has_position(self, side: str = None) -> bool:
        """Check if we have an open position of the specified side"""
        if side:
            return any(pos['side'] == side for pos in self.positions)
        return bool(self.positions)

    def _check_position_limits(self, position: Dict[str, Any]) -> bool:
        """Check if position is within limits"""
        # Max positions
        if len(self.positions) >= self.config.get('max_open_positions', 10):
            return False
            
        # Max exposure per position
        total_exposure = sum(pos['size'] * pos['price'] for pos in self.positions)
        new_exposure = position['size'] * position['price']
        
        if new_exposure > self.config.get('max_position_size', 0.2) * self.total_capital:
            return False
            
        # Max exposure for all positions
        if (total_exposure + new_exposure) > self.config.get('max_total_exposure', 0.5) * self.total_capital:
            return False
            
        return True

    def _calculate_position_risk(self, position: Dict[str, Any]) -> float:
        """Calculate risk for position as fraction of capital"""
        try:
            position_value = position['size'] * position['price']
            stop_loss_pct = self.config.get('stop_loss', 0.02)
            
            risk_amount = position_value * stop_loss_pct
            risk_fraction = risk_amount / self.total_capital
            
            return risk_fraction
            
        except Exception as e:
            logger.error(f"Error calculating position risk: {str(e)}")
            return 0.0

    def log_trade(self, trade: Dict[str, Any]) -> None:
        """Log completed trade and update statistics"""
        try:
            # Update daily stats
            self.daily_stats['trades'] += 1
            self.daily_stats['pnl'] += trade['pnl']
            
            if trade['pnl'] < 0:
                self.daily_stats['max_loss'] = min(self.daily_stats['max_loss'], trade['pnl'])
                
            # Store trade in database
            db.store_trade({
                'symbol': trade['symbol'],
                'side': trade['side'],
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'amount': trade['size'],
                'entry_time': trade['entry_time'],
                'exit_time': trade['exit_time'],
                'pnl': trade['pnl'],
                'status': 'closed',
                'strategy': self.config.get('strategy_name', 'unknown')
            })
            
        except Exception as e:
            logger.error(f"Error logging trade: {str(e)}")