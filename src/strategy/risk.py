"""
Advanced risk management system implementing portfolio-level risk controls,
position sizing, and dynamic risk adjustment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from src.core.config import config
from src.core.logger import log_manager
from src.data.database import db_manager

logger = log_manager.get_logger(__name__)

@dataclass
class RiskMetrics:
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
    def __init__(self):
        self.config = config.get_risk_params()
        self.position_limits = {
            'max_position': self.config['max_position_size'],
            'max_portfolio': self.config['max_portfolio_risk'],
            'max_correlation': self.config['correlation_limit'],
        }
        self.metrics_history = []
    
    def calculate_risk_metrics(self, portfolio_data: Dict[str, pd.DataFrame]) -> RiskMetrics:
        """Calculate comprehensive risk metrics for the portfolio"""
        try:
            # Calculate returns for each asset
            returns = {}
            for symbol, df in portfolio_data.items():
                returns[symbol] = df['close'].pct_change().dropna()
            
            # Create returns matrix
            returns_df = pd.DataFrame(returns)
            
            # Calculate portfolio metrics
            correlation_matrix = returns_df.corr()
            portfolio_return = returns_df.mean()
            portfolio_vol = returns_df.std()
            
            # Calculate Value at Risk (VaR)
            var_95 = self._calculate_var(returns_df, confidence=0.95)
            cvar_95 = self._calculate_cvar(returns_df, confidence=0.95)
            
            # Calculate risk ratios
            sharpe = self._calculate_sharpe_ratio(returns_df)
            sortino = self._calculate_sortino_ratio(returns_df)
            
            # Calculate drawdown
            max_dd = self._calculate_max_drawdown(returns_df)
            
            # Calculate beta (assuming first asset is market)
            market_returns = returns_df.iloc[:, 0]
            betas = {}
            for col in returns_df.columns[1:]:
                betas[col] = self._calculate_beta(market_returns, returns_df[col])
            avg_beta = np.mean(list(betas.values()))
            
            # Calculate current exposure
            current_exposure = self._calculate_exposure(portfolio_data)
            
            # Calculate daily volatility
            daily_vol = returns_df.std() * np.sqrt(252)
            
            # Calculate Kelly Criterion
            kelly = self._calculate_kelly_fraction(returns_df)
            
            return RiskMetrics(
                var_95=var_95,
                cvar_95=cvar_95,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                max_drawdown=max_dd,
                beta=avg_beta,
                correlation_matrix=correlation_matrix,
                current_exposure=current_exposure,
                daily_volatility=daily_vol.mean(),
                kelly_fraction=kelly
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            raise
    
    def _calculate_var(self, returns: pd.DataFrame, confidence: float) -> float:
        """Calculate Value at Risk"""
        portfolio_returns = returns.mean(axis=1)
        return np.percentile(portfolio_returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, returns: pd.DataFrame, confidence: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        portfolio_returns = returns.mean(axis=1)
        var = self._calculate_var(returns, confidence)
        return portfolio_returns[portfolio_returns <= var].mean()
    
    def _calculate_sharpe_ratio(self, returns: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe Ratio"""
        portfolio_returns = returns.mean(axis=1)
        excess_returns = portfolio_returns - (risk_free_rate / 252)
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def _calculate_sortino_ratio(self, returns: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino Ratio"""
        portfolio_returns = returns.mean(axis=1)
        excess_returns = portfolio_returns - (risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    
    def _calculate_max_drawdown(self, returns: pd.DataFrame) -> float:
        """Calculate Maximum Drawdown"""
        portfolio_returns = returns.mean(axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        return abs(drawdowns.min())
    
    def _calculate_beta(self, market_returns: pd.Series, asset_returns: pd.Series) -> float:
        """Calculate beta relative to market"""
        covariance = market_returns.cov(asset_returns)
        market_variance = market_returns.var()
        return covariance / market_variance if market_variance != 0 else 1.0
    
    def _calculate_exposure(self, portfolio_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate current portfolio exposure"""
        total_exposure = 0
        for symbol, df in portfolio_data.items():
            position_value = df['close'].iloc[-1] * df['position_size'].iloc[-1]
            total_exposure += position_value
        return total_exposure
    
    def _calculate_kelly_fraction(self, returns: pd.DataFrame) -> float:
        """Calculate Kelly Criterion optimal fraction"""
        portfolio_returns = returns.mean(axis=1)
        win_rate = len(portfolio_returns[portfolio_returns > 0]) / len(portfolio_returns)
        avg_win = portfolio_returns[portfolio_returns > 0].mean()
        avg_loss = abs(portfolio_returns[portfolio_returns < 0].mean())
        
        if avg_loss == 0:
            return 0.0
            
        kelly = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
        return max(0.0, min(kelly, 1.0))  # Bound between 0 and 1
    
    def calculate_position_size(self, 
                              symbol: str, 
                              current_price: float,
                              volatility: float,
                              portfolio_metrics: RiskMetrics) -> float:
        """Calculate optimal position size considering multiple factors"""
        try:
            # Base position size using Kelly Criterion
            base_size = portfolio_metrics.kelly_fraction * self.config['max_position_size']
            
            # Adjust for volatility
            vol_adjustment = 1.0 - (volatility / portfolio_metrics.daily_volatility)
            vol_adjustment = max(0.2, min(vol_adjustment, 1.0))
            
            # Adjust for portfolio exposure
            exposure_limit = self.config['max_portfolio_risk']
            exposure_adjustment = 1.0 - (portfolio_metrics.current_exposure / exposure_limit)
            exposure_adjustment = max(0.0, min(exposure_adjustment, 1.0))
            
            # Adjust for correlation
            correlation_penalty = self._calculate_correlation_penalty(symbol, portfolio_metrics)
            
            # Calculate final position size
            position_size = (base_size * 
                           vol_adjustment * 
                           exposure_adjustment * 
                           correlation_penalty)
            
            # Convert to units
            units = (position_size * self.config['account_balance']) / current_price
            
            # Apply minimum and maximum constraints
            min_units = (self.config['min_position_size'] * self.config['account_balance']) / current_price
            max_units = (self.config['max_position_size'] * self.config['account_balance']) / current_price
            
            return max(min_units, min(units, max_units))
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            raise
    
    def _calculate_correlation_penalty(self, symbol: str, portfolio_metrics: RiskMetrics) -> float:
        """Calculate position size penalty based on correlation with existing positions"""
        try:
            if symbol not in portfolio_metrics.correlation_matrix.columns:
                return 1.0
                
            correlations = portfolio_metrics.correlation_matrix[symbol]
            max_correlation = correlations.abs().max()
            
            if max_correlation > self.position_limits['max_correlation']:
                return 0.0
            
            # Linear penalty between 0.5 and max correlation limit
            if max_correlation > 0.5:
                penalty = 1.0 - ((max_correlation - 0.5) / (self.position_limits['max_correlation'] - 0.5))
                return max(0.2, penalty)
            
            return 1.0
            
        except Exception as e:
            logger.error(f"Error calculating correlation penalty: {str(e)}")
            return 0.0
    
    def check_risk_limits(self, portfolio_metrics: RiskMetrics) -> Tuple[bool, List[str]]:
        """Check if current portfolio meets risk limits"""
        violations = []
        
        # Check Value at Risk
        if abs(portfolio_metrics.var_95) > self.config['max_var']:
            violations.append(f"VaR exceeds limit: {portfolio_metrics.var_95:.2%}")
        
        # Check portfolio exposure
        if portfolio_metrics.current_exposure > self.config['max_portfolio_risk']:
            violations.append(f"Portfolio exposure exceeds limit: {portfolio_metrics.current_exposure:.2%}")
        
        # Check drawdown
        if portfolio_metrics.max_drawdown > self.config['max_drawdown']:
            violations.append(f"Drawdown exceeds limit: {portfolio_metrics.max_drawdown:.2%}")
        
        # Check portfolio volatility
        if portfolio_metrics.daily_volatility > self.config['max_volatility']:
            violations.append(f"Volatility exceeds limit: {portfolio_metrics.daily_volatility:.2%}")
        
        return len(violations) == 0, violations
    
    def adjust_for_market_regime(self, 
                               base_size: float, 
                               market_regime: str,
                               volatility: float) -> float:
        """Adjust position size based on market regime"""
        try:
            regime_adjustments = {
                'high_volatility': 0.5,
                'trending': 1.2,
                'ranging': 0.8,
                'normal': 1.0
            }
            
            adjustment = regime_adjustments.get(market_regime, 1.0)
            
            # Further adjust based on volatility in high vol regime
            if market_regime == 'high_volatility':
                vol_scale = max(0.3, 1.0 - (volatility / self.config['max_volatility']))
                adjustment *= vol_scale
            
            return base_size * adjustment
            
        except Exception as e:
            logger.error(f"Error adjusting for market regime: {str(e)}")
            return base_size