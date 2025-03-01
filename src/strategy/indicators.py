"""
Technical indicators library for the trading system.
Implements advanced indicators and custom calculations.
"""

import pandas as pd
import numpy as np
import talib
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from ..core.logger import log_manager

logger = log_manager.get_logger(__name__)

@dataclass
class IndicatorConfig:
    """Configuration for indicator calculations"""
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_stddev: float = 2.0
    atr_period: int = 14
    ema_periods: Tuple[int, ...] = (9, 21, 50, 200)
    stoch_k: int = 14
    stoch_d: int = 3
    stoch_smooth: int = 3
    adx_period: int = 14
    volume_ma_period: int = 20

class TechnicalIndicators:
    def __init__(self, config: Optional[IndicatorConfig] = None):
        """Initialize with optional custom configuration"""
        self.config = config or IndicatorConfig()
        self._cache = {}
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        try:
            df = df.copy()
            
            # Trend indicators
            self._add_moving_averages(df)
            self._add_macd(df)
            self._add_adx(df)
            
            # Momentum indicators
            self._add_rsi(df)
            self._add_stochastic(df)
            self._add_momentum(df)
            
            # Volatility indicators
            self._add_bollinger_bands(df)
            self._add_atr(df)
            
            # Volume indicators
            self._add_volume_indicators(df)
            
            # Custom indicators
            self._add_custom_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise
    
    def _add_moving_averages(self, df: pd.DataFrame) -> None:
        """Add various moving averages"""
        try:
            for period in self.config.ema_periods:
                df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
                df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
            
            # Add hull moving average
            def hull_ma(data: pd.Series, period: int) -> pd.Series:
                half_period = period // 2
                sqrt_period = int(np.sqrt(period))
                
                wma1 = talib.WMA(data, timeperiod=half_period)
                wma2 = talib.WMA(data, timeperiod=period)
                diff = 2 * wma1 - wma2
                return talib.WMA(diff, timeperiod=sqrt_period)
            
            df['hull_ma'] = hull_ma(df['close'], 20)
            
        except Exception as e:
            logger.error(f"Error adding moving averages: {str(e)}")
            raise
    
    def _add_macd(self, df: pd.DataFrame) -> None:
        """Add MACD indicator"""
        try:
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
                df['close'],
                fastperiod=self.config.macd_fast,
                slowperiod=self.config.macd_slow,
                signalperiod=self.config.macd_signal
            )
            
            # Add normalized MACD
            df['macd_norm'] = df['macd'] / df['close']
            
        except Exception as e:
            logger.error(f"Error adding MACD: {str(e)}")
            raise
    
    def _add_rsi(self, df: pd.DataFrame) -> None:
        """Add RSI and derived indicators"""
        try:
            df['rsi'] = talib.RSI(df['close'], timeperiod=self.config.rsi_period)
            
            # Add RSI moving average
            df['rsi_ma'] = talib.SMA(df['rsi'], timeperiod=self.config.rsi_period)
            
            # Add RSI divergence
            def calculate_divergence(price: pd.Series, rsi: pd.Series, window: int = 14) -> pd.Series:
                price_min = price.rolling(window=window).min()
                price_max = price.rolling(window=window).max()
                rsi_min = rsi.rolling(window=window).min()
                rsi_max = rsi.rolling(window=window).max()
                
                bearish = (price_max > price_max.shift(1)) & (rsi_max < rsi_max.shift(1))
                bullish = (price_min < price_min.shift(1)) & (rsi_min > rsi_min.shift(1))
                
                return pd.Series(index=price.index, data=np.where(bearish, -1, np.where(bullish, 1, 0)))
            
            df['rsi_divergence'] = calculate_divergence(df['close'], df['rsi'])
            
        except Exception as e:
            logger.error(f"Error adding RSI: {str(e)}")
            raise
    
    def _add_bollinger_bands(self, df: pd.DataFrame) -> None:
        """Add Bollinger Bands and derived indicators"""
        try:
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
                df['close'],
                timeperiod=self.config.bb_period,
                nbdevup=self.config.bb_stddev,
                nbdevdn=self.config.bb_stddev
            )
            
            # Add BB width and %B indicators
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_pct_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
        except Exception as e:
            logger.error(f"Error adding Bollinger Bands: {str(e)}")
            raise
    
    def _add_stochastic(self, df: pd.DataFrame) -> None:
        """Add Stochastic indicators"""
        try:
            df['slowk'], df['slowd'] = talib.STOCH(
                df['high'],
                df['low'],
                df['close'],
                fastk_period=self.config.stoch_k,
                slowk_period=self.config.stoch_smooth,
                slowk_matype=0,
                slowd_period=self.config.stoch_d,
                slowd_matype=0
            )
            
            # Add Stochastic RSI
            df['stoch_rsi'] = talib.STOCHRSI(
                df['close'],
                timeperiod=self.config.rsi_period,
                fastk_period=self.config.stoch_k,
                fastd_period=self.config.stoch_d
            )[0]  # STOCHRSI returns a tuple, we want the first element
            
        except Exception as e:
            logger.error(f"Error adding Stochastic: {str(e)}")
            raise
    
    def _add_adx(self, df: pd.DataFrame) -> None:
        """Add ADX and DMI indicators"""
        try:
            df['adx'] = talib.ADX(
                df['high'],
                df['low'],
                df['close'],
                timeperiod=self.config.adx_period
            )
            
            # Add DI+ and DI-
            df['plus_di'] = talib.PLUS_DI(
                df['high'],
                df['low'],
                df['close'],
                timeperiod=self.config.adx_period
            )
            df['minus_di'] = talib.MINUS_DI(
                df['high'],
                df['low'],
                df['close'],
                timeperiod=self.config.adx_period
            )
            
            # Add ADX trend strength
            df['adx_trend'] = np.where(
                df['adx'] > 25,
                np.where(df['plus_di'] > df['minus_di'], 1, -1),
                0
            )
            
        except Exception as e:
            logger.error(f"Error adding ADX: {str(e)}")
            raise
    
    def _add_atr(self, df: pd.DataFrame) -> None:
        """Add ATR and derived indicators"""
        try:
            df['atr'] = talib.ATR(
                df['high'],
                df['low'],
                df['close'],
                timeperiod=self.config.atr_period
            )
            
            # Add normalized ATR
            df['atr_pct'] = df['atr'] / df['close'] * 100
            
            # Add ATR-based channels
            df['atr_upper'] = df['close'] + (df['atr'] * 2)
            df['atr_lower'] = df['close'] - (df['atr'] * 2)
            
        except Exception as e:
            logger.error(f"Error adding ATR: {str(e)}")
            raise
    
    def _add_momentum(self, df: pd.DataFrame) -> None:
        """Add momentum indicators"""
        try:
            # ROC
            df['roc'] = talib.ROC(df['close'], timeperiod=10)
            
            # Momentum
            df['mom'] = talib.MOM(df['close'], timeperiod=10)
            
            # PPO
            df['ppo'] = talib.PPO(df['close'], fastperiod=12, slowperiod=26)
            
            # Add custom momentum score
            def momentum_score(data: pd.DataFrame) -> pd.Series:
                roc_norm = (data['roc'] - data['roc'].rolling(20).mean()) / data['roc'].rolling(20).std()
                mom_norm = (data['mom'] - data['mom'].rolling(20).mean()) / data['mom'].rolling(20).std()
                ppo_norm = (data['ppo'] - data['ppo'].rolling(20).mean()) / data['ppo'].rolling(20).std()
                
                return (roc_norm + mom_norm + ppo_norm) / 3
            
            df['momentum_score'] = momentum_score(df)
            
        except Exception as e:
            logger.error(f"Error adding momentum indicators: {str(e)}")
            raise
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> None:
        """Add volume-based indicators"""
        try:
            # OBV
            df['obv'] = talib.OBV(df['close'], df['volume'])
            
            # Volume MA
            df['volume_sma'] = talib.SMA(df['volume'], timeperiod=self.config.volume_ma_period)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Money Flow Index
            df['mfi'] = talib.MFI(
                df['high'],
                df['low'],
                df['close'],
                df['volume'],
                timeperiod=14
            )
            
        except Exception as e:
            logger.error(f"Error adding volume indicators: {str(e)}")
            raise
    
    def _add_custom_indicators(self, df: pd.DataFrame) -> None:
        """Add custom composite indicators"""
        try:
            # Trend strength indicator
            def trend_strength(data: pd.DataFrame) -> pd.Series:
                # Check if we have the required columns
                if not all(col in data.columns for col in ['ema_9', 'ema_21', 'ema_50', 'adx', 'volume_ratio']):
                    # Return a series of zeros if columns are missing
                    return pd.Series(0, index=data.index)
                
                ema_align = (data['ema_9'] > data['ema_21']) & \
                          (data['ema_21'] > data['ema_50'])
                adx_strong = data['adx'] > 25
                vol_confirm = data['volume_ratio'] > 1.0
                
                return np.where(
                    ema_align & adx_strong & vol_confirm,
                    1,
                    np.where(
                        (~ema_align) & adx_strong & vol_confirm,
                        -1,
                        0
                    )
                )
            
            df['trend_strength'] = trend_strength(df)
            
            # Volatility regime
            def volatility_regime(data: pd.DataFrame) -> pd.Series:
                if 'bb_width' not in data.columns or 'atr_pct' not in data.columns:
                    return pd.Series('normal', index=data.index)
                
                bb_high = data['bb_width'] > data['bb_width'].rolling(20).mean()
                atr_high = data['atr_pct'] > data['atr_pct'].rolling(20).mean()
                
                return np.where(
                    bb_high & atr_high,
                    'high',
                    np.where(
                        (~bb_high) & (~atr_high),
                        'low',
                        'normal'
                    )
                )
            
            df['volatility_regime'] = volatility_regime(df)
            
            # Combined momentum signal
            def momentum_signal(data: pd.DataFrame) -> pd.Series:
                if not all(col in data.columns for col in ['rsi', 'macd_hist', 'slowk']):
                    return pd.Series(0, index=data.index)
                
                rsi_signal = (data['rsi'] < 30) | (data['rsi'] > 70)
                macd_signal = data['macd_hist'] > 0
                stoch_signal = (data['slowk'] < 20) | (data['slowk'] > 80)
                
                return np.where(
                    rsi_signal & macd_signal & stoch_signal,
                    np.where(data['rsi'] < 30, 1, -1),
                    0
                )
            
            df['momentum_signal'] = momentum_signal(df)
            
        except Exception as e:
            logger.error(f"Error adding custom indicators: {str(e)}")
            raise