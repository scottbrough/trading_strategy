"""
Technical indicators library for the trading system.

This module provides reusable functions for calculating common technical indicators
using TA-Lib. Strategies can import these functions to incorporate indicator values into
their signal generation logic.
"""

import pandas as pd
import talib


def calculate_rsi(close: pd.Series, timeperiod: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) for a given price series.
    
    Args:
        close: A pandas Series of closing prices.
        timeperiod: The lookback period (default 14).
    
    Returns:
        A pandas Series containing the RSI values.
    """
    return talib.RSI(close, timeperiod=timeperiod)


def calculate_macd(close: pd.Series, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> pd.DataFrame:
    """
    Calculate the Moving Average Convergence Divergence (MACD).
    
    Args:
        close: A pandas Series of closing prices.
        fastperiod: The short-term EMA period (default 12).
        slowperiod: The long-term EMA period (default 26).
        signalperiod: The signal line EMA period (default 9).
    
    Returns:
        A DataFrame with columns 'macd', 'signal', and 'hist' representing the MACD line, signal line, and histogram.
    """
    macd, signal, hist = talib.MACD(close, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    return pd.DataFrame({
        "macd": macd,
        "signal": signal,
        "hist": hist
    })


def calculate_bbands(close: pd.Series, timeperiod: int = 20, nbdevup: int = 2, nbdevdn: int = 2) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    
    Args:
        close: A pandas Series of closing prices.
        timeperiod: The moving average period (default 20).
        nbdevup: Number of standard deviations above the moving average (default 2).
        nbdevdn: Number of standard deviations below the moving average (default 2).
    
    Returns:
        A DataFrame with columns 'upper', 'middle', and 'lower' representing the Bollinger Bands.
    """
    upper, middle, lower = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn)
    return pd.DataFrame({
        "upper": upper,
        "middle": middle,
        "lower": lower
    })


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14) -> pd.Series:
    """
    Calculate the Average Directional Index (ADX) to measure trend strength.
    
    Args:
        high: A pandas Series of high prices.
        low: A pandas Series of low prices.
        close: A pandas Series of closing prices.
        timeperiod: The period over which to calculate ADX (default 14).
    
    Returns:
        A pandas Series containing the ADX values.
    """
    return talib.ADX(high, low, close, timeperiod=timeperiod)


def calculate_ema(close: pd.Series, timeperiod: int = 20) -> pd.Series:
    """
    Calculate the Exponential Moving Average (EMA).
    
    Args:
        close: A pandas Series of closing prices.
        timeperiod: The period for the EMA (default 20).
    
    Returns:
        A pandas Series containing the EMA values.
    """
    return talib.EMA(close, timeperiod=timeperiod)


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                         fastk_period: int = 14, slowk_period: int = 3, slowk_matype: int = 0,
                         slowd_period: int = 3, slowd_matype: int = 0) -> pd.DataFrame:
    """
    Calculate the Stochastic Oscillator.
    
    Args:
        high: A pandas Series of high prices.
        low: A pandas Series of low prices.
        close: A pandas Series of closing prices.
        fastk_period: The fast %K period (default 14).
        slowk_period: The slow %K period (default 3).
        slowk_matype: The moving average type for slow %K (default 0).
        slowd_period: The slow %D period (default 3).
        slowd_matype: The moving average type for slow %D (default 0).
    
    Returns:
        A DataFrame with columns 'slowk' and 'slowd' representing the stochastic oscillator.
    """
    slowk, slowd = talib.STOCH(high, low, close,
                               fastk_period=fastk_period,
                               slowk_period=slowk_period,
                               slowk_matype=slowk_matype,
                               slowd_period=slowd_period,
                               slowd_matype=slowd_matype)
    return pd.DataFrame({
        "slowk": slowk,
        "slowd": slowd
    })


def calculate_ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
                       conversion_line_period: int = 9, base_line_period: int = 26, leading_span_b_period: int = 52) -> pd.DataFrame:
    """
    Calculate Ichimoku Cloud components.
    
    Args:
        high: A pandas Series of high prices.
        low: A pandas Series of low prices.
        close: A pandas Series of closing prices.
        conversion_line_period: Period for the conversion line (default 9).
        base_line_period: Period for the base line (default 26).
        leading_span_b_period: Period for the leading span B (default 52).
    
    Returns:
        A DataFrame with columns 'conversion_line', 'base_line', 'leading_span_a', and 'leading_span_b'.
    """
    conversion_line = (high.rolling(window=conversion_line_period).max() + low.rolling(window=conversion_line_period).min()) / 2
    base_line = (high.rolling(window=base_line_period).max() + low.rolling(window=base_line_period).min()) / 2
    leading_span_a = ((conversion_line + base_line) / 2).shift(base_line_period)
    leading_span_b = ((high.rolling(window=leading_span_b_period).max() + low.rolling(window=leading_span_b_period).min()) / 2).shift(base_line_period)
    return pd.DataFrame({
        "conversion_line": conversion_line,
        "base_line": base_line,
        "leading_span_a": leading_span_a,
        "leading_span_b": leading_span_b
    })
