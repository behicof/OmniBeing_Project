"""
Data management system for the OmniBeing Trading System.
Handles market data fetching, preprocessing, and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import requests
import time
from datetime import datetime, timedelta
from config import config


class DataManager:
    """Manages market data fetching, preprocessing, and feature engineering."""
    
    def __init__(self):
        """Initialize the data manager."""
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.live_data: List[Dict[str, Any]] = []
        self.processed_features: Dict[str, Any] = {}
        
    def fetch_historical_data(self, symbol: str, timeframe: str = '1h', 
                            limit: int = 1000) -> pd.DataFrame:
        """
        Fetch historical market data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Data timeframe (e.g., '1h', '4h', '1d')
            limit: Number of data points to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        # Mock implementation - in production, this would connect to real APIs
        # For now, generate sample data to demonstrate the structure
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='1H')
        
        # Generate realistic-looking sample data
        np.random.seed(42)
        base_price = 50000 if 'BTC' in symbol else 1.2 if 'EUR' in symbol else 2000
        
        data = {
            'timestamp': dates,
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': []
        }
        
        current_price = base_price
        for i in range(limit):
            change = np.random.normal(0, 0.02)  # 2% volatility
            open_price = current_price
            close_price = current_price * (1 + change)
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
            volume = np.random.uniform(100, 1000)
            
            data['open'].append(open_price)
            data['high'].append(high_price)
            data['low'].append(low_price)
            data['close'].append(close_price)
            data['volume'].append(volume)
            
            current_price = close_price
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        self.historical_data[symbol] = df
        return df
    
    def get_live_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get live market data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with current market data
        """
        # Mock live data - in production, this would use websocket connections
        if symbol in self.historical_data:
            last_close = self.historical_data[symbol]['close'].iloc[-1]
            change = np.random.normal(0, 0.001)  # Small price movement
            current_price = last_close * (1 + change)
        else:
            current_price = 50000 if 'BTC' in symbol else 1.2 if 'EUR' in symbol else 2000
        
        live_data = {
            'symbol': symbol,
            'price': current_price,
            'timestamp': datetime.now(),
            'volume': np.random.uniform(50, 200),
            'bid': current_price * 0.999,
            'ask': current_price * 1.001
        }
        
        self.live_data.append(live_data)
        return live_data
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for market data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()
        
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Average
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        
        return df
    
    def engineer_features(self, symbol: str) -> Dict[str, Any]:
        """
        Engineer features for machine learning models.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with engineered features
        """
        if symbol not in self.historical_data:
            self.fetch_historical_data(symbol)
        
        df = self.historical_data[symbol]
        df = self.calculate_technical_indicators(df)
        
        # Get latest values for features
        latest_idx = -1
        features = {
            'price': df['close'].iloc[latest_idx],
            'price_change': df['close'].pct_change().iloc[latest_idx],
            'volume': df['volume'].iloc[latest_idx],
            'sma_20': df['sma_20'].iloc[latest_idx],
            'sma_50': df['sma_50'].iloc[latest_idx],
            'rsi': df['rsi'].iloc[latest_idx],
            'macd': df['macd'].iloc[latest_idx],
            'macd_signal': df['macd_signal'].iloc[latest_idx],
            'volatility': df['volatility'].iloc[latest_idx],
            'bb_position': (df['close'].iloc[latest_idx] - df['bb_lower'].iloc[latest_idx]) / 
                          (df['bb_upper'].iloc[latest_idx] - df['bb_lower'].iloc[latest_idx])
        }
        
        # Add trend analysis
        features['sma_trend'] = 1 if features['sma_20'] > features['sma_50'] else 0
        features['price_vs_sma'] = features['price'] / features['sma_20'] - 1
        
        # Add momentum indicators
        features['rsi_overbought'] = 1 if features['rsi'] > 70 else 0
        features['rsi_oversold'] = 1 if features['rsi'] < 30 else 0
        
        # Normalize features for ML models
        features['sentiment'] = np.tanh(features['price_change'] * 100)  # Normalize price change
        features['volatility_normalized'] = min(features['volatility'] * 100, 1.0)
        
        self.processed_features[symbol] = features
        return features
    
    def get_market_data_for_prediction(self, symbol: str) -> Dict[str, Any]:
        """
        Get formatted market data for prediction models.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary compatible with existing prediction systems
        """
        features = self.engineer_features(symbol)
        
        # Format for compatibility with existing prediction systems
        market_data = {
            'sentiment': features.get('sentiment', 0.0),
            'volatility': features.get('volatility_normalized', 0.3),
            'price_change': features.get('price_change', 0.0),
            'rsi': features.get('rsi', 50.0),
            'volume': features.get('volume', 100.0),
            'price': features.get('price', 0.0),
            'timestamp': datetime.now()
        }
        
        return market_data
    
    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess raw market data for analysis.
        
        Args:
            data: Raw market data
            
        Returns:
            Preprocessed data
        """
        processed = data.copy()
        
        # Handle missing values
        for key, value in processed.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                processed[key] = 0.0
        
        # Ensure numeric types
        numeric_fields = ['price', 'volume', 'sentiment', 'volatility', 'price_change']
        for field in numeric_fields:
            if field in processed:
                try:
                    processed[field] = float(processed[field])
                except (ValueError, TypeError):
                    processed[field] = 0.0
        
        return processed
    
    def get_historical_features(self, symbol: str, lookback_periods: int = 100) -> np.ndarray:
        """
        Get historical features for sequence-based models.
        
        Args:
            symbol: Trading symbol
            lookback_periods: Number of historical periods to include
            
        Returns:
            Array of historical features
        """
        if symbol not in self.historical_data:
            self.fetch_historical_data(symbol, limit=lookback_periods + 50)
        
        df = self.historical_data[symbol]
        df = self.calculate_technical_indicators(df)
        
        # Select features for sequence model
        feature_columns = ['close', 'volume', 'rsi', 'macd', 'volatility']
        features = df[feature_columns].dropna()
        
        if len(features) < lookback_periods:
            # Pad with zeros if insufficient data
            padding = np.zeros((lookback_periods - len(features), len(feature_columns)))
            features_array = np.vstack([padding, features.values])
        else:
            features_array = features.values[-lookback_periods:]
        
        return features_array