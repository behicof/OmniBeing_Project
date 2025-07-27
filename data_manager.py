"""
Real-time Data Manager
Handles live price feeds, sentiment analysis, news impact, and technical indicators
"""

import asyncio
import logging
import json
import websockets
import aiohttp
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import deque
import redis

from config import get_config

logger = logging.getLogger(__name__)

class DataManager:
    """
    Manages real-time data streams from multiple sources
    """
    
    def __init__(self):
        self.config = get_config()
        self.running = False
        self.subscribers = []
        self.data_cache = {}
        
        # Redis for data caching
        try:
            self.redis_client = redis.Redis(
                host=self.config.REDIS_HOST,
                port=self.config.REDIS_PORT,
                db=self.config.REDIS_DB,
                decode_responses=True
            )
            self.redis_available = True
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.redis_available = False
        
        # Data buffers
        self.price_buffers = {symbol: deque(maxlen=1000) for symbol in self.config.SYMBOLS}
        self.volume_buffers = {symbol: deque(maxlen=1000) for symbol in self.config.SYMBOLS}
        self.sentiment_buffer = deque(maxlen=100)
        self.news_buffer = deque(maxlen=50)
        
        # Technical indicators cache
        self.indicators_cache = {}
        
        # WebSocket connections
        self.ws_connections = {}
        
        logger.info("Data Manager initialized")
    
    def subscribe(self, callback: Callable[[Dict[str, Any]], None]):
        """Subscribe to real-time data updates"""
        self.subscribers.append(callback)
        logger.info(f"New subscriber added. Total subscribers: {len(self.subscribers)}")
    
    def unsubscribe(self, callback: Callable[[Dict[str, Any]], None]):
        """Unsubscribe from data updates"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            logger.info(f"Subscriber removed. Total subscribers: {len(self.subscribers)}")
    
    async def start(self):
        """Start all data feeds"""
        if self.running:
            logger.warning("Data Manager is already running")
            return
        
        logger.info("Starting Data Manager...")
        self.running = True
        
        # Start data collection tasks
        tasks = [
            self._price_feed_task(),
            self._sentiment_feed_task(),
            self._news_feed_task(),
            self._technical_indicators_task(),
            self._data_cleanup_task()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def stop(self):
        """Stop data collection"""
        logger.info("Stopping Data Manager...")
        self.running = False
        
        # Close WebSocket connections
        for ws in self.ws_connections.values():
            if ws and not ws.closed:
                asyncio.create_task(ws.close())
    
    async def _price_feed_task(self):
        """Real-time price data collection"""
        while self.running:
            try:
                # For production, this would connect to Binance WebSocket
                # For now, simulate price data
                await self._simulate_price_data()
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in price feed: {e}")
                await asyncio.sleep(5)
    
    async def _simulate_price_data(self):
        """Simulate real-time price data (for development)"""
        current_time = datetime.now()
        
        for symbol in self.config.SYMBOLS:
            # Get last price or set initial
            last_prices = list(self.price_buffers[symbol])
            if last_prices:
                last_price = last_prices[-1]['price']
            else:
                # Initial prices
                initial_prices = {
                    'BTCUSDT': 45000,
                    'ETHUSDT': 3000,
                    'ADAUSDT': 0.5
                }
                last_price = initial_prices.get(symbol, 100)
            
            # Simulate price movement
            change_percent = np.random.normal(0, 0.001)  # 0.1% volatility
            new_price = last_price * (1 + change_percent)
            volume = np.random.uniform(1000, 10000)
            
            price_data = {
                'symbol': symbol,
                'price': new_price,
                'volume': volume,
                'timestamp': current_time,
                'change_24h': np.random.uniform(-5, 5),
                'high_24h': new_price * 1.02,
                'low_24h': new_price * 0.98
            }
            
            # Add to buffers
            self.price_buffers[symbol].append(price_data)
            self.volume_buffers[symbol].append({
                'symbol': symbol,
                'volume': volume,
                'timestamp': current_time
            })
            
            # Cache in Redis if available
            if self.redis_available:
                try:
                    self.redis_client.setex(
                        f"price:{symbol}",
                        60,  # 1 minute expiry
                        json.dumps(price_data, default=str)
                    )
                except Exception as e:
                    logger.warning(f"Redis cache error: {e}")
        
        # Notify subscribers
        await self._notify_subscribers('price_update', {
            'type': 'price_update',
            'data': {symbol: list(buffer)[-1] for symbol, buffer in self.price_buffers.items() if buffer},
            'timestamp': current_time
        })
    
    async def _sentiment_feed_task(self):
        """Real-time sentiment analysis"""
        while self.running:
            try:
                # Simulate sentiment analysis
                # In production, this would analyze social media, news, etc.
                sentiment_data = await self._analyze_sentiment()
                
                if sentiment_data:
                    self.sentiment_buffer.append(sentiment_data)
                    
                    # Notify subscribers
                    await self._notify_subscribers('sentiment_update', {
                        'type': 'sentiment_update',
                        'data': sentiment_data,
                        'timestamp': datetime.now()
                    })
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in sentiment feed: {e}")
                await asyncio.sleep(60)
    
    async def _analyze_sentiment(self) -> Optional[Dict[str, Any]]:
        """Analyze market sentiment from multiple sources"""
        # Simulate sentiment analysis
        sentiment_score = np.random.uniform(-1, 1)
        confidence = np.random.uniform(0.6, 0.95)
        
        sentiment_data = {
            'overall_sentiment': sentiment_score,
            'confidence': confidence,
            'sources': {
                'twitter': np.random.uniform(-1, 1),
                'reddit': np.random.uniform(-1, 1),
                'news': np.random.uniform(-1, 1),
                'technical': np.random.uniform(-1, 1)
            },
            'trend': 'bullish' if sentiment_score > 0.2 else 'bearish' if sentiment_score < -0.2 else 'neutral',
            'timestamp': datetime.now()
        }
        
        return sentiment_data
    
    async def _news_feed_task(self):
        """Real-time news impact analysis"""
        while self.running:
            try:
                news_data = await self._fetch_news_impact()
                
                if news_data:
                    self.news_buffer.append(news_data)
                    
                    # Notify subscribers
                    await self._notify_subscribers('news_update', {
                        'type': 'news_update',
                        'data': news_data,
                        'timestamp': datetime.now()
                    })
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in news feed: {e}")
                await asyncio.sleep(300)
    
    async def _fetch_news_impact(self) -> Optional[Dict[str, Any]]:
        """Fetch and analyze news impact"""
        # Simulate news impact analysis
        impact_score = np.random.uniform(0, 1)
        
        news_data = {
            'impact_score': impact_score,
            'sentiment': np.random.choice(['positive', 'negative', 'neutral']),
            'category': np.random.choice(['regulatory', 'adoption', 'technical', 'market']),
            'urgency': 'high' if impact_score > 0.7 else 'medium' if impact_score > 0.4 else 'low',
            'summary': f"Simulated news with impact score {impact_score:.2f}",
            'timestamp': datetime.now()
        }
        
        return news_data
    
    async def _technical_indicators_task(self):
        """Calculate technical indicators"""
        while self.running:
            try:
                for symbol in self.config.SYMBOLS:
                    indicators = await self._calculate_indicators(symbol)
                    if indicators:
                        self.indicators_cache[symbol] = indicators
                
                # Notify subscribers
                await self._notify_subscribers('indicators_update', {
                    'type': 'indicators_update',
                    'data': self.indicators_cache,
                    'timestamp': datetime.now()
                })
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error calculating indicators: {e}")
                await asyncio.sleep(60)
    
    async def _calculate_indicators(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Calculate technical indicators for a symbol"""
        prices = [item['price'] for item in self.price_buffers[symbol]]
        volumes = [item['volume'] for item in self.volume_buffers[symbol]]
        
        if len(prices) < 20:
            return None
        
        # Convert to pandas for easier calculation
        df = pd.DataFrame({
            'price': prices,
            'volume': volumes
        })
        
        indicators = {}
        
        try:
            # Simple Moving Averages
            indicators['sma_20'] = df['price'].rolling(20).mean().iloc[-1]
            indicators['sma_50'] = df['price'].rolling(50).mean().iloc[-1] if len(df) >= 50 else None
            
            # RSI (simplified)
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs.iloc[-1]))
            
            # MACD (simplified)
            ema_12 = df['price'].ewm(span=12).mean()
            ema_26 = df['price'].ewm(span=26).mean()
            indicators['macd'] = ema_12.iloc[-1] - ema_26.iloc[-1]
            indicators['macd_signal'] = (ema_12 - ema_26).ewm(span=9).mean().iloc[-1]
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            sma = df['price'].rolling(bb_period).mean()
            std = df['price'].rolling(bb_period).std()
            indicators['bb_upper'] = (sma + (std * bb_std)).iloc[-1]
            indicators['bb_middle'] = sma.iloc[-1]
            indicators['bb_lower'] = (sma - (std * bb_std)).iloc[-1]
            
            # Volume indicators
            indicators['volume_sma'] = df['volume'].rolling(20).mean().iloc[-1]
            indicators['volume_ratio'] = df['volume'].iloc[-1] / indicators['volume_sma']
            
            # Volatility
            indicators['volatility'] = df['price'].pct_change().std() * np.sqrt(24 * 60)  # Annualized
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            return None
        
        return indicators
    
    async def _data_cleanup_task(self):
        """Clean up old data periodically"""
        while self.running:
            try:
                current_time = datetime.now()
                cutoff_time = current_time - timedelta(hours=24)
                
                # Clean sentiment buffer
                self.sentiment_buffer = deque(
                    [item for item in self.sentiment_buffer 
                     if item['timestamp'] > cutoff_time],
                    maxlen=100
                )
                
                # Clean news buffer
                self.news_buffer = deque(
                    [item for item in self.news_buffer 
                     if item['timestamp'] > cutoff_time],
                    maxlen=50
                )
                
                logger.debug("Data cleanup completed")
                await asyncio.sleep(3600)  # Clean every hour
                
            except Exception as e:
                logger.error(f"Error in data cleanup: {e}")
                await asyncio.sleep(3600)
    
    async def _notify_subscribers(self, event_type: str, data: Dict[str, Any]):
        """Notify all subscribers of data updates"""
        if not self.subscribers:
            return
        
        for callback in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    def get_latest_prices(self) -> Dict[str, Any]:
        """Get latest prices for all symbols"""
        latest_prices = {}
        for symbol, buffer in self.price_buffers.items():
            if buffer:
                latest_prices[symbol] = buffer[-1]
        return latest_prices
    
    def get_price_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get price history for a symbol"""
        if symbol in self.price_buffers:
            return list(self.price_buffers[symbol])[-limit:]
        return []
    
    def get_latest_sentiment(self) -> Optional[Dict[str, Any]]:
        """Get latest sentiment analysis"""
        return self.sentiment_buffer[-1] if self.sentiment_buffer else None
    
    def get_latest_news(self) -> Optional[Dict[str, Any]]:
        """Get latest news impact"""
        return self.news_buffer[-1] if self.news_buffer else None
    
    def get_indicators(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get technical indicators for a symbol"""
        return self.indicators_cache.get(symbol)
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get comprehensive market summary"""
        summary = {
            'timestamp': datetime.now(),
            'prices': self.get_latest_prices(),
            'sentiment': self.get_latest_sentiment(),
            'news': self.get_latest_news(),
            'indicators': self.indicators_cache,
            'data_status': {
                'price_feeds': {symbol: len(buffer) for symbol, buffer in self.price_buffers.items()},
                'sentiment_entries': len(self.sentiment_buffer),
                'news_entries': len(self.news_buffer),
                'redis_available': self.redis_available
            }
        }
        
        return summary

# Global data manager instance
data_manager = None

def get_data_manager() -> DataManager:
    """Get the global data manager instance"""
    global data_manager
    if data_manager is None:
        data_manager = DataManager()
    return data_manager