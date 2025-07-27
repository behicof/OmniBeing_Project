"""
Market Connectors for the OmniBeing Trading System.
Provides integration with various exchange APIs and real-time data streams.
"""

import asyncio
import websocket
import json
import time
import ccxt
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import threading
from config import config


class ExchangeConnector:
    """Base class for exchange connectors."""
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        """
        Initialize exchange connector.
        
        Args:
            api_key: API key for the exchange
            secret_key: Secret key for the exchange
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.is_connected = False
        self.websocket = None
        self.data_callback: Optional[Callable] = None
        
    def connect(self) -> bool:
        """
        Connect to the exchange.
        
        Returns:
            True if connection successful, False otherwise
        """
        raise NotImplementedError
    
    def disconnect(self):
        """Disconnect from the exchange."""
        raise NotImplementedError
    
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get current market data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with market data
        """
        raise NotImplementedError
    
    def place_order(self, symbol: str, side: str, amount: float, 
                   price: float = None, order_type: str = 'market') -> Dict[str, Any]:
        """
        Place an order on the exchange.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            amount: Order amount
            price: Order price (for limit orders)
            order_type: Order type ('market', 'limit')
            
        Returns:
            Dictionary with order details
        """
        raise NotImplementedError
    
    def get_account_balance(self) -> Dict[str, float]:
        """
        Get account balance.
        
        Returns:
            Dictionary with asset balances
        """
        raise NotImplementedError


class MockExchangeConnector(ExchangeConnector):
    """Mock exchange connector for testing and simulation."""
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        """Initialize mock exchange connector."""
        super().__init__(api_key, secret_key)
        self.mock_balance = {'USD': 10000.0, 'BTC': 0.0, 'ETH': 0.0}
        self.mock_prices = {
            'BTCUSDT': 50000.0,
            'ETHUSDT': 3000.0,
            'XAUUSD': 2000.0
        }
        self.orders = []
        self.order_id_counter = 1
        
    def connect(self) -> bool:
        """Connect to mock exchange."""
        self.is_connected = True
        return True
    
    def disconnect(self):
        """Disconnect from mock exchange."""
        self.is_connected = False
    
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get mock market data."""
        if not self.is_connected:
            raise ConnectionError("Not connected to exchange")
        
        base_price = self.mock_prices.get(symbol, 100.0)
        
        # Simulate small price movements
        import random
        price_change = random.uniform(-0.001, 0.001)
        current_price = base_price * (1 + price_change)
        
        return {
            'symbol': symbol,
            'price': current_price,
            'bid': current_price * 0.999,
            'ask': current_price * 1.001,
            'volume': random.uniform(100, 1000),
            'timestamp': datetime.now()
        }
    
    def place_order(self, symbol: str, side: str, amount: float, 
                   price: float = None, order_type: str = 'market') -> Dict[str, Any]:
        """Place a mock order."""
        if not self.is_connected:
            raise ConnectionError("Not connected to exchange")
        
        market_data = self.get_market_data(symbol)
        execution_price = price if order_type == 'limit' and price else market_data['price']
        
        order = {
            'id': str(self.order_id_counter),
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'price': execution_price,
            'type': order_type,
            'status': 'filled',
            'timestamp': datetime.now()
        }
        
        self.orders.append(order)
        self.order_id_counter += 1
        
        # Update mock balance (simplified)
        if side == 'buy':
            self.mock_balance['USD'] -= amount * execution_price
            base_asset = symbol.replace('USDT', '').replace('USD', '')
            if base_asset not in self.mock_balance:
                self.mock_balance[base_asset] = 0.0
            self.mock_balance[base_asset] += amount
        elif side == 'sell':
            base_asset = symbol.replace('USDT', '').replace('USD', '')
            if base_asset in self.mock_balance:
                self.mock_balance[base_asset] -= amount
                self.mock_balance['USD'] += amount * execution_price
        
        return order
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get mock account balance."""
        if not self.is_connected:
            raise ConnectionError("Not connected to exchange")
        
        return self.mock_balance.copy()


class CCXTExchangeConnector(ExchangeConnector):
    """CCXT-based exchange connector for real trading."""
    
    def __init__(self, exchange_id: str = 'binance', api_key: str = None, secret_key: str = None, sandbox: bool = True):
        """
        Initialize CCXT connector.
        
        Args:
            exchange_id: Exchange identifier (e.g., 'binance', 'kraken')
            api_key: API key for the exchange
            secret_key: Secret key for the exchange
            sandbox: Use sandbox mode for testing
        """
        super().__init__(api_key, secret_key)
        self.exchange_id = exchange_id
        self.sandbox = sandbox
        self.exchange = None
        
    def connect(self) -> bool:
        """Connect to exchange via CCXT."""
        try:
            # Initialize exchange
            exchange_class = getattr(ccxt, self.exchange_id)
            
            self.exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.secret_key,
                'sandbox': self.sandbox,
                'enableRateLimit': True,
            })
            
            # Test connection
            if self.api_key and self.secret_key:
                # Try to fetch balance as a connection test
                self.exchange.fetch_balance()
                self.is_connected = True
                return True
            else:
                # No credentials provided - use public endpoints only
                self.exchange.fetch_ticker('BTC/USDT')  # Test public endpoint
                self.is_connected = True
                return True
                
        except Exception as e:
            print(f"Failed to connect to {self.exchange_id}: {e}")
            # Fall back to mock mode
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Disconnect from exchange."""
        self.is_connected = False
        self.exchange = None
    
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get real market data via CCXT."""
        if not self.is_connected or not self.exchange:
            raise ConnectionError("Not connected to exchange")
        
        try:
            # Fetch ticker data
            ticker = self.exchange.fetch_ticker(symbol)
            
            return {
                'symbol': symbol,
                'price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume'],
                'timestamp': datetime.fromtimestamp(ticker['timestamp'] / 1000)
            }
        except Exception as e:
            print(f"Error fetching market data for {symbol}: {e}")
            raise
    
    def place_order(self, symbol: str, side: str, amount: float, 
                   price: float = None, order_type: str = 'market') -> Dict[str, Any]:
        """Place order via CCXT."""
        if not self.is_connected or not self.exchange:
            raise ConnectionError("Not connected to exchange")
        
        if not self.api_key or not self.secret_key:
            raise ValueError("API credentials required for trading")
        
        try:
            if order_type == 'market':
                order = self.exchange.create_market_order(symbol, side, amount)
            else:
                order = self.exchange.create_limit_order(symbol, side, amount, price)
            
            return {
                'id': order['id'],
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': order.get('price', price),
                'type': order_type,
                'status': order['status'],
                'timestamp': datetime.fromtimestamp(order['timestamp'] / 1000)
            }
        except Exception as e:
            print(f"Error placing order: {e}")
            raise
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balance via CCXT."""
        if not self.is_connected or not self.exchange:
            raise ConnectionError("Not connected to exchange")
        
        if not self.api_key or not self.secret_key:
            raise ValueError("API credentials required for account data")
        
        try:
            balance = self.exchange.fetch_balance()
            return balance['free']  # Return free balances
        except Exception as e:
            print(f"Error fetching balance: {e}")
            raise


class WebSocketDataStream:
    """WebSocket data stream for real-time market data."""
    
    def __init__(self, url: str, symbols: List[str] = None):
        """
        Initialize WebSocket data stream.
        
        Args:
            url: WebSocket URL
            symbols: List of symbols to subscribe to
        """
        self.url = url
        self.symbols = symbols or []
        self.websocket = None
        self.is_connected = False
        self.data_callbacks: List[Callable] = []
        self.thread = None
        
    def add_data_callback(self, callback: Callable):
        """Add a callback function for incoming data."""
        self.data_callbacks.append(callback)
    
    def remove_data_callback(self, callback: Callable):
        """Remove a callback function."""
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)
    
    def connect(self):
        """Connect to WebSocket stream."""
        def on_message(ws, message):
            """Handle incoming WebSocket messages."""
            try:
                data = json.loads(message)
                for callback in self.data_callbacks:
                    callback(data)
            except Exception as e:
                print(f"Error processing WebSocket message: {e}")
        
        def on_error(ws, error):
            """Handle WebSocket errors."""
            print(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            """Handle WebSocket close."""
            print("WebSocket connection closed")
            self.is_connected = False
        
        def on_open(ws):
            """Handle WebSocket open."""
            print("WebSocket connection opened")
            self.is_connected = True
            
            # Subscribe to symbols if any
            if self.symbols:
                for symbol in self.symbols:
                    subscribe_msg = {
                        "method": "SUBSCRIBE",
                        "params": [f"{symbol.lower()}@ticker"],
                        "id": 1
                    }
                    ws.send(json.dumps(subscribe_msg))
        
        self.websocket = websocket.WebSocketApp(
            self.url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Run WebSocket in a separate thread
        self.thread = threading.Thread(target=self.websocket.run_forever, daemon=True)
        self.thread.start()
    
    def disconnect(self):
        """Disconnect from WebSocket stream."""
        if self.websocket:
            self.websocket.close()
        self.is_connected = False


class MarketConnectorManager:
    """Manager for multiple market connectors and data streams."""
    
    def __init__(self):
        """Initialize market connector manager."""
        self.connectors: Dict[str, ExchangeConnector] = {}
        self.data_streams: Dict[str, WebSocketDataStream] = {}
        self.data_callbacks: List[Callable] = []
        
    def add_connector(self, name: str, connector: ExchangeConnector):
        """Add an exchange connector."""
        self.connectors[name] = connector
    
    def remove_connector(self, name: str):
        """Remove an exchange connector."""
        if name in self.connectors:
            self.connectors[name].disconnect()
            del self.connectors[name]
    
    def connect_all(self) -> Dict[str, bool]:
        """Connect to all registered exchanges."""
        results = {}
        for name, connector in self.connectors.items():
            try:
                results[name] = connector.connect()
            except Exception as e:
                print(f"Failed to connect to {name}: {e}")
                results[name] = False
        return results
    
    def disconnect_all(self):
        """Disconnect from all exchanges."""
        for connector in self.connectors.values():
            try:
                connector.disconnect()
            except Exception as e:
                print(f"Error disconnecting: {e}")
        
        for stream in self.data_streams.values():
            try:
                stream.disconnect()
            except Exception as e:
                print(f"Error disconnecting stream: {e}")
    
    def get_market_data(self, symbol: str, exchange: str = None) -> Dict[str, Any]:
        """
        Get market data from specified exchange or best available.
        
        Args:
            symbol: Trading symbol
            exchange: Specific exchange name (optional)
            
        Returns:
            Dictionary with market data
        """
        if exchange and exchange in self.connectors:
            return self.connectors[exchange].get_market_data(symbol)
        
        # Try to get data from any connected exchange
        for name, connector in self.connectors.items():
            if connector.is_connected:
                try:
                    return connector.get_market_data(symbol)
                except Exception as e:
                    print(f"Error getting data from {name}: {e}")
                    continue
        
        raise ConnectionError("No exchange available for market data")
    
    def place_order(self, symbol: str, side: str, amount: float, 
                   exchange: str, price: float = None, 
                   order_type: str = 'market') -> Dict[str, Any]:
        """
        Place an order on specified exchange.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            amount: Order amount
            exchange: Exchange name
            price: Order price (for limit orders)
            order_type: Order type
            
        Returns:
            Dictionary with order details
        """
        if exchange not in self.connectors:
            raise ValueError(f"Exchange {exchange} not found")
        
        connector = self.connectors[exchange]
        if not connector.is_connected:
            raise ConnectionError(f"Not connected to {exchange}")
        
        return connector.place_order(symbol, side, amount, price, order_type)
    
    def add_data_callback(self, callback: Callable):
        """Add a callback for real-time data."""
        self.data_callbacks.append(callback)
    
    def setup_default_connectors(self):
        """Setup default connectors (mock for testing, CCXT for real trading)."""
        # Add mock connector for testing
        mock_connector = MockExchangeConnector()
        self.add_connector('mock', mock_connector)
        
        # Add CCXT-based connector for real trading
        try:
            ccxt_connector = CCXTExchangeConnector(
                exchange_id='binance',
                api_key=config.binance_api_key if config.binance_api_key != 'your_binance_api_key_here' else None,
                secret_key=config.binance_secret_key if config.binance_secret_key != 'your_binance_secret_key_here' else None,
                sandbox=True  # Use sandbox mode by default
            )
            self.add_connector('ccxt_binance', ccxt_connector)
        except Exception as e:
            print(f"Failed to setup CCXT connector: {e}")
        
        # Legacy placeholder (kept for compatibility)
        if config.binance_api_key and config.binance_secret_key:
            print("Note: For full CCXT integration, ensure API keys are properly configured")


# Global market connector manager instance
market_connector_manager = MarketConnectorManager()
market_connector_manager.setup_default_connectors()