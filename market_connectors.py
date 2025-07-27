"""
Market Connectors - Exchange Integration
Handles connections to multiple exchanges, order execution, and portfolio tracking
"""

import asyncio
import logging
import ccxt
import ccxt.pro as ccxtpro
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from decimal import Decimal
import websockets
from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException

from config import get_config

logger = logging.getLogger(__name__)

class MarketConnector:
    """
    Base class for exchange connectors
    """
    
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.connected = False
        self.orders = {}
        self.positions = {}
        
    async def connect(self):
        """Connect to the exchange"""
        raise NotImplementedError
    
    async def disconnect(self):
        """Disconnect from the exchange"""
        raise NotImplementedError
    
    async def get_balance(self) -> Dict[str, float]:
        """Get account balance"""
        raise NotImplementedError
    
    async def place_order(self, symbol: str, side: str, amount: float, 
                         price: Optional[float] = None, order_type: str = 'market') -> Dict[str, Any]:
        """Place an order"""
        raise NotImplementedError
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        raise NotImplementedError
    
    async def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Get order status"""
        raise NotImplementedError

class BinanceConnector(MarketConnector):
    """
    Binance exchange connector
    """
    
    def __init__(self):
        super().__init__('binance')
        self.config = get_config()
        self.client = None
        self.socket_manager = None
        self.ws_connections = {}
        
    async def connect(self):
        """Connect to Binance"""
        try:
            exchange_config = self.config.get_exchange_config()
            
            # Initialize CCXT client
            self.ccxt_client = ccxt.binance({
                'apiKey': exchange_config['apiKey'],
                'secret': exchange_config['secret'],
                'sandbox': exchange_config['sandbox'],
                'enableRateLimit': True,
                'options': exchange_config['options']
            })
            
            # Initialize Binance Python client for WebSocket
            if exchange_config['apiKey'] and exchange_config['secret']:
                self.client = await AsyncClient.create(
                    api_key=exchange_config['apiKey'],
                    api_secret=exchange_config['secret'],
                    testnet=exchange_config['sandbox']
                )
                self.socket_manager = BinanceSocketManager(self.client)
            
            # Test connection
            await self._test_connection()
            
            self.connected = True
            logger.info(f"Connected to Binance (testnet: {exchange_config['sandbox']})")
            
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Binance"""
        try:
            # Close WebSocket connections
            for ws in self.ws_connections.values():
                if hasattr(ws, 'close'):
                    await ws.close()
            
            # Close socket manager
            if self.socket_manager:
                await self.socket_manager.close()
            
            # Close client
            if self.client:
                await self.client.close_connection()
            
            self.connected = False
            logger.info("Disconnected from Binance")
            
        except Exception as e:
            logger.error(f"Error disconnecting from Binance: {e}")
    
    async def _test_connection(self):
        """Test the connection to Binance"""
        try:
            # Test with CCXT
            markets = await asyncio.get_event_loop().run_in_executor(
                None, self.ccxt_client.load_markets
            )
            
            # Test with Binance client if available
            if self.client:
                account = await self.client.get_account()
                logger.info(f"Account status: {account['accountType']}")
            
            logger.info(f"Successfully loaded {len(markets)} markets")
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            raise
    
    async def get_balance(self) -> Dict[str, float]:
        """Get account balance"""
        try:
            if self.client:
                account = await self.client.get_account()
                balances = {}
                
                for balance in account['balances']:
                    asset = balance['asset']
                    free = float(balance['free'])
                    locked = float(balance['locked'])
                    total = free + locked
                    
                    if total > 0:
                        balances[asset] = {
                            'free': free,
                            'locked': locked,
                            'total': total
                        }
                
                return balances
            else:
                # Use CCXT fallback
                balance = await asyncio.get_event_loop().run_in_executor(
                    None, self.ccxt_client.fetch_balance
                )
                return balance
                
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return {}
    
    async def place_order(self, symbol: str, side: str, amount: float, 
                         price: Optional[float] = None, order_type: str = 'market') -> Dict[str, Any]:
        """Place an order on Binance"""
        try:
            if not self.connected:
                raise Exception("Not connected to exchange")
            
            # Use Binance client if available
            if self.client:
                if order_type.lower() == 'market':
                    if side.lower() == 'buy':
                        order = await self.client.order_market_buy(
                            symbol=symbol,
                            quantity=amount
                        )
                    else:
                        order = await self.client.order_market_sell(
                            symbol=symbol,
                            quantity=amount
                        )
                elif order_type.lower() == 'limit':
                    if not price:
                        raise ValueError("Price required for limit orders")
                    
                    if side.lower() == 'buy':
                        order = await self.client.order_limit_buy(
                            symbol=symbol,
                            quantity=amount,
                            price=str(price)
                        )
                    else:
                        order = await self.client.order_limit_sell(
                            symbol=symbol,
                            quantity=amount,
                            price=str(price)
                        )
                else:
                    raise ValueError(f"Unsupported order type: {order_type}")
            
            else:
                # Use CCXT fallback
                order = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    self.ccxt_client.create_order,
                    symbol, order_type, side, amount, price
                )
            
            # Store order
            order_id = order['orderId'] if 'orderId' in order else order['id']
            self.orders[order_id] = order
            
            logger.info(f"Order placed: {side} {amount} {symbol} at {price or 'market'}")
            return order
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        try:
            if self.client:
                result = await self.client.cancel_order(
                    symbol=symbol,
                    orderId=order_id
                )
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.ccxt_client.cancel_order,
                    order_id, symbol
                )
            
            logger.info(f"Order {order_id} cancelled")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Get order status"""
        try:
            if self.client:
                order = await self.client.get_order(
                    symbol=symbol,
                    orderId=order_id
                )
            else:
                order = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.ccxt_client.fetch_order,
                    order_id, symbol
                )
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return {}
    
    async def start_price_stream(self, symbols: List[str], callback):
        """Start real-time price stream"""
        try:
            if not self.socket_manager:
                logger.warning("Socket manager not available")
                return
            
            # Start ticker stream
            ts = self.socket_manager.multiplex_socket(
                [f"{symbol.lower()}@ticker" for symbol in symbols]
            )
            
            async with ts as tscm:
                while True:
                    msg = await tscm.recv()
                    await callback(msg)
                    
        except Exception as e:
            logger.error(f"Price stream error: {e}")
    
    async def get_historical_data(self, symbol: str, interval: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical candlestick data"""
        try:
            if self.client:
                klines = await self.client.get_historical_klines(
                    symbol, interval, f"{limit} hours ago UTC"
                )
                
                candlesticks = []
                for kline in klines:
                    candlesticks.append({
                        'timestamp': kline[0],
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5])
                    })
                
                return candlesticks
            else:
                # Use CCXT
                ohlcv = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.ccxt_client.fetch_ohlcv,
                    symbol, interval, None, limit
                )
                
                candlesticks = []
                for candle in ohlcv:
                    candlesticks.append({
                        'timestamp': candle[0],
                        'open': candle[1],
                        'high': candle[2],
                        'low': candle[3],
                        'close': candle[4],
                        'volume': candle[5]
                    })
                
                return candlesticks
                
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return []

class MultiExchangeConnector:
    """
    Manages connections to multiple exchanges
    """
    
    def __init__(self):
        self.config = get_config()
        self.connectors = {}
        self.primary_exchange = 'binance'
        
        # Initialize connectors
        self.connectors['binance'] = BinanceConnector()
        
    async def connect_all(self):
        """Connect to all configured exchanges"""
        for name, connector in self.connectors.items():
            try:
                await connector.connect()
                logger.info(f"Connected to {name}")
            except Exception as e:
                logger.error(f"Failed to connect to {name}: {e}")
    
    async def disconnect_all(self):
        """Disconnect from all exchanges"""
        for name, connector in self.connectors.items():
            try:
                await connector.disconnect()
                logger.info(f"Disconnected from {name}")
            except Exception as e:
                logger.error(f"Error disconnecting from {name}: {e}")
    
    def get_connector(self, exchange_name: str = None) -> MarketConnector:
        """Get a specific exchange connector"""
        exchange_name = exchange_name or self.primary_exchange
        return self.connectors.get(exchange_name)
    
    async def place_order(self, symbol: str, side: str, amount: float, 
                         price: Optional[float] = None, order_type: str = 'market',
                         exchange: str = None) -> Dict[str, Any]:
        """Place order on specified exchange"""
        connector = self.get_connector(exchange)
        if not connector:
            raise ValueError(f"Exchange {exchange} not available")
        
        return await connector.place_order(symbol, side, amount, price, order_type)
    
    async def get_best_price(self, symbol: str, side: str) -> Dict[str, Any]:
        """Get best price across all connected exchanges"""
        best_price = None
        best_exchange = None
        
        for name, connector in self.connectors.items():
            if not connector.connected:
                continue
            
            try:
                # This would fetch order book and find best price
                # For now, just return from primary exchange
                if name == self.primary_exchange:
                    return {
                        'exchange': name,
                        'price': None,  # Would get actual price
                        'symbol': symbol,
                        'side': side
                    }
            except Exception as e:
                logger.error(f"Error getting price from {name}: {e}")
        
        return None
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary across all exchanges"""
        portfolio = {
            'total_value_usd': 0.0,
            'exchanges': {},
            'assets': {},
            'timestamp': datetime.now()
        }
        
        for name, connector in self.connectors.items():
            if not connector.connected:
                continue
            
            try:
                balance = await connector.get_balance()
                portfolio['exchanges'][name] = balance
                
                # Aggregate assets
                for asset, info in balance.items():
                    if isinstance(info, dict) and 'total' in info:
                        if asset not in portfolio['assets']:
                            portfolio['assets'][asset] = 0
                        portfolio['assets'][asset] += info['total']
                
            except Exception as e:
                logger.error(f"Error getting balance from {name}: {e}")
        
        return portfolio
    
    def get_connection_status(self) -> Dict[str, bool]:
        """Get connection status for all exchanges"""
        return {name: connector.connected for name, connector in self.connectors.items()}

class OrderManager:
    """
    Manages order execution and tracking
    """
    
    def __init__(self, multi_exchange: MultiExchangeConnector):
        self.multi_exchange = multi_exchange
        self.active_orders = {}
        self.order_history = []
        
    async def smart_order_placement(self, symbol: str, side: str, amount: float,
                                  strategy: str = 'best_price') -> Dict[str, Any]:
        """Place order using smart routing"""
        
        if strategy == 'best_price':
            # Find best price across exchanges
            best_price_info = await self.multi_exchange.get_best_price(symbol, side)
            if best_price_info:
                exchange = best_price_info['exchange']
            else:
                exchange = self.multi_exchange.primary_exchange
        else:
            exchange = self.multi_exchange.primary_exchange
        
        # Place the order
        order = await self.multi_exchange.place_order(
            symbol, side, amount, exchange=exchange
        )
        
        # Track the order
        order_id = order.get('orderId', order.get('id'))
        self.active_orders[order_id] = {
            'order': order,
            'exchange': exchange,
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'timestamp': datetime.now()
        }
        
        return order
    
    async def monitor_orders(self):
        """Monitor active orders"""
        while True:
            try:
                for order_id, order_info in list(self.active_orders.items()):
                    connector = self.multi_exchange.get_connector(order_info['exchange'])
                    if connector:
                        status = await connector.get_order_status(
                            order_id, order_info['symbol']
                        )
                        
                        if status and status.get('status') in ['FILLED', 'CANCELED', 'REJECTED']:
                            # Move to history
                            self.order_history.append({
                                **order_info,
                                'final_status': status,
                                'completed_at': datetime.now()
                            })
                            del self.active_orders[order_id]
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring orders: {e}")
                await asyncio.sleep(10)

# Global market connector instance
market_connector = None

def get_market_connector() -> MultiExchangeConnector:
    """Get the global market connector instance"""
    global market_connector
    if market_connector is None:
        market_connector = MultiExchangeConnector()
    return market_connector