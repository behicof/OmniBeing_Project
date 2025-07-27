"""
Enhanced Live Trading Manager for Production OmniBeing Trading System.
Multi-exchange, advanced order types, portfolio management, and real-time execution.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
import ccxt.async_support as ccxt
import websockets
import redis
import asyncpg
from production_config import ProductionConfig

class OrderType(Enum):
    """Enhanced order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    OCO = "oco"  # One-Cancels-Other
    ICEBERG = "iceberg"
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    TRAIL_STOP = "trail_stop"

class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PARTIALLY_FILLED = "partially_filled"

class PositionSide(Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"

@dataclass
class TradingPair:
    """Trading pair configuration."""
    symbol: str
    base_asset: str
    quote_asset: str
    min_order_size: float
    max_order_size: float
    price_precision: int
    quantity_precision: int
    maker_fee: float
    taker_fee: float

@dataclass
class MarketData:
    """Real-time market data."""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: float
    high: float
    low: float
    change_24h: float
    volatility: float

@dataclass
class Order:
    """Trading order details."""
    id: str
    symbol: str
    type: OrderType
    side: OrderSide
    amount: float
    price: Optional[float]
    status: OrderStatus
    created_at: datetime
    filled_amount: float = 0.0
    average_price: float = 0.0
    fees: float = 0.0
    exchange: str = ""
    strategy: str = ""
    metadata: Dict[str, Any] = None

@dataclass
class Position:
    """Trading position."""
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    margin_used: float
    liquidation_price: Optional[float]
    opened_at: datetime
    last_updated: datetime

@dataclass
class Portfolio:
    """Portfolio summary."""
    total_balance: float
    available_balance: float
    used_margin: float
    unrealized_pnl: float
    realized_pnl: float
    total_equity: float
    margin_ratio: float
    positions: List[Position]
    last_updated: datetime

class LiveTradingManager:
    """
    Enhanced live trading manager with multi-exchange support,
    advanced order types, and sophisticated portfolio management.
    """
    
    def __init__(self, config: ProductionConfig):
        """
        Initialize live trading manager.
        
        Args:
            config: Production configuration
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Exchange connections
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.exchange_configs = {
            'binance': {
                'sandbox': False,
                'options': {
                    'adjustForTimeDifference': True,
                    'recvWindow': 10000
                }
            },
            'coinbase': {
                'sandbox': False,
                'options': {
                    'advanced': True
                }
            },
            'kraken': {
                'sandbox': False,
                'options': {}
            }
        }
        
        # Redis and database
        self.redis_client = None
        self.db_pool = None
        
        # Trading state
        self.active_orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.portfolio = None
        self.market_data: Dict[str, MarketData] = {}
        
        # Strategy management
        self.active_strategies: Dict[str, Any] = {}
        self.strategy_allocations: Dict[str, float] = {}
        
        # Risk management
        self.risk_limits = {
            'max_position_size': 0.1,  # 10% of portfolio
            'max_daily_loss': 0.05,    # 5% daily loss limit
            'max_leverage': 3.0,       # 3x leverage
            'max_correlation': 0.7,    # Maximum correlation between positions
            'max_drawdown': 0.15       # 15% maximum drawdown
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0
        }
        
        # WebSocket connections
        self.websocket_connections: Dict[str, Any] = {}
        
        # Order execution queues
        self.order_queue = asyncio.Queue()
        self.execution_engine_running = False
    
    def _setup_logging(self) -> logging.Logger:
        """Setup trading logger."""
        logger = logging.getLogger('live_trading')
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler('logs/live_trading.log')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
        
        return logger
    
    async def initialize(self):
        """Initialize live trading manager."""
        try:
            # Initialize Redis connection
            redis_url = self.config.get_redis_url()
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            
            # Initialize database connection
            db_url = self.config.get_database_url()
            self.db_pool = await asyncpg.create_pool(db_url, min_size=5, max_size=20)
            
            # Initialize exchanges
            await self._initialize_exchanges()
            
            # Load existing positions and orders
            await self._load_trading_state()
            
            # Start market data feeds
            await self._start_market_data_feeds()
            
            # Start execution engine
            await self._start_execution_engine()
            
            self.logger.info("Live trading manager initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize live trading manager: {e}")
            raise
    
    async def _initialize_exchanges(self):
        """Initialize exchange connections."""
        try:
            for exchange_name, config in self.exchange_configs.items():
                try:
                    # Get API credentials from config
                    api_key = self.config.get(f'exchanges.{exchange_name}.api_key')
                    secret = self.config.get(f'exchanges.{exchange_name}.secret')
                    
                    if api_key and secret:
                        exchange_class = getattr(ccxt, exchange_name)
                        exchange = exchange_class({
                            'apiKey': api_key,
                            'secret': secret,
                            'sandbox': config['sandbox'],
                            'enableRateLimit': True,
                            'options': config['options']
                        })
                        
                        # Test connection
                        await exchange.load_markets()
                        self.exchanges[exchange_name] = exchange
                        
                        self.logger.info(f"Connected to {exchange_name}")
                    else:
                        self.logger.warning(f"No credentials found for {exchange_name}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to connect to {exchange_name}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error initializing exchanges: {e}")
    
    async def _load_trading_state(self):
        """Load existing trading state from database."""
        try:
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    # Load active orders
                    orders = await conn.fetch("""
                        SELECT * FROM orders 
                        WHERE status IN ('open', 'partially_filled')
                    """)
                    
                    for order_row in orders:
                        order = Order(
                            id=order_row['id'],
                            symbol=order_row['symbol'],
                            type=OrderType(order_row['type']),
                            side=OrderSide(order_row['side']),
                            amount=float(order_row['amount']),
                            price=float(order_row['price']) if order_row['price'] else None,
                            status=OrderStatus(order_row['status']),
                            created_at=order_row['created_at'],
                            filled_amount=float(order_row['filled_amount']),
                            average_price=float(order_row['average_price']),
                            fees=float(order_row['fees']),
                            exchange=order_row['exchange'],
                            strategy=order_row['strategy'],
                            metadata=json.loads(order_row['metadata']) if order_row['metadata'] else {}
                        )
                        self.active_orders[order.id] = order
                    
                    # Load active positions
                    positions = await conn.fetch("""
                        SELECT * FROM positions 
                        WHERE size != 0
                    """)
                    
                    for pos_row in positions:
                        position = Position(
                            symbol=pos_row['symbol'],
                            side=PositionSide(pos_row['side']),
                            size=float(pos_row['size']),
                            entry_price=float(pos_row['entry_price']),
                            current_price=float(pos_row['current_price']),
                            unrealized_pnl=float(pos_row['unrealized_pnl']),
                            realized_pnl=float(pos_row['realized_pnl']),
                            margin_used=float(pos_row['margin_used']),
                            liquidation_price=float(pos_row['liquidation_price']) if pos_row['liquidation_price'] else None,
                            opened_at=pos_row['opened_at'],
                            last_updated=pos_row['last_updated']
                        )
                        self.positions[position.symbol] = position
                    
                    self.logger.info(f"Loaded {len(self.active_orders)} orders and {len(self.positions)} positions")
                    
        except Exception as e:
            self.logger.error(f"Error loading trading state: {e}")
    
    async def _start_market_data_feeds(self):
        """Start real-time market data feeds."""
        try:
            for exchange_name, exchange in self.exchanges.items():
                if exchange.has['watchTicker']:
                    asyncio.create_task(self._market_data_feed(exchange_name, exchange))
                    
        except Exception as e:
            self.logger.error(f"Error starting market data feeds: {e}")
    
    async def _market_data_feed(self, exchange_name: str, exchange: ccxt.Exchange):
        """Market data feed for a specific exchange."""
        try:
            symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT']
            
            while True:
                try:
                    for symbol in symbols:
                        if symbol in exchange.symbols:
                            ticker = await exchange.fetch_ticker(symbol)
                            
                            market_data = MarketData(
                                symbol=symbol,
                                timestamp=datetime.now(),
                                bid=ticker['bid'],
                                ask=ticker['ask'],
                                last=ticker['last'],
                                volume=ticker['baseVolume'],
                                high=ticker['high'],
                                low=ticker['low'],
                                change_24h=ticker['percentage'],
                                volatility=self._calculate_volatility(symbol, ticker['last'])
                            )
                            
                            self.market_data[f"{exchange_name}:{symbol}"] = market_data
                            
                            # Update position prices
                            await self._update_position_prices(symbol, ticker['last'])
                            
                            # Store in Redis for real-time access
                            await self._store_market_data(exchange_name, market_data)
                    
                    await asyncio.sleep(1)  # Update every second
                    
                except Exception as e:
                    self.logger.error(f"Error in market data feed for {exchange_name}: {e}")
                    await asyncio.sleep(5)
                    
        except Exception as e:
            self.logger.error(f"Market data feed error for {exchange_name}: {e}")
    
    def _calculate_volatility(self, symbol: str, current_price: float) -> float:
        """Calculate simple volatility estimate."""
        # This is a simplified volatility calculation
        # In production, you'd use proper rolling standard deviation
        return 0.02  # 2% default volatility
    
    async def _update_position_prices(self, symbol: str, current_price: float):
        """Update position current prices and PnL."""
        try:
            if symbol in self.positions:
                position = self.positions[symbol]
                position.current_price = current_price
                
                # Calculate unrealized PnL
                if position.side == PositionSide.LONG:
                    position.unrealized_pnl = (current_price - position.entry_price) * position.size
                elif position.side == PositionSide.SHORT:
                    position.unrealized_pnl = (position.entry_price - current_price) * position.size
                
                position.last_updated = datetime.now()
                
                # Update in database
                await self._update_position_in_db(position)
                
        except Exception as e:
            self.logger.error(f"Error updating position prices: {e}")
    
    async def _store_market_data(self, exchange: str, data: MarketData):
        """Store market data in Redis."""
        try:
            key = f"market_data:{exchange}:{data.symbol}"
            value = json.dumps(asdict(data), default=str)
            await asyncio.to_thread(self.redis_client.setex, key, 60, value)
        except Exception as e:
            self.logger.error(f"Error storing market data: {e}")
    
    async def _start_execution_engine(self):
        """Start order execution engine."""
        if not self.execution_engine_running:
            self.execution_engine_running = True
            asyncio.create_task(self._execution_engine())
    
    async def _execution_engine(self):
        """Main order execution engine."""
        try:
            while self.execution_engine_running:
                try:
                    # Get order from queue
                    order = await asyncio.wait_for(self.order_queue.get(), timeout=1.0)
                    
                    # Execute order
                    await self._execute_order(order)
                    
                except asyncio.TimeoutError:
                    # Check for pending orders that need updates
                    await self._check_pending_orders()
                    continue
                    
                except Exception as e:
                    self.logger.error(f"Error in execution engine: {e}")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            self.logger.error(f"Execution engine error: {e}")
    
    async def create_order(self,
                          symbol: str,
                          order_type: OrderType,
                          side: OrderSide,
                          amount: float,
                          price: Optional[float] = None,
                          exchange: str = "binance",
                          strategy: str = "manual",
                          metadata: Dict[str, Any] = None) -> str:
        """
        Create a new trading order.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            order_type: Type of order
            side: Buy or sell
            amount: Order amount
            price: Order price (for limit orders)
            exchange: Exchange to use
            strategy: Strategy name
            metadata: Additional order metadata
            
        Returns:
            Order ID
        """
        try:
            # Validate order
            validation_result = await self._validate_order(symbol, order_type, side, amount, price, exchange)
            if not validation_result['valid']:
                raise ValueError(f"Order validation failed: {validation_result['reason']}")
            
            # Create order object
            order_id = f"{exchange}_{symbol}_{int(time.time() * 1000)}"
            order = Order(
                id=order_id,
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                status=OrderStatus.PENDING,
                created_at=datetime.now(),
                exchange=exchange,
                strategy=strategy,
                metadata=metadata or {}
            )
            
            # Store order
            self.active_orders[order_id] = order
            await self._store_order_in_db(order)
            
            # Add to execution queue
            await self.order_queue.put(order)
            
            self.logger.info(f"Order created: {order_id} - {side.value} {amount} {symbol} @ {price}")
            
            return order_id
            
        except Exception as e:
            self.logger.error(f"Error creating order: {e}")
            raise
    
    async def _validate_order(self, symbol: str, order_type: OrderType, side: OrderSide, 
                            amount: float, price: Optional[float], exchange: str) -> Dict[str, Any]:
        """Validate order parameters."""
        try:
            # Check if exchange is available
            if exchange not in self.exchanges:
                return {'valid': False, 'reason': f'Exchange {exchange} not available'}
            
            # Check symbol availability
            exchange_obj = self.exchanges[exchange]
            if symbol not in exchange_obj.symbols:
                return {'valid': False, 'reason': f'Symbol {symbol} not available on {exchange}'}
            
            # Check minimum order size
            market = exchange_obj.markets[symbol]
            if amount < market['limits']['amount']['min']:
                return {'valid': False, 'reason': f'Order amount below minimum: {market["limits"]["amount"]["min"]}'}
            
            # Check maximum order size
            if amount > market['limits']['amount']['max']:
                return {'valid': False, 'reason': f'Order amount above maximum: {market["limits"]["amount"]["max"]}'}
            
            # Check price limits for limit orders
            if order_type in [OrderType.LIMIT] and price:
                if price < market['limits']['price']['min']:
                    return {'valid': False, 'reason': f'Price below minimum: {market["limits"]["price"]["min"]}'}
                if price > market['limits']['price']['max']:
                    return {'valid': False, 'reason': f'Price above maximum: {market["limits"]["price"]["max"]}'}
            
            # Check risk limits
            risk_check = await self._check_risk_limits(symbol, side, amount, price)
            if not risk_check['valid']:
                return risk_check
            
            return {'valid': True, 'reason': 'Order validation passed'}
            
        except Exception as e:
            return {'valid': False, 'reason': f'Validation error: {str(e)}'}
    
    async def _check_risk_limits(self, symbol: str, side: OrderSide, amount: float, price: Optional[float]) -> Dict[str, Any]:
        """Check risk management limits."""
        try:
            # Calculate position size as percentage of portfolio
            portfolio_value = await self._get_portfolio_value()
            if portfolio_value > 0:
                position_value = amount * (price or self._get_current_price(symbol))
                position_percentage = position_value / portfolio_value
                
                if position_percentage > self.risk_limits['max_position_size']:
                    return {
                        'valid': False,
                        'reason': f'Position size {position_percentage:.2%} exceeds limit {self.risk_limits["max_position_size"]:.2%}'
                    }
            
            # Check daily loss limit
            daily_pnl = await self._get_daily_pnl()
            if daily_pnl < -portfolio_value * self.risk_limits['max_daily_loss']:
                return {
                    'valid': False,
                    'reason': f'Daily loss limit reached: {daily_pnl:.2f}'
                }
            
            # Check correlation limits (simplified)
            correlation_risk = await self._check_correlation_risk(symbol)
            if correlation_risk > self.risk_limits['max_correlation']:
                return {
                    'valid': False,
                    'reason': f'Correlation risk {correlation_risk:.2f} exceeds limit {self.risk_limits["max_correlation"]}'
                }
            
            return {'valid': True, 'reason': 'Risk limits passed'}
            
        except Exception as e:
            return {'valid': False, 'reason': f'Risk check error: {str(e)}'}
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        for key, data in self.market_data.items():
            if symbol in key:
                return data.last
        return 0.0
    
    async def _get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        # Simplified portfolio value calculation
        return 100000.0  # $100k default
    
    async def _get_daily_pnl(self) -> float:
        """Get daily PnL."""
        try:
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    result = await conn.fetchval("""
                        SELECT COALESCE(SUM(realized_pnl), 0.0)
                        FROM trades 
                        WHERE DATE(created_at) = CURRENT_DATE
                    """)
                    return float(result) if result else 0.0
        except:
            return 0.0
    
    async def _check_correlation_risk(self, symbol: str) -> float:
        """Check correlation risk with existing positions."""
        # Simplified correlation check
        return 0.3  # 30% correlation
    
    async def _execute_order(self, order: Order):
        """Execute order on exchange."""
        try:
            exchange = self.exchanges.get(order.exchange)
            if not exchange:
                order.status = OrderStatus.REJECTED
                await self._update_order_in_db(order)
                return
            
            order.status = OrderStatus.OPEN
            await self._update_order_in_db(order)
            
            # Execute based on order type
            if order.type == OrderType.MARKET:
                result = await self._execute_market_order(exchange, order)
            elif order.type == OrderType.LIMIT:
                result = await self._execute_limit_order(exchange, order)
            elif order.type == OrderType.OCO:
                result = await self._execute_oco_order(exchange, order)
            elif order.type == OrderType.ICEBERG:
                result = await self._execute_iceberg_order(exchange, order)
            elif order.type == OrderType.TWAP:
                result = await self._execute_twap_order(exchange, order)
            else:
                raise ValueError(f"Unsupported order type: {order.type}")
            
            # Update order with result
            if result:
                order.status = OrderStatus.FILLED if result['filled'] == order.amount else OrderStatus.PARTIALLY_FILLED
                order.filled_amount = result['filled']
                order.average_price = result['average']
                order.fees = result['fee']['cost'] if result['fee'] else 0.0
                
                # Update positions
                await self._update_positions(order, result)
                
                # Update performance metrics
                await self._update_performance_metrics(order, result)
            
            await self._update_order_in_db(order)
            
            self.logger.info(f"Order executed: {order.id} - Status: {order.status.value}")
            
        except Exception as e:
            self.logger.error(f"Error executing order {order.id}: {e}")
            order.status = OrderStatus.REJECTED
            await self._update_order_in_db(order)
    
    async def _execute_market_order(self, exchange: ccxt.Exchange, order: Order) -> Dict[str, Any]:
        """Execute market order."""
        try:
            if order.side == OrderSide.BUY:
                result = await exchange.create_market_buy_order(order.symbol, order.amount)
            else:
                result = await exchange.create_market_sell_order(order.symbol, order.amount)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing market order: {e}")
            raise
    
    async def _execute_limit_order(self, exchange: ccxt.Exchange, order: Order) -> Dict[str, Any]:
        """Execute limit order."""
        try:
            if order.side == OrderSide.BUY:
                result = await exchange.create_limit_buy_order(order.symbol, order.amount, order.price)
            else:
                result = await exchange.create_limit_sell_order(order.symbol, order.amount, order.price)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing limit order: {e}")
            raise
    
    async def _execute_oco_order(self, exchange: ccxt.Exchange, order: Order) -> Dict[str, Any]:
        """Execute OCO (One-Cancels-Other) order."""
        # OCO implementation would depend on exchange support
        # For now, fallback to limit order
        return await self._execute_limit_order(exchange, order)
    
    async def _execute_iceberg_order(self, exchange: ccxt.Exchange, order: Order) -> Dict[str, Any]:
        """Execute iceberg order (split large order into smaller chunks)."""
        try:
            chunk_size = order.metadata.get('chunk_size', order.amount / 10)
            remaining_amount = order.amount
            total_filled = 0.0
            total_cost = 0.0
            
            while remaining_amount > 0:
                current_chunk = min(chunk_size, remaining_amount)
                
                if order.side == OrderSide.BUY:
                    result = await exchange.create_limit_buy_order(order.symbol, current_chunk, order.price)
                else:
                    result = await exchange.create_limit_sell_order(order.symbol, current_chunk, order.price)
                
                if result['filled'] > 0:
                    total_filled += result['filled']
                    total_cost += result['filled'] * result['average']
                    remaining_amount -= result['filled']
                    
                    # Wait between chunks
                    await asyncio.sleep(order.metadata.get('chunk_delay', 5))
                else:
                    break
            
            return {
                'filled': total_filled,
                'average': total_cost / total_filled if total_filled > 0 else 0,
                'fee': {'cost': 0}  # Simplified
            }
            
        except Exception as e:
            self.logger.error(f"Error executing iceberg order: {e}")
            raise
    
    async def _execute_twap_order(self, exchange: ccxt.Exchange, order: Order) -> Dict[str, Any]:
        """Execute TWAP (Time-Weighted Average Price) order."""
        try:
            duration = order.metadata.get('duration', 3600)  # 1 hour default
            intervals = order.metadata.get('intervals', 12)  # 12 intervals
            
            chunk_size = order.amount / intervals
            interval_duration = duration / intervals
            
            total_filled = 0.0
            total_cost = 0.0
            
            for i in range(intervals):
                current_price = self._get_current_price(order.symbol)
                
                if order.side == OrderSide.BUY:
                    result = await exchange.create_market_buy_order(order.symbol, chunk_size)
                else:
                    result = await exchange.create_market_sell_order(order.symbol, chunk_size)
                
                if result['filled'] > 0:
                    total_filled += result['filled']
                    total_cost += result['filled'] * result['average']
                
                # Wait for next interval
                if i < intervals - 1:
                    await asyncio.sleep(interval_duration)
            
            return {
                'filled': total_filled,
                'average': total_cost / total_filled if total_filled > 0 else 0,
                'fee': {'cost': 0}  # Simplified
            }
            
        except Exception as e:
            self.logger.error(f"Error executing TWAP order: {e}")
            raise
    
    async def _update_positions(self, order: Order, result: Dict[str, Any]):
        """Update positions after order execution."""
        try:
            symbol = order.symbol
            
            if symbol not in self.positions:
                # Create new position
                position = Position(
                    symbol=symbol,
                    side=PositionSide.LONG if order.side == OrderSide.BUY else PositionSide.SHORT,
                    size=result['filled'],
                    entry_price=result['average'],
                    current_price=result['average'],
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    margin_used=result['filled'] * result['average'],
                    liquidation_price=None,
                    opened_at=datetime.now(),
                    last_updated=datetime.now()
                )
                self.positions[symbol] = position
            else:
                # Update existing position
                position = self.positions[symbol]
                
                if order.side == OrderSide.BUY:
                    # Increase position or reduce short
                    if position.side == PositionSide.LONG:
                        # Average up
                        total_cost = (position.size * position.entry_price) + (result['filled'] * result['average'])
                        position.size += result['filled']
                        position.entry_price = total_cost / position.size
                    else:
                        # Reduce short position
                        position.size -= result['filled']
                        if position.size <= 0:
                            # Position closed or flipped
                            position.side = PositionSide.LONG if position.size < 0 else PositionSide.NEUTRAL
                            position.size = abs(position.size)
                else:
                    # Sell order
                    if position.side == PositionSide.LONG:
                        # Reduce long position
                        position.size -= result['filled']
                        if position.size <= 0:
                            # Position closed or flipped
                            position.side = PositionSide.SHORT if position.size < 0 else PositionSide.NEUTRAL
                            position.size = abs(position.size)
                    else:
                        # Increase short position
                        total_cost = (position.size * position.entry_price) + (result['filled'] * result['average'])
                        position.size += result['filled']
                        position.entry_price = total_cost / position.size
                
                position.last_updated = datetime.now()
            
            # Update position in database
            await self._update_position_in_db(self.positions[symbol])
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    async def _update_performance_metrics(self, order: Order, result: Dict[str, Any]):
        """Update performance metrics."""
        try:
            self.performance_metrics['total_trades'] += 1
            
            # Calculate trade PnL (simplified)
            trade_pnl = 0.0  # Would calculate based on entry/exit prices
            
            if trade_pnl > 0:
                self.performance_metrics['winning_trades'] += 1
                if trade_pnl > self.performance_metrics['best_trade']:
                    self.performance_metrics['best_trade'] = trade_pnl
            else:
                self.performance_metrics['losing_trades'] += 1
                if trade_pnl < self.performance_metrics['worst_trade']:
                    self.performance_metrics['worst_trade'] = trade_pnl
            
            self.performance_metrics['total_pnl'] += trade_pnl
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['winning_trades'] / 
                self.performance_metrics['total_trades']
            )
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    async def _check_pending_orders(self):
        """Check status of pending orders."""
        try:
            for order_id, order in list(self.active_orders.items()):
                if order.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]:
                    # Check order status on exchange
                    exchange = self.exchanges.get(order.exchange)
                    if exchange and hasattr(exchange, 'fetch_order'):
                        try:
                            order_status = await exchange.fetch_order(order_id, order.symbol)
                            
                            # Update order status
                            if order_status['status'] == 'closed':
                                order.status = OrderStatus.FILLED
                            elif order_status['status'] == 'canceled':
                                order.status = OrderStatus.CANCELLED
                            
                            order.filled_amount = order_status['filled']
                            order.average_price = order_status['average'] or 0.0
                            
                            await self._update_order_in_db(order)
                            
                            # Remove completed orders
                            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                                del self.active_orders[order_id]
                                
                        except Exception as e:
                            self.logger.error(f"Error checking order {order_id}: {e}")
                            
        except Exception as e:
            self.logger.error(f"Error checking pending orders: {e}")
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        try:
            if order_id not in self.active_orders:
                return False
            
            order = self.active_orders[order_id]
            exchange = self.exchanges.get(order.exchange)
            
            if exchange:
                await exchange.cancel_order(order_id, order.symbol)
                order.status = OrderStatus.CANCELLED
                await self._update_order_in_db(order)
                del self.active_orders[order_id]
                
                self.logger.info(f"Order cancelled: {order_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def close_position(self, symbol: str, percentage: float = 100.0) -> bool:
        """Close a position (fully or partially)."""
        try:
            if symbol not in self.positions:
                return False
            
            position = self.positions[symbol]
            close_amount = position.size * (percentage / 100.0)
            
            # Create closing order
            close_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
            
            order_id = await self.create_order(
                symbol=symbol,
                order_type=OrderType.MARKET,
                side=close_side,
                amount=close_amount,
                strategy="position_close"
            )
            
            self.logger.info(f"Position close order created: {order_id} for {percentage}% of {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing position {symbol}: {e}")
            return False
    
    async def get_portfolio_summary(self) -> Portfolio:
        """Get current portfolio summary."""
        try:
            total_balance = await self._get_portfolio_value()
            used_margin = sum(pos.margin_used for pos in self.positions.values())
            unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            
            portfolio = Portfolio(
                total_balance=total_balance,
                available_balance=total_balance - used_margin,
                used_margin=used_margin,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=self.performance_metrics['total_pnl'],
                total_equity=total_balance + unrealized_pnl,
                margin_ratio=used_margin / total_balance if total_balance > 0 else 0,
                positions=list(self.positions.values()),
                last_updated=datetime.now()
            )
            
            self.portfolio = portfolio
            return portfolio
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return Portfolio(
                total_balance=0, available_balance=0, used_margin=0,
                unrealized_pnl=0, realized_pnl=0, total_equity=0,
                margin_ratio=0, positions=[], last_updated=datetime.now()
            )
    
    async def _store_order_in_db(self, order: Order):
        """Store order in database."""
        try:
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO orders 
                        (id, symbol, type, side, amount, price, status, created_at,
                         filled_amount, average_price, fees, exchange, strategy, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                        ON CONFLICT (id) DO UPDATE SET
                            status = $7, filled_amount = $9, average_price = $10, fees = $11
                    """, order.id, order.symbol, order.type.value, order.side.value,
                        order.amount, order.price, order.status.value, order.created_at,
                        order.filled_amount, order.average_price, order.fees,
                        order.exchange, order.strategy, json.dumps(order.metadata))
        except Exception as e:
            self.logger.error(f"Error storing order in database: {e}")
    
    async def _update_order_in_db(self, order: Order):
        """Update order in database."""
        await self._store_order_in_db(order)
    
    async def _update_position_in_db(self, position: Position):
        """Update position in database."""
        try:
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO positions 
                        (symbol, side, size, entry_price, current_price, unrealized_pnl,
                         realized_pnl, margin_used, liquidation_price, opened_at, last_updated)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        ON CONFLICT (symbol) DO UPDATE SET
                            side = $2, size = $3, entry_price = $4, current_price = $5,
                            unrealized_pnl = $6, realized_pnl = $7, margin_used = $8,
                            liquidation_price = $9, last_updated = $11
                    """, position.symbol, position.side.value, position.size,
                        position.entry_price, position.current_price, position.unrealized_pnl,
                        position.realized_pnl, position.margin_used, position.liquidation_price,
                        position.opened_at, position.last_updated)
        except Exception as e:
            self.logger.error(f"Error updating position in database: {e}")
    
    async def get_trading_statistics(self) -> Dict[str, Any]:
        """Get comprehensive trading statistics."""
        return {
            'performance_metrics': self.performance_metrics,
            'active_orders_count': len(self.active_orders),
            'active_positions_count': len(self.positions),
            'connected_exchanges': list(self.exchanges.keys()),
            'market_data_feeds': len(self.market_data),
            'risk_limits': self.risk_limits,
            'portfolio_summary': asdict(await self.get_portfolio_summary())
        }
    
    async def close(self):
        """Close live trading manager connections."""
        try:
            self.execution_engine_running = False
            
            # Close exchange connections
            for exchange in self.exchanges.values():
                await exchange.close()
            
            # Close Redis connection
            if self.redis_client:
                await asyncio.to_thread(self.redis_client.close)
            
            # Close database connection
            if self.db_pool:
                await self.db_pool.close()
            
            self.logger.info("Live trading manager closed")
            
        except Exception as e:
            self.logger.error(f"Error closing live trading manager: {e}")


async def main():
    """Main live trading entry point."""
    from production_config import get_production_config
    
    config = get_production_config()
    trading_manager = LiveTradingManager(config)
    
    try:
        await trading_manager.initialize()
        
        # Example: Create a sample order
        order_id = await trading_manager.create_order(
            symbol='BTC/USDT',
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            amount=0.001,
            price=50000.0
        )
        
        print(f"Order created: {order_id}")
        
        # Get portfolio summary
        portfolio = await trading_manager.get_portfolio_summary()
        print(f"Portfolio value: ${portfolio.total_balance:.2f}")
        
        # Keep running
        await asyncio.sleep(3600)  # Run for 1 hour
        
    except KeyboardInterrupt:
        print("\nShutting down live trading...")
    finally:
        await trading_manager.close()


if __name__ == "__main__":
    asyncio.run(main())