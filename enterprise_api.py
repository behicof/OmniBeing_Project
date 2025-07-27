"""
Enterprise API Gateway for OmniBeing Trading System.
Production-grade API with GraphQL, REST, rate limiting, and real-time capabilities.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import jwt
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, Request, Response, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.websocket import WebSocketState
from pydantic import BaseModel, Field
import strawberry
from strawberry.fastapi import GraphQLRouter
import redis
import asyncpg
from production_config import ProductionConfig
from security_hardening import SecurityManager, authenticate_user
from live_trading_manager import LiveTradingManager, OrderType, OrderSide
from monitoring_suite import MonitoringSuite

# Pydantic models for API
class OrderRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol (e.g., BTC/USDT)")
    type: str = Field(..., description="Order type (market, limit, stop_loss, etc.)")
    side: str = Field(..., description="Order side (buy, sell)")
    amount: float = Field(..., gt=0, description="Order amount")
    price: Optional[float] = Field(None, description="Order price for limit orders")
    exchange: str = Field("binance", description="Exchange to use")
    strategy: str = Field("manual", description="Strategy name")

class OrderResponse(BaseModel):
    order_id: str
    status: str
    message: str

class PositionResponse(BaseModel):
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    margin_used: float

class PortfolioResponse(BaseModel):
    total_balance: float
    available_balance: float
    unrealized_pnl: float
    realized_pnl: float
    margin_ratio: float
    positions: List[PositionResponse]

class MarketDataResponse(BaseModel):
    symbol: str
    price: float
    bid: float
    ask: float
    volume: float
    change_24h: float
    timestamp: datetime

class SystemStatusResponse(BaseModel):
    status: str
    uptime: float
    version: str
    connected_exchanges: List[str]
    active_orders: int
    active_positions: int

# GraphQL types
@strawberry.type
class TradingOrder:
    id: str
    symbol: str
    type: str
    side: str
    amount: float
    price: Optional[float]
    status: str
    created_at: datetime
    filled_amount: float
    average_price: float

@strawberry.type
class Position:
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float

@strawberry.type
class Portfolio:
    total_balance: float
    available_balance: float
    unrealized_pnl: float
    total_equity: float
    positions: List[Position]

@strawberry.type
class MarketData:
    symbol: str
    price: float
    bid: float
    ask: float
    volume: float
    change_24h: float

# Connection manager for WebSockets
class ConnectionManager:
    """WebSocket connection manager."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept and store WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = {
            'market_data': set(),
            'orders': True,
            'positions': True,
            'portfolio': True
        }
    
    def disconnect(self, client_id: str):
        """Remove WebSocket connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
    
    async def send_personal_message(self, message: dict, client_id: str):
        """Send message to specific client."""
        if client_id in self.active_connections:
            try:
                websocket = self.active_connections[client_id]
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(json.dumps(message))
            except Exception as e:
                # Connection probably closed
                self.disconnect(client_id)
    
    async def broadcast(self, message: dict, subscription_type: str = None):
        """Broadcast message to all connected clients."""
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    # Check subscription preferences
                    if subscription_type and client_id in self.subscriptions:
                        if not self.subscriptions[client_id].get(subscription_type, False):
                            continue
                    
                    await websocket.send_text(json.dumps(message))
                else:
                    disconnected_clients.append(client_id)
            except Exception as e:
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)

class EnterpriseAPI:
    """
    Enterprise-grade API gateway with REST, GraphQL, and WebSocket support.
    """
    
    def __init__(self, config: ProductionConfig):
        """
        Initialize enterprise API.
        
        Args:
            config: Production configuration
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="OmniBeing Trading System API",
            description="Enterprise-grade trading API with REST, GraphQL, and WebSocket support",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Components
        self.security_manager = None
        self.trading_manager = None
        self.monitoring_suite = None
        self.redis_client = None
        self.db_pool = None
        
        # WebSocket manager
        self.connection_manager = ConnectionManager()
        
        # API state
        self.api_stats = {
            'requests_total': 0,
            'requests_success': 0,
            'requests_error': 0,
            'active_connections': 0,
            'start_time': datetime.now()
        }
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
        self._setup_graphql()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup API logger."""
        logger = logging.getLogger('enterprise_api')
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler('logs/api.log')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
        
        return logger
    
    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.security.allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Gzip compression
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Request logging and rate limiting
        @self.app.middleware("http")
        async def process_request(request: Request, call_next):
            start_time = time.time()
            self.api_stats['requests_total'] += 1
            
            try:
                # Rate limiting check
                if self.security_manager:
                    rate_limit_ok = await self.security_manager.check_rate_limit(request)
                    if not rate_limit_ok:
                        self.api_stats['requests_error'] += 1
                        return JSONResponse(
                            status_code=429,
                            content={"error": "Rate limit exceeded"}
                        )
                    
                    # Attack detection
                    attack_detected = not await self.security_manager.detect_attacks(request)
                    if attack_detected:
                        self.api_stats['requests_error'] += 1
                        return JSONResponse(
                            status_code=403,
                            content={"error": "Forbidden"}
                        )
                
                # Process request
                response = await call_next(request)
                
                # Log request
                process_time = time.time() - start_time
                self.logger.info(
                    f"{request.method} {request.url.path} - "
                    f"Status: {response.status_code} - "
                    f"Time: {process_time:.4f}s"
                )
                
                if response.status_code < 400:
                    self.api_stats['requests_success'] += 1
                else:
                    self.api_stats['requests_error'] += 1
                
                # Add performance headers
                response.headers["X-Process-Time"] = str(process_time)
                response.headers["X-API-Version"] = "1.0.0"
                
                return response
                
            except Exception as e:
                self.api_stats['requests_error'] += 1
                self.logger.error(f"Request processing error: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": "Internal server error"}
                )
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", tags=["system"])
        async def root():
            """API root endpoint."""
            return {
                "message": "OmniBeing Trading System API",
                "version": "1.0.0",
                "status": "operational",
                "endpoints": {
                    "docs": "/docs",
                    "graphql": "/graphql",
                    "websocket": "/ws/{client_id}",
                    "health": "/health"
                }
            }
        
        @self.app.get("/health", tags=["system"])
        async def health_check():
            """Health check endpoint."""
            try:
                # Check database connection
                db_healthy = False
                if self.db_pool:
                    async with self.db_pool.acquire() as conn:
                        await conn.fetchval("SELECT 1")
                        db_healthy = True
                
                # Check Redis connection
                redis_healthy = False
                if self.redis_client:
                    await asyncio.to_thread(self.redis_client.ping)
                    redis_healthy = True
                
                # Check trading manager
                trading_healthy = self.trading_manager is not None
                
                overall_health = db_healthy and redis_healthy and trading_healthy
                
                return {
                    "status": "healthy" if overall_health else "unhealthy",
                    "timestamp": datetime.now(),
                    "components": {
                        "database": "healthy" if db_healthy else "unhealthy",
                        "redis": "healthy" if redis_healthy else "unhealthy",
                        "trading_manager": "healthy" if trading_healthy else "unhealthy"
                    }
                }
                
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.now()
                }
        
        @self.app.get("/status", tags=["system"])
        async def system_status(user: dict = Depends(authenticate_user)) -> SystemStatusResponse:
            """Get system status."""
            try:
                uptime = (datetime.now() - self.api_stats['start_time']).total_seconds()
                
                connected_exchanges = []
                active_orders = 0
                active_positions = 0
                
                if self.trading_manager:
                    stats = await self.trading_manager.get_trading_statistics()
                    connected_exchanges = stats.get('connected_exchanges', [])
                    active_orders = stats.get('active_orders_count', 0)
                    active_positions = stats.get('active_positions_count', 0)
                
                return SystemStatusResponse(
                    status="operational",
                    uptime=uptime,
                    version="1.0.0",
                    connected_exchanges=connected_exchanges,
                    active_orders=active_orders,
                    active_positions=active_positions
                )
                
            except Exception as e:
                self.logger.error(f"Error getting system status: {e}")
                raise HTTPException(status_code=500, detail="Failed to get system status")
        
        # Trading endpoints
        @self.app.post("/api/orders", tags=["trading"])
        async def create_order(
            order: OrderRequest,
            user: dict = Depends(authenticate_user)
        ) -> OrderResponse:
            """Create a new trading order."""
            try:
                if not self.trading_manager:
                    raise HTTPException(status_code=503, detail="Trading manager not available")
                
                order_id = await self.trading_manager.create_order(
                    symbol=order.symbol,
                    order_type=OrderType(order.type),
                    side=OrderSide(order.side),
                    amount=order.amount,
                    price=order.price,
                    exchange=order.exchange,
                    strategy=order.strategy
                )
                
                # Broadcast order update
                await self.connection_manager.broadcast({
                    "type": "order_created",
                    "data": {
                        "order_id": order_id,
                        "symbol": order.symbol,
                        "type": order.type,
                        "side": order.side,
                        "amount": order.amount,
                        "price": order.price
                    }
                }, "orders")
                
                return OrderResponse(
                    order_id=order_id,
                    status="created",
                    message="Order created successfully"
                )
                
            except Exception as e:
                self.logger.error(f"Error creating order: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/orders", tags=["trading"])
        async def get_orders(user: dict = Depends(authenticate_user)) -> List[dict]:
            """Get all orders."""
            try:
                if not self.trading_manager:
                    raise HTTPException(status_code=503, detail="Trading manager not available")
                
                # Get orders from trading manager
                orders = []
                for order in self.trading_manager.active_orders.values():
                    orders.append({
                        "id": order.id,
                        "symbol": order.symbol,
                        "type": order.type.value,
                        "side": order.side.value,
                        "amount": order.amount,
                        "price": order.price,
                        "status": order.status.value,
                        "created_at": order.created_at,
                        "filled_amount": order.filled_amount,
                        "average_price": order.average_price
                    })
                
                return orders
                
            except Exception as e:
                self.logger.error(f"Error getting orders: {e}")
                raise HTTPException(status_code=500, detail="Failed to get orders")
        
        @self.app.delete("/api/orders/{order_id}", tags=["trading"])
        async def cancel_order(
            order_id: str,
            user: dict = Depends(authenticate_user)
        ) -> dict:
            """Cancel an order."""
            try:
                if not self.trading_manager:
                    raise HTTPException(status_code=503, detail="Trading manager not available")
                
                success = await self.trading_manager.cancel_order(order_id)
                
                if success:
                    # Broadcast order cancellation
                    await self.connection_manager.broadcast({
                        "type": "order_cancelled",
                        "data": {"order_id": order_id}
                    }, "orders")
                    
                    return {"message": "Order cancelled successfully"}
                else:
                    raise HTTPException(status_code=404, detail="Order not found or cannot be cancelled")
                
            except Exception as e:
                self.logger.error(f"Error cancelling order: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/positions", tags=["trading"])
        async def get_positions(user: dict = Depends(authenticate_user)) -> List[PositionResponse]:
            """Get all positions."""
            try:
                if not self.trading_manager:
                    raise HTTPException(status_code=503, detail="Trading manager not available")
                
                positions = []
                for position in self.trading_manager.positions.values():
                    positions.append(PositionResponse(
                        symbol=position.symbol,
                        side=position.side.value,
                        size=position.size,
                        entry_price=position.entry_price,
                        current_price=position.current_price,
                        unrealized_pnl=position.unrealized_pnl,
                        margin_used=position.margin_used
                    ))
                
                return positions
                
            except Exception as e:
                self.logger.error(f"Error getting positions: {e}")
                raise HTTPException(status_code=500, detail="Failed to get positions")
        
        @self.app.post("/api/positions/{symbol}/close", tags=["trading"])
        async def close_position(
            symbol: str,
            percentage: float = 100.0,
            user: dict = Depends(authenticate_user)
        ) -> dict:
            """Close a position."""
            try:
                if not self.trading_manager:
                    raise HTTPException(status_code=503, detail="Trading manager not available")
                
                success = await self.trading_manager.close_position(symbol, percentage)
                
                if success:
                    # Broadcast position update
                    await self.connection_manager.broadcast({
                        "type": "position_closed",
                        "data": {"symbol": symbol, "percentage": percentage}
                    }, "positions")
                    
                    return {"message": f"Position {symbol} closed ({percentage}%)"}
                else:
                    raise HTTPException(status_code=404, detail="Position not found")
                
            except Exception as e:
                self.logger.error(f"Error closing position: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/portfolio", tags=["trading"])
        async def get_portfolio(user: dict = Depends(authenticate_user)) -> PortfolioResponse:
            """Get portfolio summary."""
            try:
                if not self.trading_manager:
                    raise HTTPException(status_code=503, detail="Trading manager not available")
                
                portfolio = await self.trading_manager.get_portfolio_summary()
                
                positions = [
                    PositionResponse(
                        symbol=pos.symbol,
                        side=pos.side.value,
                        size=pos.size,
                        entry_price=pos.entry_price,
                        current_price=pos.current_price,
                        unrealized_pnl=pos.unrealized_pnl,
                        margin_used=pos.margin_used
                    )
                    for pos in portfolio.positions
                ]
                
                return PortfolioResponse(
                    total_balance=portfolio.total_balance,
                    available_balance=portfolio.available_balance,
                    unrealized_pnl=portfolio.unrealized_pnl,
                    realized_pnl=portfolio.realized_pnl,
                    margin_ratio=portfolio.margin_ratio,
                    positions=positions
                )
                
            except Exception as e:
                self.logger.error(f"Error getting portfolio: {e}")
                raise HTTPException(status_code=500, detail="Failed to get portfolio")
        
        @self.app.get("/api/market/{symbol}", tags=["market"])
        async def get_market_data(
            symbol: str,
            user: dict = Depends(authenticate_user)
        ) -> MarketDataResponse:
            """Get market data for a symbol."""
            try:
                # Get market data from trading manager
                market_data = None
                if self.trading_manager:
                    for key, data in self.trading_manager.market_data.items():
                        if symbol in key:
                            market_data = data
                            break
                
                if not market_data:
                    raise HTTPException(status_code=404, detail="Market data not found")
                
                return MarketDataResponse(
                    symbol=market_data.symbol,
                    price=market_data.last,
                    bid=market_data.bid,
                    ask=market_data.ask,
                    volume=market_data.volume,
                    change_24h=market_data.change_24h,
                    timestamp=market_data.timestamp
                )
                
            except Exception as e:
                self.logger.error(f"Error getting market data: {e}")
                raise HTTPException(status_code=500, detail="Failed to get market data")
        
        @self.app.get("/api/analytics/performance", tags=["analytics"])
        async def get_performance_analytics(user: dict = Depends(authenticate_user)) -> dict:
            """Get performance analytics."""
            try:
                if not self.trading_manager:
                    raise HTTPException(status_code=503, detail="Trading manager not available")
                
                stats = await self.trading_manager.get_trading_statistics()
                return stats
                
            except Exception as e:
                self.logger.error(f"Error getting performance analytics: {e}")
                raise HTTPException(status_code=500, detail="Failed to get analytics")
        
        # WebSocket endpoint
        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            """WebSocket endpoint for real-time data."""
            await self.connection_manager.connect(websocket, client_id)
            self.api_stats['active_connections'] += 1
            
            try:
                while True:
                    # Wait for client messages
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Handle subscription requests
                    if message.get("type") == "subscribe":
                        await self._handle_subscription(client_id, message)
                    elif message.get("type") == "unsubscribe":
                        await self._handle_unsubscription(client_id, message)
                    elif message.get("type") == "ping":
                        await self.connection_manager.send_personal_message(
                            {"type": "pong", "timestamp": datetime.now().isoformat()},
                            client_id
                        )
                    
            except WebSocketDisconnect:
                self.connection_manager.disconnect(client_id)
                self.api_stats['active_connections'] -= 1
                self.logger.info(f"Client {client_id} disconnected")
    
    def _setup_graphql(self):
        """Setup GraphQL endpoint."""
        
        @strawberry.type
        class Query:
            
            @strawberry.field
            async def orders(self, info) -> List[TradingOrder]:
                """Get all trading orders."""
                # Authentication would be handled here
                if not self.trading_manager:
                    return []
                
                orders = []
                for order in self.trading_manager.active_orders.values():
                    orders.append(TradingOrder(
                        id=order.id,
                        symbol=order.symbol,
                        type=order.type.value,
                        side=order.side.value,
                        amount=order.amount,
                        price=order.price,
                        status=order.status.value,
                        created_at=order.created_at,
                        filled_amount=order.filled_amount,
                        average_price=order.average_price
                    ))
                
                return orders
            
            @strawberry.field
            async def positions(self, info) -> List[Position]:
                """Get all positions."""
                if not self.trading_manager:
                    return []
                
                positions = []
                for position in self.trading_manager.positions.values():
                    positions.append(Position(
                        symbol=position.symbol,
                        side=position.side.value,
                        size=position.size,
                        entry_price=position.entry_price,
                        current_price=position.current_price,
                        unrealized_pnl=position.unrealized_pnl,
                        realized_pnl=position.realized_pnl
                    ))
                
                return positions
            
            @strawberry.field
            async def portfolio(self, info) -> Optional[Portfolio]:
                """Get portfolio summary."""
                if not self.trading_manager:
                    return None
                
                portfolio = await self.trading_manager.get_portfolio_summary()
                
                positions = [
                    Position(
                        symbol=pos.symbol,
                        side=pos.side.value,
                        size=pos.size,
                        entry_price=pos.entry_price,
                        current_price=pos.current_price,
                        unrealized_pnl=pos.unrealized_pnl,
                        realized_pnl=pos.realized_pnl
                    )
                    for pos in portfolio.positions
                ]
                
                return Portfolio(
                    total_balance=portfolio.total_balance,
                    available_balance=portfolio.available_balance,
                    unrealized_pnl=portfolio.unrealized_pnl,
                    total_equity=portfolio.total_equity,
                    positions=positions
                )
        
        schema = strawberry.Schema(query=Query)
        graphql_app = GraphQLRouter(schema)
        self.app.include_router(graphql_app, prefix="/graphql")
    
    async def _handle_subscription(self, client_id: str, message: dict):
        """Handle WebSocket subscription."""
        subscription_type = message.get("subscription")
        
        if client_id not in self.connection_manager.subscriptions:
            return
        
        if subscription_type == "market_data":
            symbol = message.get("symbol")
            if symbol:
                self.connection_manager.subscriptions[client_id]["market_data"].add(symbol)
        elif subscription_type in ["orders", "positions", "portfolio"]:
            self.connection_manager.subscriptions[client_id][subscription_type] = True
        
        await self.connection_manager.send_personal_message(
            {"type": "subscription_confirmed", "subscription": subscription_type},
            client_id
        )
    
    async def _handle_unsubscription(self, client_id: str, message: dict):
        """Handle WebSocket unsubscription."""
        subscription_type = message.get("subscription")
        
        if client_id not in self.connection_manager.subscriptions:
            return
        
        if subscription_type == "market_data":
            symbol = message.get("symbol")
            if symbol and symbol in self.connection_manager.subscriptions[client_id]["market_data"]:
                self.connection_manager.subscriptions[client_id]["market_data"].remove(symbol)
        elif subscription_type in ["orders", "positions", "portfolio"]:
            self.connection_manager.subscriptions[client_id][subscription_type] = False
        
        await self.connection_manager.send_personal_message(
            {"type": "unsubscription_confirmed", "subscription": subscription_type},
            client_id
        )
    
    async def initialize(self):
        """Initialize enterprise API."""
        try:
            # Initialize Redis connection
            redis_url = self.config.get_redis_url()
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            
            # Initialize database connection
            db_url = self.config.get_database_url()
            self.db_pool = await asyncpg.create_pool(db_url, min_size=2, max_size=10)
            
            # Initialize security manager
            from security_hardening import SecurityManager
            self.security_manager = SecurityManager(self.config)
            await self.security_manager.initialize()
            
            # Initialize trading manager
            self.trading_manager = LiveTradingManager(self.config)
            await self.trading_manager.initialize()
            
            # Initialize monitoring suite
            self.monitoring_suite = MonitoringSuite(self.config)
            await self.monitoring_suite.initialize()
            
            # Start real-time broadcasting
            asyncio.create_task(self._real_time_broadcast_loop())
            
            self.logger.info("Enterprise API initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize enterprise API: {e}")
            raise
    
    async def _real_time_broadcast_loop(self):
        """Broadcast real-time updates to WebSocket clients."""
        while True:
            try:
                # Broadcast market data updates
                if self.trading_manager:
                    for key, market_data in self.trading_manager.market_data.items():
                        await self.connection_manager.broadcast({
                            "type": "market_data_update",
                            "data": {
                                "symbol": market_data.symbol,
                                "price": market_data.last,
                                "bid": market_data.bid,
                                "ask": market_data.ask,
                                "timestamp": market_data.timestamp.isoformat()
                            }
                        }, "market_data")
                
                # Broadcast portfolio updates
                if self.trading_manager:
                    portfolio = await self.trading_manager.get_portfolio_summary()
                    await self.connection_manager.broadcast({
                        "type": "portfolio_update",
                        "data": {
                            "total_balance": portfolio.total_balance,
                            "unrealized_pnl": portfolio.unrealized_pnl,
                            "total_equity": portfolio.total_equity,
                            "margin_ratio": portfolio.margin_ratio
                        }
                    }, "portfolio")
                
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                self.logger.error(f"Error in real-time broadcast: {e}")
                await asyncio.sleep(5)
    
    def get_app(self) -> FastAPI:
        """Get FastAPI application."""
        return self.app
    
    async def close(self):
        """Close enterprise API connections."""
        try:
            if self.trading_manager:
                await self.trading_manager.close()
            
            if self.security_manager:
                await self.security_manager.close()
            
            if self.monitoring_suite:
                await self.monitoring_suite.close()
            
            if self.redis_client:
                await asyncio.to_thread(self.redis_client.close)
            
            if self.db_pool:
                await self.db_pool.close()
            
            self.logger.info("Enterprise API closed")
            
        except Exception as e:
            self.logger.error(f"Error closing enterprise API: {e}")


# Create global API instance
enterprise_api = None

async def get_enterprise_api() -> EnterpriseAPI:
    """Get the global enterprise API instance."""
    global enterprise_api
    if enterprise_api is None:
        from production_config import get_production_config
        config = get_production_config()
        enterprise_api = EnterpriseAPI(config)
        await enterprise_api.initialize()
    return enterprise_api


async def main():
    """Main API entry point."""
    from production_config import get_production_config
    import uvicorn
    
    config = get_production_config()
    api = EnterpriseAPI(config)
    
    try:
        await api.initialize()
        
        # Run with uvicorn
        uvicorn_config = uvicorn.Config(
            app=api.get_app(),
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )
        
        server = uvicorn.Server(uvicorn_config)
        await server.serve()
        
    except KeyboardInterrupt:
        print("\nShutting down API server...")
    finally:
        await api.close()


if __name__ == "__main__":
    asyncio.run(main())