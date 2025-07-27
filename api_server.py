"""
REST API Server
FastAPI web server for trading system management and monitoring
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import json

from config import get_config
from main_trading_system import get_trading_system
from data_manager import get_data_manager
from market_connectors import get_market_connector
from enhanced_risk_manager import get_enhanced_risk_manager
from backtesting_engine import get_backtest_engine, MovingAverageCrossStrategy, RSIStrategy
from logging_system import get_logging_system, TradeLogEntry, DecisionLogEntry, PerformanceLogEntry

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class TradeRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    side: str = Field(..., description="Trade side: buy or sell")
    amount: float = Field(..., gt=0, description="Trade amount")
    price: Optional[float] = Field(None, description="Limit price (optional)")
    order_type: str = Field("market", description="Order type: market or limit")

class PositionRequest(BaseModel):
    symbol: str
    size: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class BacktestRequest(BaseModel):
    strategy_name: str = Field(..., description="Strategy name: ma_cross or rsi")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    initial_balance: float = Field(100000, gt=0, description="Initial balance")
    parameters: Dict[str, Any] = Field({}, description="Strategy parameters")

class SystemConfig(BaseModel):
    risk_level: Optional[float] = Field(None, ge=0, le=1)
    max_position_size: Optional[float] = Field(None, ge=0, le=1)
    stop_loss_percentage: Optional[float] = Field(None, ge=0, le=1)
    take_profit_percentage: Optional[float] = Field(None, ge=0, le=1)

class WebSocketManager:
    """Manage WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket client disconnected. Total: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

# Create FastAPI app
app = FastAPI(
    title="OmniBeing Trading System API",
    description="Advanced AI Trading System REST API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
config = get_config()
ws_manager = WebSocketManager()

# Dependency injection
def get_current_config():
    return config

# API Routes

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "OmniBeing Trading System API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now(),
        "endpoints": {
            "system": "/system/status",
            "trading": "/trading/",
            "data": "/data/",
            "risk": "/risk/",
            "backtest": "/backtest/",
            "logs": "/logs/",
            "websocket": "/ws"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        trading_system = get_trading_system()
        data_manager = get_data_manager()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "components": {
                "trading_system": trading_system.running,
                "data_manager": data_manager.running if hasattr(data_manager, 'running') else False,
                "market_connector": True,  # Assume healthy for now
                "risk_manager": True,
                "logging_system": True
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

# System Management Routes

@app.get("/system/status")
async def get_system_status():
    """Get comprehensive system status"""
    try:
        trading_system = get_trading_system()
        data_manager = get_data_manager()
        risk_manager = get_enhanced_risk_manager()
        logging_system = get_logging_system()
        
        return {
            "trading_system": trading_system.get_system_status(),
            "data_manager": data_manager.get_market_summary(),
            "risk_manager": risk_manager.get_risk_report(),
            "logging_stats": logging_system.get_log_statistics(),
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/system/start")
async def start_trading_system(background_tasks: BackgroundTasks):
    """Start the trading system"""
    try:
        trading_system = get_trading_system()
        data_manager = get_data_manager()
        
        # Start components in background
        background_tasks.add_task(trading_system.start)
        background_tasks.add_task(data_manager.start)
        
        return {"message": "Trading system started", "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"Error starting trading system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/system/stop")
async def stop_trading_system():
    """Stop the trading system"""
    try:
        trading_system = get_trading_system()
        data_manager = get_data_manager()
        
        trading_system.stop()
        data_manager.stop()
        
        return {"message": "Trading system stopped", "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"Error stopping trading system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/system/config")
async def update_system_config(config_update: SystemConfig):
    """Update system configuration"""
    try:
        # Update configuration values
        if config_update.risk_level is not None:
            config.DEFAULT_RISK_LEVEL = config_update.risk_level
        
        if config_update.max_position_size is not None:
            config.MAX_POSITION_SIZE = config_update.max_position_size
        
        if config_update.stop_loss_percentage is not None:
            config.STOP_LOSS_PERCENTAGE = config_update.stop_loss_percentage
        
        if config_update.take_profit_percentage is not None:
            config.TAKE_PROFIT_PERCENTAGE = config_update.take_profit_percentage
        
        return {
            "message": "Configuration updated",
            "new_config": {
                "risk_level": config.DEFAULT_RISK_LEVEL,
                "max_position_size": config.MAX_POSITION_SIZE,
                "stop_loss_percentage": config.STOP_LOSS_PERCENTAGE,
                "take_profit_percentage": config.TAKE_PROFIT_PERCENTAGE
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Trading Routes

@app.post("/trading/order")
async def place_order(trade_request: TradeRequest):
    """Place a trading order"""
    try:
        market_connector = get_market_connector()
        
        # Validate request
        if trade_request.side.lower() not in ['buy', 'sell']:
            raise HTTPException(status_code=400, detail="Side must be 'buy' or 'sell'")
        
        if trade_request.order_type.lower() not in ['market', 'limit']:
            raise HTTPException(status_code=400, detail="Order type must be 'market' or 'limit'")
        
        if trade_request.order_type.lower() == 'limit' and not trade_request.price:
            raise HTTPException(status_code=400, detail="Price required for limit orders")
        
        # Place order
        order = await market_connector.place_order(
            symbol=trade_request.symbol,
            side=trade_request.side.lower(),
            amount=trade_request.amount,
            price=trade_request.price,
            order_type=trade_request.order_type.lower()
        )
        
        # Log the trade
        logging_system = get_logging_system()
        trade_log = TradeLogEntry(
            timestamp=datetime.now(),
            symbol=trade_request.symbol,
            action=trade_request.side.upper(),
            quantity=trade_request.amount,
            price=trade_request.price or 0,
            order_id=order.get('orderId', order.get('id', 'unknown')),
            status='PENDING',
            reasoning={"manual_order": True}
        )
        logging_system.log_trade(trade_log)
        
        return {
            "message": "Order placed successfully",
            "order": order,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trading/positions")
async def get_positions():
    """Get current trading positions"""
    try:
        risk_manager = get_enhanced_risk_manager()
        
        positions = {}
        for symbol, position in risk_manager.positions.items():
            positions[symbol] = {
                "symbol": position.symbol,
                "size": position.position_size,
                "risk_amount": position.risk_amount,
                "max_loss_percent": position.max_loss_percent,
                "risk_score": position.current_risk_score,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit
            }
        
        return {
            "positions": positions,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trading/position")
async def add_position(position_request: PositionRequest):
    """Add or update a position"""
    try:
        risk_manager = get_enhanced_risk_manager()
        
        risk_manager.add_position(
            symbol=position_request.symbol,
            size=position_request.size,
            entry_price=position_request.entry_price,
            stop_loss=position_request.stop_loss,
            take_profit=position_request.take_profit
        )
        
        return {
            "message": f"Position added/updated for {position_request.symbol}",
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error adding position: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/trading/position/{symbol}")
async def remove_position(symbol: str):
    """Remove a position"""
    try:
        risk_manager = get_enhanced_risk_manager()
        risk_manager.remove_position(symbol)
        
        return {
            "message": f"Position removed for {symbol}",
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error removing position: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trading/balance")
async def get_balance():
    """Get account balance"""
    try:
        market_connector = get_market_connector()
        binance_connector = market_connector.get_connector('binance')
        
        if binance_connector and binance_connector.connected:
            balance = await binance_connector.get_balance()
            return {
                "balance": balance,
                "timestamp": datetime.now()
            }
        else:
            raise HTTPException(status_code=503, detail="Exchange not connected")
    except Exception as e:
        logger.error(f"Error getting balance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Data Routes

@app.get("/data/prices")
async def get_latest_prices():
    """Get latest prices for all symbols"""
    try:
        data_manager = get_data_manager()
        prices = data_manager.get_latest_prices()
        
        return {
            "prices": prices,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting prices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/history/{symbol}")
async def get_price_history(symbol: str, limit: int = 100):
    """Get price history for a symbol"""
    try:
        data_manager = get_data_manager()
        history = data_manager.get_price_history(symbol, limit)
        
        return {
            "symbol": symbol,
            "history": history,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting price history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/sentiment")
async def get_sentiment():
    """Get latest sentiment analysis"""
    try:
        data_manager = get_data_manager()
        sentiment = data_manager.get_latest_sentiment()
        
        return {
            "sentiment": sentiment,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/indicators/{symbol}")
async def get_technical_indicators(symbol: str):
    """Get technical indicators for a symbol"""
    try:
        data_manager = get_data_manager()
        indicators = data_manager.get_indicators(symbol)
        
        return {
            "symbol": symbol,
            "indicators": indicators,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting indicators: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Risk Management Routes

@app.get("/risk/report")
async def get_risk_report():
    """Get comprehensive risk report"""
    try:
        risk_manager = get_enhanced_risk_manager()
        report = risk_manager.get_risk_report()
        
        return report
    except Exception as e:
        logger.error(f"Error getting risk report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/risk/recommendations")
async def get_risk_recommendations():
    """Get position recommendations"""
    try:
        risk_manager = get_enhanced_risk_manager()
        recommendations = risk_manager.get_position_recommendations()
        
        return {
            "recommendations": recommendations,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/risk/emergency_stop")
async def set_emergency_stop(enabled: bool, reason: str = "Manual trigger"):
    """Enable/disable emergency stop"""
    try:
        risk_manager = get_enhanced_risk_manager()
        risk_manager.set_emergency_stop(enabled, reason)
        
        return {
            "message": f"Emergency stop {'enabled' if enabled else 'disabled'}",
            "reason": reason,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error setting emergency stop: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Backtesting Routes

@app.post("/backtest/run")
async def run_backtest(backtest_request: BacktestRequest):
    """Run a backtest"""
    try:
        backtest_engine = get_backtest_engine()
        
        # Create strategy based on name
        if backtest_request.strategy_name.lower() == 'ma_cross':
            strategy = MovingAverageCrossStrategy(
                fast_period=backtest_request.parameters.get('fast_period', 10),
                slow_period=backtest_request.parameters.get('slow_period', 20)
            )
        elif backtest_request.strategy_name.lower() == 'rsi':
            strategy = RSIStrategy(
                rsi_period=backtest_request.parameters.get('rsi_period', 14),
                oversold=backtest_request.parameters.get('oversold', 30),
                overbought=backtest_request.parameters.get('overbought', 70)
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported strategy")
        
        # Generate sample data (in production, use real historical data)
        import pandas as pd
        import numpy as np
        
        start_date = datetime.strptime(backtest_request.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(backtest_request.end_date, '%Y-%m-%d')
        
        # Generate synthetic price data
        dates = pd.date_range(start_date, end_date, freq='H')
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        
        data = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.uniform(1000, 10000, len(dates))
        }, index=dates)
        
        # Run backtest
        result = backtest_engine.run_backtest(strategy, data, backtest_request.initial_balance)
        
        return {
            "strategy_name": result.strategy_name,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "initial_balance": result.initial_balance,
            "final_balance": result.final_balance,
            "total_return": result.total_return,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "parameters": result.parameters
        }
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Logging Routes

@app.get("/logs/trades")
async def get_trade_logs(limit: int = 100):
    """Get recent trade logs"""
    try:
        logging_system = get_logging_system()
        trades = logging_system.get_recent_trades(limit)
        
        return {
            "trades": trades,
            "count": len(trades),
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting trade logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs/decisions")
async def get_decision_logs(limit: int = 50):
    """Get recent decision logs"""
    try:
        logging_system = get_logging_system()
        decisions = logging_system.get_recent_decisions(limit)
        
        return {
            "decisions": decisions,
            "count": len(decisions),
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting decision logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs/performance")
async def get_performance_logs(hours: int = 24):
    """Get performance history"""
    try:
        logging_system = get_logging_system()
        performance = logging_system.get_performance_history(hours)
        
        return {
            "performance": performance,
            "count": len(performance),
            "hours": hours,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting performance logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Send periodic updates
            await asyncio.sleep(5)
            
            # Get latest data
            data_manager = get_data_manager()
            latest_prices = data_manager.get_latest_prices()
            
            # Send update
            update = {
                "type": "price_update",
                "data": latest_prices,
                "timestamp": datetime.now().isoformat()
            }
            
            await ws_manager.send_personal_message(json.dumps(update, default=str), websocket)
            
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)

# Background task to broadcast updates
async def broadcast_updates():
    """Broadcast updates to all WebSocket clients"""
    while True:
        try:
            await asyncio.sleep(10)  # Broadcast every 10 seconds
            
            if ws_manager.active_connections:
                # Get system status
                trading_system = get_trading_system()
                status = trading_system.get_system_status()
                
                update = {
                    "type": "system_update",
                    "data": status,
                    "timestamp": datetime.now().isoformat()
                }
                
                await ws_manager.broadcast(json.dumps(update, default=str))
                
        except Exception as e:
            logger.error(f"Error broadcasting updates: {e}")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info("Starting Trading System API Server")
    
    # Start background update broadcaster
    asyncio.create_task(broadcast_updates())

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Trading System API Server")
    
    try:
        # Stop trading system components
        trading_system = get_trading_system()
        data_manager = get_data_manager()
        
        trading_system.stop()
        data_manager.stop()
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api_server:app",
        host=config.API_HOST,
        port=config.API_PORT,
        debug=config.API_DEBUG,
        reload=config.API_DEBUG
    )