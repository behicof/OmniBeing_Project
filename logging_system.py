"""
Logging and Monitoring System for the OmniBeing Trading System.
Provides comprehensive logging, performance monitoring, and system health tracking.
"""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import threading
import time
from dataclasses import dataclass, asdict
from config import config


@dataclass
class TradeLog:
    """Data class for trade logging."""
    timestamp: datetime
    symbol: str
    action: str
    price: float
    quantity: float
    order_id: str
    execution_time: float
    commission: float = 0.0
    pnl: Optional[float] = None
    strategy: str = "default"
    metadata: Dict[str, Any] = None


@dataclass
class PerformanceMetric:
    """Data class for performance metrics."""
    timestamp: datetime
    metric_name: str
    value: float
    category: str = "trading"
    tags: Dict[str, str] = None


@dataclass
class SystemHealthMetric:
    """Data class for system health metrics."""
    timestamp: datetime
    component: str
    status: str  # 'healthy', 'warning', 'error'
    message: str
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    response_time: Optional[float] = None


class TradeLogger:
    """Logger specifically for trading activities."""
    
    def __init__(self, log_directory: str = "logs"):
        """
        Initialize trade logger.
        
        Args:
            log_directory: Directory to store log files
        """
        self.log_directory = log_directory
        self.trade_logs: List[TradeLog] = []
        self._ensure_log_directory()
        
        # Setup file logger
        self.logger = logging.getLogger('TradeLogger')
        self.logger.setLevel(logging.INFO)
        
        # File handler for trade logs
        trade_log_file = os.path.join(log_directory, f"trades_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(trade_log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
    
    def _ensure_log_directory(self):
        """Ensure log directory exists."""
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)
    
    def log_trade(self, trade_log: TradeLog):
        """
        Log a trade.
        
        Args:
            trade_log: TradeLog object with trade details
        """
        # Add to memory storage
        self.trade_logs.append(trade_log)
        
        # Log to file
        trade_data = asdict(trade_log)
        trade_data['timestamp'] = trade_log.timestamp.isoformat()
        
        self.logger.info(f"TRADE: {json.dumps(trade_data)}")
    
    def log_order(self, symbol: str, action: str, price: float, quantity: float, 
                 order_id: str, execution_time: float, **kwargs):
        """
        Log an order execution.
        
        Args:
            symbol: Trading symbol
            action: Trade action ('buy', 'sell')
            price: Execution price
            quantity: Order quantity
            order_id: Order ID
            execution_time: Time taken to execute order
            **kwargs: Additional metadata
        """
        trade_log = TradeLog(
            timestamp=datetime.now(),
            symbol=symbol,
            action=action,
            price=price,
            quantity=quantity,
            order_id=order_id,
            execution_time=execution_time,
            commission=kwargs.get('commission', 0.0),
            pnl=kwargs.get('pnl'),
            strategy=kwargs.get('strategy', 'default'),
            metadata=kwargs
        )
        
        self.log_trade(trade_log)
    
    def get_trades_by_symbol(self, symbol: str) -> List[TradeLog]:
        """Get all trades for a specific symbol."""
        return [trade for trade in self.trade_logs if trade.symbol == symbol]
    
    def get_trades_by_date(self, date: datetime) -> List[TradeLog]:
        """Get all trades for a specific date."""
        target_date = date.date()
        return [trade for trade in self.trade_logs if trade.timestamp.date() == target_date]
    
    def get_recent_trades(self, hours: int = 24) -> List[TradeLog]:
        """Get trades from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [trade for trade in self.trade_logs if trade.timestamp >= cutoff_time]


class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self, log_directory: str = "logs"):
        """
        Initialize performance monitor.
        
        Args:
            log_directory: Directory to store log files
        """
        self.log_directory = log_directory
        self.performance_metrics: List[PerformanceMetric] = []
        self._ensure_log_directory()
        
        # Setup logger
        self.logger = logging.getLogger('PerformanceMonitor')
        self.logger.setLevel(logging.INFO)
        
        # File handler for performance logs
        perf_log_file = os.path.join(log_directory, f"performance_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(perf_log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
    
    def _ensure_log_directory(self):
        """Ensure log directory exists."""
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)
    
    def log_metric(self, metric_name: str, value: float, category: str = "trading", 
                  tags: Dict[str, str] = None):
        """
        Log a performance metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            category: Metric category
            tags: Additional tags for the metric
        """
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_name=metric_name,
            value=value,
            category=category,
            tags=tags or {}
        )
        
        self.performance_metrics.append(metric)
        
        # Log to file
        metric_data = asdict(metric)
        metric_data['timestamp'] = metric.timestamp.isoformat()
        
        self.logger.info(f"METRIC: {json.dumps(metric_data)}")
    
    def log_trade_performance(self, symbol: str, pnl: float, win_rate: float, 
                            total_trades: int):
        """Log trading performance metrics."""
        self.log_metric("pnl", pnl, "trading", {"symbol": symbol})
        self.log_metric("win_rate", win_rate, "trading", {"symbol": symbol})
        self.log_metric("total_trades", total_trades, "trading", {"symbol": symbol})
    
    def log_execution_time(self, operation: str, execution_time: float):
        """Log operation execution time."""
        self.log_metric("execution_time", execution_time, "performance", {"operation": operation})
    
    def log_prediction_accuracy(self, model_name: str, accuracy: float):
        """Log prediction model accuracy."""
        self.log_metric("prediction_accuracy", accuracy, "ml", {"model": model_name})
    
    def get_metrics_by_category(self, category: str, hours: int = 24) -> List[PerformanceMetric]:
        """Get metrics by category from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            metric for metric in self.performance_metrics 
            if metric.category == category and metric.timestamp >= cutoff_time
        ]


class SystemHealthMonitor:
    """Monitor system health and component status."""
    
    def __init__(self, log_directory: str = "logs"):
        """
        Initialize system health monitor.
        
        Args:
            log_directory: Directory to store log files
        """
        self.log_directory = log_directory
        self.health_metrics: List[SystemHealthMetric] = []
        self.component_status: Dict[str, str] = {}
        self._ensure_log_directory()
        
        # Setup logger
        self.logger = logging.getLogger('SystemHealthMonitor')
        self.logger.setLevel(logging.INFO)
        
        # File handler for health logs
        health_log_file = os.path.join(log_directory, f"health_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(health_log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
        
        # Start health monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
    
    def _ensure_log_directory(self):
        """Ensure log directory exists."""
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)
    
    def log_health_metric(self, component: str, status: str, message: str, 
                         cpu_usage: float = None, memory_usage: float = None, 
                         response_time: float = None):
        """
        Log a system health metric.
        
        Args:
            component: Component name
            status: Component status ('healthy', 'warning', 'error')
            message: Status message
            cpu_usage: CPU usage percentage
            memory_usage: Memory usage percentage
            response_time: Component response time
        """
        health_metric = SystemHealthMetric(
            timestamp=datetime.now(),
            component=component,
            status=status,
            message=message,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            response_time=response_time
        )
        
        self.health_metrics.append(health_metric)
        self.component_status[component] = status
        
        # Log to file
        health_data = asdict(health_metric)
        health_data['timestamp'] = health_metric.timestamp.isoformat()
        
        log_level = logging.ERROR if status == 'error' else logging.WARNING if status == 'warning' else logging.INFO
        self.logger.log(log_level, f"HEALTH: {json.dumps(health_data)}")
    
    def check_component_health(self, component: str, check_function: callable):
        """
        Check health of a specific component.
        
        Args:
            component: Component name
            check_function: Function that returns (status, message, metrics)
        """
        try:
            start_time = time.time()
            status, message, metrics = check_function()
            response_time = time.time() - start_time
            
            self.log_health_metric(
                component=component,
                status=status,
                message=message,
                response_time=response_time,
                **metrics
            )
            
        except Exception as e:
            self.log_health_metric(
                component=component,
                status='error',
                message=f"Health check failed: {str(e)}"
            )
    
    def get_component_status(self, component: str) -> str:
        """Get current status of a component."""
        return self.component_status.get(component, 'unknown')
    
    def get_overall_system_health(self) -> str:
        """Get overall system health status."""
        if not self.component_status:
            return 'unknown'
        
        if any(status == 'error' for status in self.component_status.values()):
            return 'error'
        elif any(status == 'warning' for status in self.component_status.values()):
            return 'warning'
        else:
            return 'healthy'
    
    def start_monitoring(self, interval: int = 60):
        """
        Start continuous health monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    # Basic system checks
                    self._check_system_resources()
                    time.sleep(interval)
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(interval)
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
    
    def _check_system_resources(self):
        """Check basic system resources."""
        try:
            import psutil
            
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            
            # Determine status based on usage
            if cpu_usage > 90 or memory_usage > 90:
                status = 'error'
                message = f"High resource usage: CPU {cpu_usage}%, Memory {memory_usage}%"
            elif cpu_usage > 80 or memory_usage > 80:
                status = 'warning'
                message = f"Elevated resource usage: CPU {cpu_usage}%, Memory {memory_usage}%"
            else:
                status = 'healthy'
                message = f"Normal resource usage: CPU {cpu_usage}%, Memory {memory_usage}%"
            
            self.log_health_metric(
                component='system',
                status=status,
                message=message,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage
            )
            
        except ImportError:
            # psutil not available, log basic health
            self.log_health_metric(
                component='system',
                status='healthy',
                message='System monitoring (basic check passed, psutil not available)'
            )


class LoggingSystem:
    """Centralized logging system that coordinates all logging components."""
    
    def __init__(self, log_directory: str = "logs"):
        """
        Initialize the logging system.
        
        Args:
            log_directory: Directory to store all log files
        """
        self.log_directory = log_directory
        
        # Initialize component loggers
        self.trade_logger = TradeLogger(log_directory)
        self.performance_monitor = PerformanceMonitor(log_directory)
        self.health_monitor = SystemHealthMonitor(log_directory)
        
        # Setup main system logger
        self.logger = logging.getLogger('TradingSystemLogger')
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # File handler for general logs
        general_log_file = os.path.join(log_directory, f"system_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(general_log_file)
        file_handler.setFormatter(console_formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
    
    def start_monitoring(self):
        """Start all monitoring components."""
        self.health_monitor.start_monitoring()
        self.logger.info("Logging system monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring components."""
        self.health_monitor.stop_monitoring()
        self.logger.info("Logging system monitoring stopped")
    
    def log_trading_activity(self, activity_type: str, details: Dict[str, Any]):
        """
        Log general trading activity.
        
        Args:
            activity_type: Type of activity
            details: Activity details
        """
        self.logger.info(f"TRADING_ACTIVITY: {activity_type} - {json.dumps(details)}")
    
    def log_error(self, component: str, error: Exception, context: Dict[str, Any] = None):
        """
        Log an error with context.
        
        Args:
            component: Component where error occurred
            error: Exception object
            context: Additional context information
        """
        error_details = {
            'component': component,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {}
        }
        
        self.logger.error(f"ERROR: {json.dumps(error_details)}")
        
        # Also log to health monitor
        self.health_monitor.log_health_metric(
            component=component,
            status='error',
            message=f"Error: {str(error)}"
        )
    
    def generate_daily_report(self) -> Dict[str, Any]:
        """
        Generate a daily summary report.
        
        Returns:
            Dictionary with daily report data
        """
        today = datetime.now().date()
        
        # Get today's trades
        today_trades = self.trade_logger.get_trades_by_date(datetime.now())
        
        # Get performance metrics
        performance_metrics = self.performance_monitor.get_metrics_by_category('trading', 24)
        
        # Calculate summary statistics
        total_trades = len(today_trades)
        total_pnl = sum(trade.pnl for trade in today_trades if trade.pnl is not None)
        
        winning_trades = [trade for trade in today_trades if trade.pnl and trade.pnl > 0]
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        
        report = {
            'date': today.isoformat(),
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'system_health': self.health_monitor.get_overall_system_health(),
            'component_status': self.health_monitor.component_status.copy(),
            'performance_metrics_count': len(performance_metrics)
        }
        
        self.logger.info(f"DAILY_REPORT: {json.dumps(report)}")
        return report


# Global logging system instance
logging_system = LoggingSystem()