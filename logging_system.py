"""
Comprehensive Logging System
Trade execution logs, decision reasoning, performance tracking, and monitoring
"""

import logging
import logging.handlers
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from queue import Queue
import asyncio

from config import get_config

class LogLevel(Enum):
    """Custom log levels for trading system"""
    TRADE = 25      # Trade execution
    DECISION = 26   # Decision making
    PERFORMANCE = 27 # Performance metrics
    RISK = 28       # Risk management
    ALERT = 29      # System alerts

class TradingLoggerAdapter(logging.LoggerAdapter):
    """Custom logger adapter for trading-specific logging"""
    
    def __init__(self, logger, extra=None):
        super().__init__(logger, extra or {})
    
    def trade(self, message, *args, **kwargs):
        """Log trade execution"""
        self.log(LogLevel.TRADE.value, message, *args, **kwargs)
    
    def decision(self, message, *args, **kwargs):
        """Log decision making"""
        self.log(LogLevel.DECISION.value, message, *args, **kwargs)
    
    def performance(self, message, *args, **kwargs):
        """Log performance metrics"""
        self.log(LogLevel.PERFORMANCE.value, message, *args, **kwargs)
    
    def risk_alert(self, message, *args, **kwargs):
        """Log risk management alerts"""
        self.log(LogLevel.RISK.value, message, *args, **kwargs)
    
    def system_alert(self, message, *args, **kwargs):
        """Log system alerts"""
        self.log(LogLevel.ALERT.value, message, *args, **kwargs)

@dataclass
class TradeLogEntry:
    """Trade log entry structure"""
    timestamp: datetime
    symbol: str
    action: str  # 'BUY', 'SELL', 'CANCEL'
    quantity: float
    price: float
    order_id: str
    status: str  # 'PENDING', 'FILLED', 'CANCELLED', 'FAILED'
    reasoning: Dict[str, Any]
    execution_time_ms: Optional[float] = None
    fees: Optional[float] = None
    slippage: Optional[float] = None

@dataclass
class DecisionLogEntry:
    """Decision reasoning log entry"""
    timestamp: datetime
    decision_type: str  # 'TRADE', 'RISK', 'POSITION'
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    reasoning: Dict[str, Any]
    confidence: float
    module_scores: Dict[str, float]
    execution_time_ms: float

@dataclass
class PerformanceLogEntry:
    """Performance metrics log entry"""
    timestamp: datetime
    portfolio_value: float
    total_pnl: float
    daily_pnl: float
    open_positions: int
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    risk_score: float

@dataclass
class RiskLogEntry:
    """Risk management log entry"""
    timestamp: datetime
    risk_level: str
    risk_score: float
    positions: Dict[str, Any]
    violations: List[str]
    actions_taken: List[str]
    portfolio_exposure: float

class LoggingSystem:
    """
    Comprehensive logging system for the trading platform
    """
    
    def __init__(self):
        self.config = get_config()
        
        # Initialize custom log levels
        self._setup_custom_levels()
        
        # Create loggers
        self.setup_loggers()
        
        # In-memory buffers for real-time access
        self.trade_buffer = Queue(maxsize=1000)
        self.decision_buffer = Queue(maxsize=1000)
        self.performance_buffer = Queue(maxsize=1000)
        self.risk_buffer = Queue(maxsize=1000)
        
        # Log statistics
        self.log_stats = {
            'trades_logged': 0,
            'decisions_logged': 0,
            'performance_logged': 0,
            'risks_logged': 0,
            'errors_logged': 0,
            'start_time': datetime.now()
        }
        
        # Background log processor
        self.log_queue = Queue()
        self.log_processor_running = False
        self.log_processor_thread = None
        
        self.logger = self.get_logger('main')
        self.logger.info("Logging System initialized")
    
    def _setup_custom_levels(self):
        """Setup custom logging levels"""
        logging.addLevelName(LogLevel.TRADE.value, "TRADE")
        logging.addLevelName(LogLevel.DECISION.value, "DECISION")
        logging.addLevelName(LogLevel.PERFORMANCE.value, "PERFORMANCE")
        logging.addLevelName(LogLevel.RISK.value, "RISK")
        logging.addLevelName(LogLevel.ALERT.value, "ALERT")
    
    def setup_loggers(self):
        """Setup different loggers for different purposes"""
        
        # Ensure log directory exists
        log_dir = os.path.dirname(self.config.LOG_FILE)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Main system logger
        self.main_logger = self._create_logger(
            'trading_system',
            self.config.LOG_FILE,
            format_str='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        
        # Trade execution logger
        self.trade_logger = self._create_logger(
            'trades',
            'trades.log',
            format_str='%(asctime)s [TRADE] %(message)s'
        )
        
        # Decision reasoning logger
        self.decision_logger = self._create_logger(
            'decisions',
            'decisions.log',
            format_str='%(asctime)s [DECISION] %(message)s'
        )
        
        # Performance tracking logger
        self.performance_logger = self._create_logger(
            'performance',
            'performance.log',
            format_str='%(asctime)s [PERFORMANCE] %(message)s'
        )
        
        # Risk management logger
        self.risk_logger = self._create_logger(
            'risk',
            'risk.log',
            format_str='%(asctime)s [RISK] %(message)s'
        )
        
        # Error logger
        self.error_logger = self._create_logger(
            'errors',
            'errors.log',
            format_str='%(asctime)s [ERROR] %(name)s: %(message)s',
            level=logging.ERROR
        )
    
    def _create_logger(self, name: str, filename: str, format_str: str, 
                      level: int = None) -> logging.Logger:
        """Create a configured logger"""
        logger = logging.getLogger(name)
        logger.setLevel(level or getattr(logging, self.config.LOG_LEVEL))
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            filename,
            maxBytes=self.config.LOG_MAX_BYTES,
            backupCount=self.config.LOG_BACKUP_COUNT
        )
        file_handler.setFormatter(logging.Formatter(format_str))
        logger.addHandler(file_handler)
        
        # Console handler for important logs
        if level is None or level <= logging.WARNING:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(format_str))
            console_handler.setLevel(logging.INFO)
            logger.addHandler(console_handler)
        
        return logger
    
    def get_logger(self, name: str) -> TradingLoggerAdapter:
        """Get a trading logger adapter"""
        base_logger = logging.getLogger(f"trading_system.{name}")
        return TradingLoggerAdapter(base_logger)
    
    def start_log_processor(self):
        """Start background log processor"""
        if not self.log_processor_running:
            self.log_processor_running = True
            self.log_processor_thread = threading.Thread(
                target=self._log_processor_worker,
                daemon=True
            )
            self.log_processor_thread.start()
            self.logger.info("Log processor started")
    
    def stop_log_processor(self):
        """Stop background log processor"""
        self.log_processor_running = False
        if self.log_processor_thread:
            self.log_processor_thread.join(timeout=5)
            self.logger.info("Log processor stopped")
    
    def _log_processor_worker(self):
        """Background worker to process logs"""
        while self.log_processor_running:
            try:
                # Process log queue
                if not self.log_queue.empty():
                    log_entry = self.log_queue.get(timeout=1)
                    self._process_log_entry(log_entry)
                else:
                    # No logs to process, sleep briefly
                    threading.Event().wait(0.1)
            except Exception as e:
                self.error_logger.error(f"Error in log processor: {e}")
    
    def _process_log_entry(self, log_entry: Dict[str, Any]):
        """Process a log entry"""
        entry_type = log_entry.get('type')
        
        if entry_type == 'trade':
            self._add_to_buffer(self.trade_buffer, log_entry)
        elif entry_type == 'decision':
            self._add_to_buffer(self.decision_buffer, log_entry)
        elif entry_type == 'performance':
            self._add_to_buffer(self.performance_buffer, log_entry)
        elif entry_type == 'risk':
            self._add_to_buffer(self.risk_buffer, log_entry)
    
    def _add_to_buffer(self, buffer: Queue, entry: Dict[str, Any]):
        """Add entry to buffer with overflow handling"""
        try:
            if buffer.full():
                buffer.get_nowait()  # Remove oldest entry
            buffer.put_nowait(entry)
        except Exception as e:
            self.error_logger.error(f"Error adding to buffer: {e}")
    
    def log_trade(self, trade_entry: TradeLogEntry):
        """Log a trade execution"""
        entry_dict = asdict(trade_entry)
        entry_dict['type'] = 'trade'
        
        # Log to file
        self.trade_logger.info(json.dumps(entry_dict, default=str))
        
        # Add to processing queue
        self.log_queue.put(entry_dict)
        
        self.log_stats['trades_logged'] += 1
        
        # Console log for important trades
        if trade_entry.status in ['FILLED', 'FAILED']:
            self.logger.trade(
                f"{trade_entry.action} {trade_entry.quantity} {trade_entry.symbol} "
                f"at {trade_entry.price} - {trade_entry.status}"
            )
    
    def log_decision(self, decision_entry: DecisionLogEntry):
        """Log a decision reasoning"""
        entry_dict = asdict(decision_entry)
        entry_dict['type'] = 'decision'
        
        # Log to file
        self.decision_logger.info(json.dumps(entry_dict, default=str))
        
        # Add to processing queue
        self.log_queue.put(entry_dict)
        
        self.log_stats['decisions_logged'] += 1
        
        # Console log for important decisions
        if decision_entry.confidence > 0.7:
            self.logger.decision(
                f"{decision_entry.decision_type} decision: "
                f"{decision_entry.outputs} (confidence: {decision_entry.confidence:.2f})"
            )
    
    def log_performance(self, perf_entry: PerformanceLogEntry):
        """Log performance metrics"""
        entry_dict = asdict(perf_entry)
        entry_dict['type'] = 'performance'
        
        # Log to file
        self.performance_logger.info(json.dumps(entry_dict, default=str))
        
        # Add to processing queue
        self.log_queue.put(entry_dict)
        
        self.log_stats['performance_logged'] += 1
        
        # Console log summary
        self.logger.performance(
            f"Portfolio: ${perf_entry.portfolio_value:.2f}, "
            f"PnL: ${perf_entry.total_pnl:.2f}, "
            f"Positions: {perf_entry.open_positions}, "
            f"Win Rate: {perf_entry.win_rate:.1%}"
        )
    
    def log_risk(self, risk_entry: RiskLogEntry):
        """Log risk management information"""
        entry_dict = asdict(risk_entry)
        entry_dict['type'] = 'risk'
        
        # Log to file
        self.risk_logger.info(json.dumps(entry_dict, default=str))
        
        # Add to processing queue
        self.log_queue.put(entry_dict)
        
        self.log_stats['risks_logged'] += 1
        
        # Console log for high risk or violations
        if risk_entry.risk_level in ['high', 'critical'] or risk_entry.violations:
            self.logger.risk_alert(
                f"Risk Level: {risk_entry.risk_level}, "
                f"Score: {risk_entry.risk_score:.2f}, "
                f"Violations: {len(risk_entry.violations)}"
            )
    
    def log_error(self, error_msg: str, exception: Exception = None, 
                  context: Dict[str, Any] = None):
        """Log an error with context"""
        error_entry = {
            'timestamp': datetime.now(),
            'message': error_msg,
            'exception': str(exception) if exception else None,
            'exception_type': type(exception).__name__ if exception else None,
            'context': context or {}
        }
        
        # Log to file
        self.error_logger.error(json.dumps(error_entry, default=str))
        
        self.log_stats['errors_logged'] += 1
        
        # Console log
        self.logger.error(f"{error_msg}: {exception}")
    
    def log_system_alert(self, alert_type: str, message: str, 
                        severity: str = 'info', context: Dict[str, Any] = None):
        """Log system alerts"""
        alert_entry = {
            'timestamp': datetime.now(),
            'alert_type': alert_type,
            'message': message,
            'severity': severity,
            'context': context or {}
        }
        
        # Log to appropriate logger based on severity
        if severity == 'critical':
            self.error_logger.critical(json.dumps(alert_entry, default=str))
            self.logger.system_alert(f"CRITICAL ALERT [{alert_type}]: {message}")
        elif severity == 'warning':
            self.main_logger.warning(json.dumps(alert_entry, default=str))
            self.logger.system_alert(f"WARNING [{alert_type}]: {message}")
        else:
            self.main_logger.info(json.dumps(alert_entry, default=str))
    
    def get_recent_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trade logs"""
        trades = []
        temp_buffer = Queue()
        
        # Extract from buffer
        while not self.trade_buffer.empty() and len(trades) < limit:
            entry = self.trade_buffer.get()
            trades.append(entry)
            temp_buffer.put(entry)
        
        # Restore buffer
        while not temp_buffer.empty():
            self.trade_buffer.put(temp_buffer.get())
        
        return trades[-limit:]
    
    def get_recent_decisions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent decision logs"""
        decisions = []
        temp_buffer = Queue()
        
        # Extract from buffer
        while not self.decision_buffer.empty() and len(decisions) < limit:
            entry = self.decision_buffer.get()
            decisions.append(entry)
            temp_buffer.put(entry)
        
        # Restore buffer
        while not temp_buffer.empty():
            self.decision_buffer.put(temp_buffer.get())
        
        return decisions[-limit:]
    
    def get_performance_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get performance history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        performance_data = []
        temp_buffer = Queue()
        
        # Extract from buffer
        while not self.performance_buffer.empty():
            entry = self.performance_buffer.get()
            if datetime.fromisoformat(entry['timestamp']) > cutoff_time:
                performance_data.append(entry)
            temp_buffer.put(entry)
        
        # Restore buffer
        while not temp_buffer.empty():
            self.performance_buffer.put(temp_buffer.get())
        
        return sorted(performance_data, key=lambda x: x['timestamp'])
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging system statistics"""
        runtime = datetime.now() - self.log_stats['start_time']
        
        stats = {
            **self.log_stats,
            'runtime_hours': runtime.total_seconds() / 3600,
            'buffer_sizes': {
                'trades': self.trade_buffer.qsize(),
                'decisions': self.decision_buffer.qsize(),
                'performance': self.performance_buffer.qsize(),
                'risks': self.risk_buffer.qsize()
            },
            'log_processor_running': self.log_processor_running
        }
        
        return stats
    
    def export_logs(self, start_date: datetime, end_date: datetime, 
                   log_types: List[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Export logs for a date range"""
        # This would typically read from log files
        # For now, return current buffer contents
        
        exported_logs = {}
        
        if not log_types:
            log_types = ['trades', 'decisions', 'performance', 'risks']
        
        if 'trades' in log_types:
            exported_logs['trades'] = self.get_recent_trades(1000)
        
        if 'decisions' in log_types:
            exported_logs['decisions'] = self.get_recent_decisions(500)
        
        if 'performance' in log_types:
            exported_logs['performance'] = self.get_performance_history(24 * 7)  # 1 week
        
        return exported_logs
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        log_files = [
            self.config.LOG_FILE,
            'trades.log',
            'decisions.log',
            'performance.log',
            'risk.log',
            'errors.log'
        ]
        
        for log_file in log_files:
            if os.path.exists(log_file):
                # Check file modification time
                mod_time = datetime.fromtimestamp(os.path.getmtime(log_file))
                if mod_time < cutoff_date:
                    try:
                        os.remove(log_file)
                        self.logger.info(f"Removed old log file: {log_file}")
                    except Exception as e:
                        self.logger.error(f"Failed to remove log file {log_file}: {e}")

# Global logging system instance
logging_system = None

def get_logging_system() -> LoggingSystem:
    """Get the global logging system instance"""
    global logging_system
    if logging_system is None:
        logging_system = LoggingSystem()
        logging_system.start_log_processor()
    return logging_system

def get_logger(name: str) -> TradingLoggerAdapter:
    """Get a trading logger for a specific module"""
    return get_logging_system().get_logger(name)