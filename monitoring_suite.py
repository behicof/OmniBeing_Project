"""
Enterprise Monitoring Suite for OmniBeing Trading System.
Comprehensive production monitoring with real-time analytics and alerting.
"""

import asyncio
import logging
import time
import smtplib
import json
import requests
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import psutil
import redis
import asyncpg
from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
from production_config import ProductionConfig

# Metrics
trade_counter = Counter('trading_operations_total', 'Total trading operations', ['operation', 'status'])
trade_duration = Histogram('trading_operation_duration_seconds', 'Trading operation duration')
system_cpu = Gauge('system_cpu_percent', 'System CPU usage percentage')
system_memory = Gauge('system_memory_percent', 'System memory usage percentage')
active_positions = Gauge('active_positions_total', 'Number of active trading positions')
portfolio_value = Gauge('portfolio_value_usd', 'Total portfolio value in USD')
api_requests = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])

@dataclass
class AlertThreshold:
    """Alert threshold configuration."""
    metric: str
    operator: str  # 'gt', 'lt', 'eq'
    value: float
    duration: int  # seconds
    severity: str  # 'low', 'medium', 'high', 'critical'

@dataclass
class TradingMetrics:
    """Trading performance metrics."""
    timestamp: datetime
    total_trades: int
    successful_trades: int
    failed_trades: int
    total_pnl: float
    win_rate: float
    avg_trade_duration: float
    portfolio_value: float
    active_positions: int
    max_drawdown: float
    sharpe_ratio: float

@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_in_bytes: int
    network_out_bytes: int
    open_connections: int
    response_time_ms: float

@dataclass
class Alert:
    """Alert definition."""
    id: str
    severity: str
    title: str
    description: str
    timestamp: datetime
    metric_name: str
    current_value: float
    threshold_value: float
    resolved: bool = False
    resolution_time: Optional[datetime] = None

class MonitoringSuite:
    """
    Comprehensive monitoring suite for production trading system.
    Handles metrics collection, alerting, and performance analytics.
    """
    
    def __init__(self, config: ProductionConfig):
        """
        Initialize monitoring suite.
        
        Args:
            config: Production configuration
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.redis_client = None
        self.db_pool = None
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_thresholds: List[AlertThreshold] = []
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.last_metrics_collection = datetime.now()
        self.collection_interval = 30  # seconds
        
        # Setup default alert thresholds
        self._setup_default_thresholds()
        
        # Start metrics server
        self.registry = CollectorRegistry()
        self._start_metrics_server()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup monitoring logger."""
        logger = logging.getLogger('monitoring')
        logger.setLevel(getattr(logging, self.config.monitoring.log_level))
        
        # File handler
        file_handler = logging.FileHandler('logs/monitoring.log')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_default_thresholds(self):
        """Setup default alert thresholds."""
        self.alert_thresholds = [
            AlertThreshold('cpu_percent', 'gt', 80.0, 300, 'high'),
            AlertThreshold('memory_percent', 'gt', 85.0, 180, 'high'),
            AlertThreshold('disk_usage_percent', 'gt', 90.0, 600, 'critical'),
            AlertThreshold('response_time_ms', 'gt', 1000.0, 120, 'medium'),
            AlertThreshold('win_rate', 'lt', 0.4, 1800, 'medium'),
            AlertThreshold('max_drawdown', 'gt', 0.1, 600, 'high'),
            AlertThreshold('active_positions', 'gt', 50, 300, 'medium'),
            AlertThreshold('failed_trades', 'gt', 10, 900, 'medium'),
        ]
    
    def _start_metrics_server(self):
        """Start Prometheus metrics server."""
        try:
            start_http_server(8001, registry=self.registry)
            self.logger.info("Metrics server started on port 8001")
        except Exception as e:
            self.logger.error(f"Failed to start metrics server: {e}")
    
    async def initialize(self):
        """Initialize monitoring connections."""
        try:
            # Initialize Redis connection
            redis_url = self.config.get_redis_url()
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            await self._test_redis_connection()
            
            # Initialize database connection
            db_url = self.config.get_database_url()
            self.db_pool = await asyncpg.create_pool(db_url, min_size=2, max_size=5)
            
            self.logger.info("Monitoring suite initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring: {e}")
            raise
    
    async def _test_redis_connection(self):
        """Test Redis connection."""
        try:
            await asyncio.to_thread(self.redis_client.ping)
            self.logger.info("Redis connection established")
        except Exception as e:
            self.logger.error(f"Redis connection failed: {e}")
            raise
    
    async def start_monitoring(self):
        """Start the monitoring loop."""
        self.logger.info("Starting monitoring suite...")
        
        # Initialize connections
        await self.initialize()
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._collect_metrics_loop()),
            asyncio.create_task(self._check_alerts_loop()),
            asyncio.create_task(self._cleanup_old_data_loop()),
            asyncio.create_task(self._generate_reports_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")
            raise
    
    async def _collect_metrics_loop(self):
        """Main metrics collection loop."""
        while True:
            try:
                # Collect system metrics
                system_metrics = await self._collect_system_metrics()
                
                # Collect trading metrics
                trading_metrics = await self._collect_trading_metrics()
                
                # Store metrics
                await self._store_metrics(system_metrics, trading_metrics)
                
                # Update Prometheus metrics
                self._update_prometheus_metrics(system_metrics, trading_metrics)
                
                self.logger.debug("Metrics collected successfully")
                
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")
            
            await asyncio.sleep(self.collection_interval)
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network stats
        network = psutil.net_io_counters()
        
        # Connection count
        connections = len(psutil.net_connections())
        
        # Response time (ping localhost)
        start_time = time.time()
        try:
            requests.get('http://localhost:8000/health', timeout=5)
            response_time = (time.time() - start_time) * 1000
        except:
            response_time = 5000  # Timeout value
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_usage_percent=(disk.used / disk.total) * 100,
            network_in_bytes=network.bytes_recv,
            network_out_bytes=network.bytes_sent,
            open_connections=connections,
            response_time_ms=response_time
        )
    
    async def _collect_trading_metrics(self) -> TradingMetrics:
        """Collect trading performance metrics."""
        try:
            # Get trading data from database
            async with self.db_pool.acquire() as conn:
                # Total trades today
                trades_query = """
                    SELECT 
                        COUNT(*) as total_trades,
                        COUNT(*) FILTER (WHERE status = 'completed') as successful_trades,
                        COUNT(*) FILTER (WHERE status = 'failed') as failed_trades,
                        COALESCE(SUM(price * quantity), 0) as total_pnl,
                        COALESCE(AVG(EXTRACT(EPOCH FROM (updated_at - created_at))), 0) as avg_duration
                    FROM trades 
                    WHERE DATE(timestamp) = CURRENT_DATE
                """
                
                trade_stats = await conn.fetchrow(trades_query)
                
                # Portfolio value (simplified - would come from broker API)
                portfolio_value = await self._get_portfolio_value()
                
                # Active positions
                active_positions_count = await self._get_active_positions_count()
                
                # Calculate performance metrics
                win_rate = (trade_stats['successful_trades'] / max(trade_stats['total_trades'], 1))
                max_drawdown = await self._calculate_max_drawdown(conn)
                sharpe_ratio = await self._calculate_sharpe_ratio(conn)
                
                return TradingMetrics(
                    timestamp=datetime.now(),
                    total_trades=trade_stats['total_trades'],
                    successful_trades=trade_stats['successful_trades'],
                    failed_trades=trade_stats['failed_trades'],
                    total_pnl=float(trade_stats['total_pnl']),
                    win_rate=win_rate,
                    avg_trade_duration=float(trade_stats['avg_duration']),
                    portfolio_value=portfolio_value,
                    active_positions=active_positions_count,
                    max_drawdown=max_drawdown,
                    sharpe_ratio=sharpe_ratio
                )
                
        except Exception as e:
            self.logger.error(f"Error collecting trading metrics: {e}")
            # Return default metrics
            return TradingMetrics(
                timestamp=datetime.now(),
                total_trades=0, successful_trades=0, failed_trades=0,
                total_pnl=0.0, win_rate=0.0, avg_trade_duration=0.0,
                portfolio_value=0.0, active_positions=0,
                max_drawdown=0.0, sharpe_ratio=0.0
            )
    
    async def _get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        # This would typically connect to broker API
        # For now, return a placeholder value
        return 10000.0
    
    async def _get_active_positions_count(self) -> int:
        """Get number of active positions."""
        # This would typically connect to broker API
        # For now, return a placeholder value
        return 5
    
    async def _calculate_max_drawdown(self, conn) -> float:
        """Calculate maximum drawdown."""
        try:
            query = """
                SELECT COALESCE(MAX(pnl) - MIN(pnl), 0.0) as max_drawdown
                FROM performance 
                WHERE date >= CURRENT_DATE - INTERVAL '30 days'
            """
            result = await conn.fetchval(query)
            return float(result) if result else 0.0
        except:
            return 0.0
    
    async def _calculate_sharpe_ratio(self, conn) -> float:
        """Calculate Sharpe ratio."""
        try:
            query = """
                SELECT 
                    CASE 
                        WHEN STDDEV(pnl) > 0 THEN AVG(pnl) / STDDEV(pnl)
                        ELSE 0.0 
                    END as sharpe_ratio
                FROM performance 
                WHERE date >= CURRENT_DATE - INTERVAL '30 days'
            """
            result = await conn.fetchval(query)
            return float(result) if result else 0.0
        except:
            return 0.0
    
    async def _store_metrics(self, system_metrics: SystemMetrics, trading_metrics: TradingMetrics):
        """Store metrics in Redis and database."""
        try:
            # Store in Redis for real-time access
            metrics_data = {
                'system': asdict(system_metrics),
                'trading': asdict(trading_metrics)
            }
            
            # Convert datetime objects to ISO format
            for category in metrics_data.values():
                for key, value in category.items():
                    if isinstance(value, datetime):
                        category[key] = value.isoformat()
            
            await asyncio.to_thread(
                self.redis_client.setex,
                'latest_metrics',
                300,  # 5 minute expiry
                json.dumps(metrics_data)
            )
            
            # Store in database for historical analysis
            if self.db_pool:
                await self._store_metrics_in_db(system_metrics, trading_metrics)
            
        except Exception as e:
            self.logger.error(f"Error storing metrics: {e}")
    
    async def _store_metrics_in_db(self, system_metrics: SystemMetrics, trading_metrics: TradingMetrics):
        """Store metrics in database."""
        try:
            async with self.db_pool.acquire() as conn:
                # Store system metrics
                await conn.execute("""
                    INSERT INTO system_metrics 
                    (timestamp, cpu_percent, memory_percent, disk_usage_percent, 
                     network_in_bytes, network_out_bytes, open_connections, response_time_ms)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, system_metrics.timestamp, system_metrics.cpu_percent,
                    system_metrics.memory_percent, system_metrics.disk_usage_percent,
                    system_metrics.network_in_bytes, system_metrics.network_out_bytes,
                    system_metrics.open_connections, system_metrics.response_time_ms)
                
                # Store trading metrics
                await conn.execute("""
                    INSERT INTO trading_metrics 
                    (timestamp, total_trades, successful_trades, failed_trades,
                     total_pnl, win_rate, avg_trade_duration, portfolio_value,
                     active_positions, max_drawdown, sharpe_ratio)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """, trading_metrics.timestamp, trading_metrics.total_trades,
                    trading_metrics.successful_trades, trading_metrics.failed_trades,
                    trading_metrics.total_pnl, trading_metrics.win_rate,
                    trading_metrics.avg_trade_duration, trading_metrics.portfolio_value,
                    trading_metrics.active_positions, trading_metrics.max_drawdown,
                    trading_metrics.sharpe_ratio)
                
        except Exception as e:
            self.logger.error(f"Error storing metrics in database: {e}")
    
    def _update_prometheus_metrics(self, system_metrics: SystemMetrics, trading_metrics: TradingMetrics):
        """Update Prometheus metrics."""
        try:
            # System metrics
            system_cpu.set(system_metrics.cpu_percent)
            system_memory.set(system_metrics.memory_percent)
            
            # Trading metrics
            active_positions.set(trading_metrics.active_positions)
            portfolio_value.set(trading_metrics.portfolio_value)
            
            # Update counters
            trade_counter.labels(operation='total', status='all').inc(trading_metrics.total_trades)
            trade_counter.labels(operation='successful', status='completed').inc(trading_metrics.successful_trades)
            trade_counter.labels(operation='failed', status='failed').inc(trading_metrics.failed_trades)
            
        except Exception as e:
            self.logger.error(f"Error updating Prometheus metrics: {e}")
    
    async def _check_alerts_loop(self):
        """Check alert conditions."""
        while True:
            try:
                await self._check_alert_conditions()
            except Exception as e:
                self.logger.error(f"Error checking alerts: {e}")
            
            await asyncio.sleep(60)  # Check every minute
    
    async def _check_alert_conditions(self):
        """Check all alert conditions."""
        try:
            # Get latest metrics
            metrics_json = await asyncio.to_thread(self.redis_client.get, 'latest_metrics')
            if not metrics_json:
                return
            
            metrics = json.loads(metrics_json)
            system_metrics = metrics.get('system', {})
            trading_metrics = metrics.get('trading', {})
            
            # Check each threshold
            for threshold in self.alert_thresholds:
                await self._check_threshold(threshold, system_metrics, trading_metrics)
                
        except Exception as e:
            self.logger.error(f"Error in alert checking: {e}")
    
    async def _check_threshold(self, threshold: AlertThreshold, system_metrics: Dict, trading_metrics: Dict):
        """Check a specific threshold."""
        try:
            # Get current value
            current_value = None
            if threshold.metric in system_metrics:
                current_value = system_metrics[threshold.metric]
            elif threshold.metric in trading_metrics:
                current_value = trading_metrics[threshold.metric]
            
            if current_value is None:
                return
            
            # Check condition
            triggered = False
            if threshold.operator == 'gt' and current_value > threshold.value:
                triggered = True
            elif threshold.operator == 'lt' and current_value < threshold.value:
                triggered = True
            elif threshold.operator == 'eq' and current_value == threshold.value:
                triggered = True
            
            alert_id = f"{threshold.metric}_{threshold.operator}_{threshold.value}"
            
            if triggered:
                if alert_id not in self.active_alerts:
                    # Create new alert
                    alert = Alert(
                        id=alert_id,
                        severity=threshold.severity,
                        title=f"{threshold.metric.replace('_', ' ').title()} Alert",
                        description=f"{threshold.metric} is {current_value}, threshold is {threshold.operator} {threshold.value}",
                        timestamp=datetime.now(),
                        metric_name=threshold.metric,
                        current_value=current_value,
                        threshold_value=threshold.value
                    )
                    
                    self.active_alerts[alert_id] = alert
                    await self._send_alert(alert)
            else:
                # Resolve alert if it exists
                if alert_id in self.active_alerts:
                    alert = self.active_alerts[alert_id]
                    alert.resolved = True
                    alert.resolution_time = datetime.now()
                    await self._send_alert_resolution(alert)
                    del self.active_alerts[alert_id]
                    
        except Exception as e:
            self.logger.error(f"Error checking threshold {threshold.metric}: {e}")
    
    async def _send_alert(self, alert: Alert):
        """Send alert notification."""
        self.logger.warning(f"🚨 ALERT: {alert.title} - {alert.description}")
        
        # Send email alert
        if self.config.monitoring.alert_email:
            await self._send_email_alert(alert)
        
        # Send Slack alert
        slack_webhook = self.config.get_slack_webhook()
        if slack_webhook:
            await self._send_slack_alert(alert, slack_webhook)
    
    async def _send_alert_resolution(self, alert: Alert):
        """Send alert resolution notification."""
        self.logger.info(f"✅ RESOLVED: {alert.title}")
        
        # Send resolution notifications
        if self.config.monitoring.alert_email:
            await self._send_email_resolution(alert)
    
    async def _send_email_alert(self, alert: Alert):
        """Send email alert."""
        try:
            msg = MimeMultipart()
            msg['From'] = 'alerts@omnibeing.com'
            msg['To'] = self.config.monitoring.alert_email
            msg['Subject'] = f"🚨 OmniBeing Alert: {alert.title}"
            
            body = f"""
            Alert Details:
            - Severity: {alert.severity.upper()}
            - Metric: {alert.metric_name}
            - Current Value: {alert.current_value}
            - Threshold: {alert.threshold_value}
            - Time: {alert.timestamp}
            - Description: {alert.description}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Note: Email sending would require SMTP configuration
            self.logger.info(f"Email alert prepared for {alert.title}")
            
        except Exception as e:
            self.logger.error(f"Error sending email alert: {e}")
    
    async def _send_email_resolution(self, alert: Alert):
        """Send email resolution."""
        try:
            self.logger.info(f"Email resolution prepared for {alert.title}")
        except Exception as e:
            self.logger.error(f"Error sending email resolution: {e}")
    
    async def _send_slack_alert(self, alert: Alert, webhook_url: str):
        """Send Slack alert."""
        try:
            color = {
                'low': '#36a64f',
                'medium': '#ff9500',
                'high': '#ff0000',
                'critical': '#8b0000'
            }.get(alert.severity, '#36a64f')
            
            payload = {
                'attachments': [{
                    'color': color,
                    'title': f"🚨 {alert.title}",
                    'text': alert.description,
                    'fields': [
                        {'title': 'Severity', 'value': alert.severity.upper(), 'short': True},
                        {'title': 'Current Value', 'value': str(alert.current_value), 'short': True},
                        {'title': 'Threshold', 'value': str(alert.threshold_value), 'short': True},
                        {'title': 'Time', 'value': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), 'short': True}
                    ],
                    'footer': 'OmniBeing Trading System',
                    'ts': int(alert.timestamp.timestamp())
                }]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.status_code == 200:
                self.logger.info(f"Slack alert sent for {alert.title}")
            else:
                self.logger.error(f"Failed to send Slack alert: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error sending Slack alert: {e}")
    
    async def _cleanup_old_data_loop(self):
        """Clean up old monitoring data."""
        while True:
            try:
                await self._cleanup_old_data()
            except Exception as e:
                self.logger.error(f"Error in cleanup: {e}")
            
            # Run cleanup daily
            await asyncio.sleep(86400)
    
    async def _cleanup_old_data(self):
        """Remove old monitoring data."""
        try:
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    # Remove old system metrics (older than retention period)
                    cutoff_date = datetime.now() - timedelta(days=self.config.monitoring.metrics_retention_days)
                    
                    await conn.execute(
                        "DELETE FROM system_metrics WHERE timestamp < $1",
                        cutoff_date
                    )
                    
                    await conn.execute(
                        "DELETE FROM trading_metrics WHERE timestamp < $1",
                        cutoff_date
                    )
                    
                    self.logger.info("Old monitoring data cleaned up")
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    async def _generate_reports_loop(self):
        """Generate periodic reports."""
        while True:
            try:
                await self._generate_daily_report()
            except Exception as e:
                self.logger.error(f"Error generating reports: {e}")
            
            # Generate daily reports
            await asyncio.sleep(86400)
    
    async def _generate_daily_report(self):
        """Generate daily performance report."""
        try:
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    # Get daily stats
                    query = """
                        SELECT 
                            DATE(timestamp) as date,
                            AVG(cpu_percent) as avg_cpu,
                            AVG(memory_percent) as avg_memory,
                            AVG(response_time_ms) as avg_response_time
                        FROM system_metrics 
                        WHERE DATE(timestamp) = CURRENT_DATE
                        GROUP BY DATE(timestamp)
                    """
                    
                    system_stats = await conn.fetchrow(query)
                    
                    trading_query = """
                        SELECT 
                            SUM(total_trades) as daily_trades,
                            AVG(win_rate) as avg_win_rate,
                            SUM(total_pnl) as daily_pnl
                        FROM trading_metrics 
                        WHERE DATE(timestamp) = CURRENT_DATE
                    """
                    
                    trading_stats = await conn.fetchrow(trading_query)
                    
                    # Log daily report
                    self.logger.info(f"📊 Daily Report - {datetime.now().strftime('%Y-%m-%d')}")
                    if system_stats:
                        self.logger.info(f"System: CPU {system_stats['avg_cpu']:.1f}%, Memory {system_stats['avg_memory']:.1f}%, Response {system_stats['avg_response_time']:.1f}ms")
                    if trading_stats:
                        self.logger.info(f"Trading: {trading_stats['daily_trades']} trades, {trading_stats['avg_win_rate']:.2%} win rate, ${trading_stats['daily_pnl']:.2f} PnL")
                    
        except Exception as e:
            self.logger.error(f"Error generating daily report: {e}")
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        try:
            metrics_json = await asyncio.to_thread(self.redis_client.get, 'latest_metrics')
            if metrics_json:
                return json.loads(metrics_json)
            return {}
        except Exception as e:
            self.logger.error(f"Error getting metrics summary: {e}")
            return {}
    
    async def close(self):
        """Close monitoring connections."""
        try:
            if self.redis_client:
                await asyncio.to_thread(self.redis_client.close)
            if self.db_pool:
                await self.db_pool.close()
            self.logger.info("Monitoring suite closed")
        except Exception as e:
            self.logger.error(f"Error closing monitoring: {e}")


async def main():
    """Main monitoring entry point."""
    from production_config import get_production_config
    
    config = get_production_config()
    monitoring = MonitoringSuite(config)
    
    try:
        await monitoring.start_monitoring()
    except KeyboardInterrupt:
        print("\nShutting down monitoring...")
    finally:
        await monitoring.close()


if __name__ == "__main__":
    asyncio.run(main())