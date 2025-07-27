"""
Dynamic Auto-Scaling System for OmniBeing Trading Platform.
Handles auto-scaling based on CPU, memory, trading volume, and market conditions.
"""

import asyncio
import logging
import psutil
import time
import docker
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import redis
import asyncpg
from production_config import ProductionConfig

@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_io_percent: float
    network_throughput: float
    active_connections: int
    trading_volume: float
    market_volatility: float
    response_time_ms: float
    error_rate: float

@dataclass
class ScalingAction:
    """Scaling action details."""
    timestamp: datetime
    action_type: str  # 'scale_up', 'scale_down', 'maintain'
    target_instances: int
    current_instances: int
    trigger_metric: str
    metric_value: float
    threshold_value: float
    reason: str

@dataclass
class InstanceInfo:
    """Information about a service instance."""
    instance_id: str
    service_name: str
    cpu_usage: float
    memory_usage: float
    status: str
    created_at: datetime
    last_health_check: datetime

class AutoScalingManager:
    """
    Comprehensive auto-scaling manager for production trading system.
    Scales services based on multiple metrics and trading conditions.
    """
    
    def __init__(self, config: ProductionConfig):
        """
        Initialize auto-scaling manager.
        
        Args:
            config: Production configuration
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Docker and infrastructure
        self.docker_client = None
        self.redis_client = None
        self.db_pool = None
        
        # Scaling state
        self.current_instances: Dict[str, int] = {}
        self.last_scaling_action: Dict[str, datetime] = {}
        self.scaling_history: List[ScalingAction] = []
        self.instance_info: Dict[str, InstanceInfo] = {}
        
        # Metrics collection
        self.metrics_history: List[ScalingMetrics] = []
        self.prediction_window = 300  # 5 minutes
        
        # Services to manage
        self.managed_services = [
            'trading-system',
            'redis',
            'postgresql'
        ]
        
        # Circuit breaker pattern
        self.circuit_breakers: Dict[str, Dict] = {}
        
        # Connection pools
        self.connection_pools: Dict[str, Any] = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup auto-scaling logger."""
        logger = logging.getLogger('autoscaling')
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler('logs/autoscaling.log')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
        
        return logger
    
    async def initialize(self):
        """Initialize auto-scaling manager."""
        try:
            # Initialize Docker client
            self.docker_client = docker.from_env()
            
            # Initialize Redis connection
            redis_url = self.config.get_redis_url()
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            
            # Initialize database connection
            db_url = self.config.get_database_url()
            self.db_pool = await asyncpg.create_pool(db_url, min_size=2, max_size=10)
            
            # Initialize current instance counts
            await self._discover_current_instances()
            
            # Initialize circuit breakers
            self._initialize_circuit_breakers()
            
            self.logger.info("Auto-scaling manager initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize auto-scaling manager: {e}")
            raise
    
    async def _discover_current_instances(self):
        """Discover currently running instances."""
        try:
            containers = self.docker_client.containers.list()
            
            for service in self.managed_services:
                service_containers = [
                    c for c in containers 
                    if c.name.startswith(f"omnibeing-{service}")
                ]
                self.current_instances[service] = len(service_containers)
                
                # Store instance info
                for container in service_containers:
                    self.instance_info[container.id] = InstanceInfo(
                        instance_id=container.id,
                        service_name=service,
                        cpu_usage=0.0,
                        memory_usage=0.0,
                        status=container.status,
                        created_at=datetime.now(),
                        last_health_check=datetime.now()
                    )
            
            self.logger.info(f"Discovered instances: {self.current_instances}")
            
        except Exception as e:
            self.logger.error(f"Error discovering instances: {e}")
    
    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for services."""
        for service in self.managed_services:
            self.circuit_breakers[service] = {
                'state': 'closed',  # closed, open, half_open
                'failure_count': 0,
                'failure_threshold': 5,
                'recovery_timeout': 60,
                'last_failure_time': None,
                'success_threshold': 3  # for half_open -> closed
            }
    
    async def start_auto_scaling(self):
        """Start the auto-scaling loop."""
        self.logger.info("Starting auto-scaling system...")
        
        # Initialize connections
        await self.initialize()
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._scaling_decision_loop()),
            asyncio.create_task(self._health_monitoring_loop()),
            asyncio.create_task(self._connection_pool_management_loop()),
            asyncio.create_task(self._cleanup_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Auto-scaling error: {e}")
            raise
    
    async def _metrics_collection_loop(self):
        """Collect metrics for scaling decisions."""
        while True:
            try:
                metrics = await self._collect_scaling_metrics()
                
                # Store metrics
                self.metrics_history.append(metrics)
                
                # Keep only recent metrics (last hour)
                cutoff_time = datetime.now() - timedelta(hours=1)
                self.metrics_history = [
                    m for m in self.metrics_history if m.timestamp > cutoff_time
                ]
                
                # Store in Redis for real-time access
                await self._store_metrics_in_redis(metrics)
                
                self.logger.debug(f"Collected scaling metrics: CPU {metrics.cpu_percent:.1f}%, Memory {metrics.memory_percent:.1f}%")
                
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")
            
            await asyncio.sleep(30)  # Collect every 30 seconds
    
    async def _collect_scaling_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network = psutil.net_io_counters()
        
        # Calculate network throughput (bytes per second)
        network_throughput = await self._calculate_network_throughput()
        
        # Trading-specific metrics
        trading_volume = await self._get_trading_volume()
        market_volatility = await self._get_market_volatility()
        
        # Performance metrics
        response_time = await self._measure_response_time()
        error_rate = await self._calculate_error_rate()
        
        # Connection count
        active_connections = len(psutil.net_connections())
        
        return ScalingMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_io_percent=self._calculate_disk_io_percent(),
            network_throughput=network_throughput,
            active_connections=active_connections,
            trading_volume=trading_volume,
            market_volatility=market_volatility,
            response_time_ms=response_time,
            error_rate=error_rate
        )
    
    async def _calculate_network_throughput(self) -> float:
        """Calculate network throughput in MB/s."""
        try:
            # Get current network stats
            current_stats = psutil.net_io_counters()
            
            # Calculate throughput if we have previous data
            if hasattr(self, '_last_network_stats'):
                time_diff = time.time() - self._last_network_time
                bytes_diff = (current_stats.bytes_sent + current_stats.bytes_recv) - \
                           (self._last_network_stats.bytes_sent + self._last_network_stats.bytes_recv)
                throughput_mbps = (bytes_diff / time_diff) / (1024 * 1024)
            else:
                throughput_mbps = 0.0
            
            self._last_network_stats = current_stats
            self._last_network_time = time.time()
            
            return throughput_mbps
            
        except:
            return 0.0
    
    def _calculate_disk_io_percent(self) -> float:
        """Calculate disk I/O utilization percentage."""
        try:
            disk_io = psutil.disk_io_counters()
            if hasattr(self, '_last_disk_io'):
                # Calculate I/O rate (simplified)
                time_diff = time.time() - self._last_disk_time
                io_diff = (disk_io.read_bytes + disk_io.write_bytes) - \
                         (self._last_disk_io.read_bytes + self._last_disk_io.write_bytes)
                io_rate = io_diff / time_diff if time_diff > 0 else 0
                # Convert to percentage (rough estimate)
                io_percent = min(io_rate / (100 * 1024 * 1024), 100)  # Normalize to 100MB/s max
            else:
                io_percent = 0.0
            
            self._last_disk_io = disk_io
            self._last_disk_time = time.time()
            
            return io_percent
            
        except:
            return 0.0
    
    async def _get_trading_volume(self) -> float:
        """Get current trading volume."""
        try:
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    result = await conn.fetchval("""
                        SELECT COALESCE(SUM(quantity * price), 0.0) as volume
                        FROM trades 
                        WHERE timestamp >= NOW() - INTERVAL '5 minutes'
                    """)
                    return float(result) if result else 0.0
        except:
            return 0.0
    
    async def _get_market_volatility(self) -> float:
        """Get current market volatility."""
        try:
            # Placeholder - would connect to market data API
            # For now, return a simulated volatility
            return 0.15  # 15% volatility
        except:
            return 0.0
    
    async def _measure_response_time(self) -> float:
        """Measure average response time."""
        try:
            import requests
            start_time = time.time()
            response = requests.get('http://localhost:8000/health', timeout=5)
            response_time = (time.time() - start_time) * 1000
            return response_time if response.status_code == 200 else 5000
        except:
            return 5000  # Return high value on error
    
    async def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        try:
            if self.redis_client:
                # Get error count from last 5 minutes
                errors = await asyncio.to_thread(self.redis_client.get, 'error_count_5min') or 0
                requests = await asyncio.to_thread(self.redis_client.get, 'request_count_5min') or 1
                return float(errors) / float(requests) * 100
        except:
            return 0.0
    
    async def _store_metrics_in_redis(self, metrics: ScalingMetrics):
        """Store metrics in Redis for real-time access."""
        try:
            metrics_dict = asdict(metrics)
            # Convert datetime to ISO format
            metrics_dict['timestamp'] = metrics.timestamp.isoformat()
            
            await asyncio.to_thread(
                self.redis_client.setex,
                'scaling_metrics',
                300,  # 5 minute expiry
                str(metrics_dict)
            )
        except Exception as e:
            self.logger.error(f"Error storing metrics in Redis: {e}")
    
    async def _scaling_decision_loop(self):
        """Main scaling decision loop."""
        while True:
            try:
                await self._make_scaling_decisions()
            except Exception as e:
                self.logger.error(f"Error in scaling decisions: {e}")
            
            await asyncio.sleep(60)  # Make decisions every minute
    
    async def _make_scaling_decisions(self):
        """Make scaling decisions based on current metrics."""
        if not self.metrics_history:
            return
        
        latest_metrics = self.metrics_history[-1]
        
        for service in self.managed_services:
            try:
                scaling_action = await self._evaluate_service_scaling(service, latest_metrics)
                
                if scaling_action.action_type != 'maintain':
                    await self._execute_scaling_action(service, scaling_action)
                    
            except Exception as e:
                self.logger.error(f"Error evaluating scaling for {service}: {e}")
    
    async def _evaluate_service_scaling(self, service: str, metrics: ScalingMetrics) -> ScalingAction:
        """Evaluate scaling needs for a specific service."""
        current_instances = self.current_instances.get(service, 1)
        scaling_config = self.config.scaling
        
        # Check cooldown period
        last_action_time = self.last_scaling_action.get(service)
        if last_action_time:
            cooldown_time = scaling_config.scale_up_cooldown if current_instances < scaling_config.max_instances else scaling_config.scale_down_cooldown
            if (datetime.now() - last_action_time).total_seconds() < cooldown_time:
                return ScalingAction(
                    timestamp=datetime.now(),
                    action_type='maintain',
                    target_instances=current_instances,
                    current_instances=current_instances,
                    trigger_metric='cooldown',
                    metric_value=0,
                    threshold_value=0,
                    reason='In cooldown period'
                )
        
        # Service-specific scaling logic
        if service == 'trading-system':
            return await self._evaluate_trading_system_scaling(current_instances, metrics)
        elif service == 'redis':
            return await self._evaluate_redis_scaling(current_instances, metrics)
        elif service == 'postgresql':
            return await self._evaluate_database_scaling(current_instances, metrics)
        
        # Default maintain action
        return ScalingAction(
            timestamp=datetime.now(),
            action_type='maintain',
            target_instances=current_instances,
            current_instances=current_instances,
            trigger_metric='none',
            metric_value=0,
            threshold_value=0,
            reason='No scaling needed'
        )
    
    async def _evaluate_trading_system_scaling(self, current_instances: int, metrics: ScalingMetrics) -> ScalingAction:
        """Evaluate scaling for trading system based on multiple factors."""
        scaling_config = self.config.scaling
        
        # CPU-based scaling
        if metrics.cpu_percent > scaling_config.cpu_threshold and current_instances < scaling_config.max_instances:
            return ScalingAction(
                timestamp=datetime.now(),
                action_type='scale_up',
                target_instances=min(current_instances + 1, scaling_config.max_instances),
                current_instances=current_instances,
                trigger_metric='cpu_percent',
                metric_value=metrics.cpu_percent,
                threshold_value=scaling_config.cpu_threshold,
                reason=f'CPU usage {metrics.cpu_percent:.1f}% > {scaling_config.cpu_threshold}%'
            )
        
        # Memory-based scaling
        if metrics.memory_percent > scaling_config.memory_threshold and current_instances < scaling_config.max_instances:
            return ScalingAction(
                timestamp=datetime.now(),
                action_type='scale_up',
                target_instances=min(current_instances + 1, scaling_config.max_instances),
                current_instances=current_instances,
                trigger_metric='memory_percent',
                metric_value=metrics.memory_percent,
                threshold_value=scaling_config.memory_threshold,
                reason=f'Memory usage {metrics.memory_percent:.1f}% > {scaling_config.memory_threshold}%'
            )
        
        # Trading volume-based scaling
        if metrics.trading_volume > 100000 and current_instances < scaling_config.max_instances:
            return ScalingAction(
                timestamp=datetime.now(),
                action_type='scale_up',
                target_instances=min(current_instances + 1, scaling_config.max_instances),
                current_instances=current_instances,
                trigger_metric='trading_volume',
                metric_value=metrics.trading_volume,
                threshold_value=100000,
                reason=f'High trading volume: {metrics.trading_volume}'
            )
        
        # Response time-based scaling
        if metrics.response_time_ms > 1000 and current_instances < scaling_config.max_instances:
            return ScalingAction(
                timestamp=datetime.now(),
                action_type='scale_up',
                target_instances=min(current_instances + 1, scaling_config.max_instances),
                current_instances=current_instances,
                trigger_metric='response_time_ms',
                metric_value=metrics.response_time_ms,
                threshold_value=1000,
                reason=f'High response time: {metrics.response_time_ms}ms'
            )
        
        # Scale down conditions
        if (metrics.cpu_percent < scaling_config.cpu_threshold * 0.5 and 
            metrics.memory_percent < scaling_config.memory_threshold * 0.5 and
            metrics.trading_volume < 10000 and
            current_instances > scaling_config.min_instances):
            
            return ScalingAction(
                timestamp=datetime.now(),
                action_type='scale_down',
                target_instances=max(current_instances - 1, scaling_config.min_instances),
                current_instances=current_instances,
                trigger_metric='low_utilization',
                metric_value=metrics.cpu_percent,
                threshold_value=scaling_config.cpu_threshold * 0.5,
                reason='Low resource utilization'
            )
        
        return ScalingAction(
            timestamp=datetime.now(),
            action_type='maintain',
            target_instances=current_instances,
            current_instances=current_instances,
            trigger_metric='none',
            metric_value=0,
            threshold_value=0,
            reason='No scaling needed'
        )
    
    async def _evaluate_redis_scaling(self, current_instances: int, metrics: ScalingMetrics) -> ScalingAction:
        """Evaluate Redis scaling needs."""
        # Redis scaling based on memory and connections
        if metrics.memory_percent > 80 and current_instances < 3:
            return ScalingAction(
                timestamp=datetime.now(),
                action_type='scale_up',
                target_instances=current_instances + 1,
                current_instances=current_instances,
                trigger_metric='memory_percent',
                metric_value=metrics.memory_percent,
                threshold_value=80,
                reason='Redis memory usage high'
            )
        
        return ScalingAction(
            timestamp=datetime.now(),
            action_type='maintain',
            target_instances=current_instances,
            current_instances=current_instances,
            trigger_metric='none',
            metric_value=0,
            threshold_value=0,
            reason='Redis scaling not needed'
        )
    
    async def _evaluate_database_scaling(self, current_instances: int, metrics: ScalingMetrics) -> ScalingAction:
        """Evaluate database scaling needs."""
        # Database scaling based on connections and I/O
        if metrics.active_connections > 100 and current_instances < 2:
            return ScalingAction(
                timestamp=datetime.now(),
                action_type='scale_up',
                target_instances=current_instances + 1,
                current_instances=current_instances,
                trigger_metric='active_connections',
                metric_value=metrics.active_connections,
                threshold_value=100,
                reason='High database connection count'
            )
        
        return ScalingAction(
            timestamp=datetime.now(),
            action_type='maintain',
            target_instances=current_instances,
            current_instances=current_instances,
            trigger_metric='none',
            metric_value=0,
            threshold_value=0,
            reason='Database scaling not needed'
        )
    
    async def _execute_scaling_action(self, service: str, action: ScalingAction):
        """Execute the scaling action."""
        try:
            self.logger.info(f"Executing {action.action_type} for {service}: {action.current_instances} -> {action.target_instances}")
            
            if action.action_type == 'scale_up':
                await self._scale_up_service(service, action.target_instances - action.current_instances)
            elif action.action_type == 'scale_down':
                await self._scale_down_service(service, action.current_instances - action.target_instances)
            
            # Update state
            self.current_instances[service] = action.target_instances
            self.last_scaling_action[service] = datetime.now()
            self.scaling_history.append(action)
            
            # Store action in database
            await self._store_scaling_action(action)
            
            self.logger.info(f"Scaling action completed for {service}")
            
        except Exception as e:
            self.logger.error(f"Error executing scaling action for {service}: {e}")
    
    async def _scale_up_service(self, service: str, additional_instances: int):
        """Scale up a service by adding instances."""
        try:
            for i in range(additional_instances):
                # Use docker-compose to scale
                result = subprocess.run([
                    'docker-compose', 'up', '-d', '--scale', 
                    f'{service}={self.current_instances[service] + i + 1}'
                ], capture_output=True, text=True, cwd='.')
                
                if result.returncode != 0:
                    raise Exception(f"Docker compose scale failed: {result.stderr}")
                
                self.logger.info(f"Added instance for {service}")
                
        except Exception as e:
            self.logger.error(f"Error scaling up {service}: {e}")
            raise
    
    async def _scale_down_service(self, service: str, instances_to_remove: int):
        """Scale down a service by removing instances."""
        try:
            # Get service containers
            containers = self.docker_client.containers.list(
                filters={'name': f'omnibeing-{service}'}
            )
            
            # Remove the specified number of instances
            for i in range(min(instances_to_remove, len(containers) - 1)):  # Keep at least 1
                container = containers[i]
                
                # Graceful shutdown
                container.stop(timeout=30)
                container.remove()
                
                self.logger.info(f"Removed instance {container.id} for {service}")
                
        except Exception as e:
            self.logger.error(f"Error scaling down {service}: {e}")
            raise
    
    async def _store_scaling_action(self, action: ScalingAction):
        """Store scaling action in database."""
        try:
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO scaling_actions 
                        (timestamp, action_type, target_instances, current_instances,
                         trigger_metric, metric_value, threshold_value, reason)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """, action.timestamp, action.action_type, action.target_instances,
                        action.current_instances, action.trigger_metric, action.metric_value,
                        action.threshold_value, action.reason)
        except Exception as e:
            self.logger.error(f"Error storing scaling action: {e}")
    
    async def _health_monitoring_loop(self):
        """Monitor health of all instances."""
        while True:
            try:
                await self._check_instance_health()
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
            
            await asyncio.sleep(self.config.scaling.health_check_interval)
    
    async def _check_instance_health(self):
        """Check health of all running instances."""
        try:
            containers = self.docker_client.containers.list()
            
            for container in containers:
                if any(service in container.name for service in self.managed_services):
                    health_status = await self._check_container_health(container)
                    
                    # Update instance info
                    if container.id in self.instance_info:
                        self.instance_info[container.id].last_health_check = datetime.now()
                        self.instance_info[container.id].status = health_status
                    
                    # Handle unhealthy instances
                    if health_status == 'unhealthy':
                        await self._handle_unhealthy_instance(container)
                        
        except Exception as e:
            self.logger.error(f"Error checking instance health: {e}")
    
    async def _check_container_health(self, container) -> str:
        """Check health of a specific container."""
        try:
            # Reload container to get latest stats
            container.reload()
            
            if container.status != 'running':
                return 'unhealthy'
            
            # Check if container has health check
            if 'Health' in container.attrs.get('State', {}):
                health = container.attrs['State']['Health']['Status']
                return 'healthy' if health == 'healthy' else 'unhealthy'
            
            # If no health check, assume healthy if running
            return 'healthy'
            
        except Exception as e:
            self.logger.error(f"Error checking container {container.id} health: {e}")
            return 'unhealthy'
    
    async def _handle_unhealthy_instance(self, container):
        """Handle an unhealthy instance."""
        try:
            service_name = self._get_service_name_from_container(container)
            
            self.logger.warning(f"Unhealthy instance detected: {container.name}")
            
            # Try to restart the container
            container.restart(timeout=30)
            
            # Update circuit breaker
            await self._update_circuit_breaker(service_name, success=False)
            
            self.logger.info(f"Restarted unhealthy instance: {container.name}")
            
        except Exception as e:
            self.logger.error(f"Error handling unhealthy instance {container.id}: {e}")
    
    def _get_service_name_from_container(self, container) -> str:
        """Extract service name from container name."""
        for service in self.managed_services:
            if service in container.name:
                return service
        return 'unknown'
    
    async def _update_circuit_breaker(self, service: str, success: bool):
        """Update circuit breaker state for a service."""
        breaker = self.circuit_breakers.get(service)
        if not breaker:
            return
        
        if success:
            if breaker['state'] == 'half_open':
                breaker['failure_count'] = 0
                if breaker.get('success_count', 0) >= breaker['success_threshold']:
                    breaker['state'] = 'closed'
                    self.logger.info(f"Circuit breaker for {service} closed (recovered)")
                else:
                    breaker['success_count'] = breaker.get('success_count', 0) + 1
            elif breaker['state'] == 'closed':
                breaker['failure_count'] = 0
        else:
            breaker['failure_count'] += 1
            breaker['last_failure_time'] = time.time()
            
            if breaker['state'] == 'closed' and breaker['failure_count'] >= breaker['failure_threshold']:
                breaker['state'] = 'open'
                self.logger.warning(f"Circuit breaker for {service} opened (too many failures)")
            elif breaker['state'] == 'half_open':
                breaker['state'] = 'open'
                self.logger.warning(f"Circuit breaker for {service} reopened (failure during recovery)")
    
    async def _connection_pool_management_loop(self):
        """Manage database connection pools."""
        while True:
            try:
                await self._manage_connection_pools()
            except Exception as e:
                self.logger.error(f"Error managing connection pools: {e}")
            
            await asyncio.sleep(300)  # Check every 5 minutes
    
    async def _manage_connection_pools(self):
        """Dynamically manage database connection pools."""
        try:
            # Adjust pool sizes based on load
            if self.db_pool:
                current_connections = self.db_pool.get_size()
                max_connections = self.db_pool.get_max_size()
                
                latest_metrics = self.metrics_history[-1] if self.metrics_history else None
                if latest_metrics:
                    # Increase pool size if high load
                    if (latest_metrics.active_connections > current_connections * 0.8 and 
                        current_connections < max_connections):
                        # Would resize pool if asyncpg supported it
                        self.logger.info(f"High connection usage: {latest_metrics.active_connections}/{current_connections}")
                        
        except Exception as e:
            self.logger.error(f"Error managing connection pools: {e}")
    
    async def _cleanup_loop(self):
        """Clean up old data and optimize resources."""
        while True:
            try:
                await self._cleanup_old_data()
            except Exception as e:
                self.logger.error(f"Error in cleanup: {e}")
            
            # Run cleanup every hour
            await asyncio.sleep(3600)
    
    async def _cleanup_old_data(self):
        """Clean up old scaling data."""
        try:
            # Keep only recent metrics (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.metrics_history = [
                m for m in self.metrics_history if m.timestamp > cutoff_time
            ]
            
            # Keep only recent scaling actions (last week)
            cutoff_time = datetime.now() - timedelta(days=7)
            self.scaling_history = [
                a for a in self.scaling_history if a.timestamp > cutoff_time
            ]
            
            self.logger.info("Auto-scaling data cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    async def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        return {
            'current_instances': self.current_instances,
            'recent_actions': len([a for a in self.scaling_history if a.timestamp > datetime.now() - timedelta(hours=1)]),
            'circuit_breakers': {k: v['state'] for k, v in self.circuit_breakers.items()},
            'last_metrics': asdict(self.metrics_history[-1]) if self.metrics_history else None,
            'managed_services': self.managed_services
        }
    
    async def close(self):
        """Close auto-scaling manager connections."""
        try:
            if self.redis_client:
                await asyncio.to_thread(self.redis_client.close)
            if self.db_pool:
                await self.db_pool.close()
            self.logger.info("Auto-scaling manager closed")
        except Exception as e:
            self.logger.error(f"Error closing auto-scaling manager: {e}")


async def main():
    """Main auto-scaling entry point."""
    from production_config import get_production_config
    
    config = get_production_config()
    autoscaler = AutoScalingManager(config)
    
    try:
        await autoscaler.start_auto_scaling()
    except KeyboardInterrupt:
        print("\nShutting down auto-scaling...")
    finally:
        await autoscaler.close()


if __name__ == "__main__":
    asyncio.run(main())