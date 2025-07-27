"""
Enterprise Monitoring System for OmniBeing Platform.
Provides real-time monitoring, SLA tracking, and institutional-grade analytics.
"""

import time
import threading
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

try:
    from enterprise_config import enterprise_config
except ImportError:
    enterprise_config = None


@dataclass
class SLAMetric:
    """SLA metric tracking data structure."""
    name: str
    current_value: float
    threshold: float
    unit: str
    compliant: bool
    timestamp: datetime


@dataclass
class AlertEvent:
    """Alert event data structure."""
    timestamp: datetime
    level: str  # INFO, WARNING, CRITICAL
    category: str  # SLA, SECURITY, PERFORMANCE, TRADING
    message: str
    details: Dict[str, Any]


class EnterpriseMonitoringSystem:
    """
    Enterprise-grade monitoring system with SLA tracking and real-time alerts.
    """
    
    def __init__(self):
        """Initialize enterprise monitoring system."""
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.monitor_thread = None
        
        # Monitoring configuration
        self.monitoring_enabled = False
        self.real_time_enabled = False
        self.alert_history: List[AlertEvent] = []
        self.metrics_history: List[SLAMetric] = []
        
        # SLA thresholds from enterprise config
        self.sla_thresholds = self._load_sla_thresholds()
        
        # Performance metrics
        self.performance_metrics = {
            'processing_delay_ms': 0.0,
            'trading_error_percentage': 0.0,
            'ram_usage_percentage': 0.0,
            'cpu_usage_percentage': 0.0,
            'active_positions': 0,
            'trades_per_second': 0.0,
            'api_response_time_ms': 0.0
        }
        
        # Enterprise metrics
        self.enterprise_metrics = {
            'institutional_clients': 0,
            'total_aum': 0.0,
            'hedge_fund_pilots': 0,
            'compliance_score': 100.0,
            'risk_utilization': 0.0
        }
        
        self._initialize_monitoring()
    
    def _load_sla_thresholds(self) -> Dict[str, float]:
        """Load SLA thresholds from enterprise configuration."""
        if enterprise_config:
            return {
                'processing_delay_ms': enterprise_config.processing_delay_threshold,
                'trading_error_percentage': enterprise_config.trading_error_threshold,
                'ram_usage_percentage': enterprise_config.ram_usage_threshold
            }
        
        # Default SLA thresholds
        return {
            'processing_delay_ms': 5.0,
            'trading_error_percentage': 0.1,
            'ram_usage_percentage': 85.0
        }
    
    def _initialize_monitoring(self):
        """Initialize monitoring system based on enterprise configuration."""
        if enterprise_config and enterprise_config.monitoring_enabled:
            self.monitoring_enabled = True
            self.real_time_enabled = enterprise_config.real_time_monitoring
            self.logger.info("Enterprise monitoring system initialized")
        else:
            self.logger.info("Monitoring system initialized in basic mode")
    
    def start_monitoring(self):
        """Start the monitoring system."""
        if self.is_running:
            self.logger.warning("Monitoring system already running")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("🚀 Enterprise monitoring system started")
        self._generate_alert("INFO", "SYSTEM", "Monitoring system started", {})
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("⏹️ Enterprise monitoring system stopped")
        self._generate_alert("INFO", "SYSTEM", "Monitoring system stopped", {})
    
    def _monitoring_loop(self):
        """Main monitoring loop that runs in background thread."""
        while self.is_running:
            try:
                # Update performance metrics
                self._update_performance_metrics()
                
                # Check SLA compliance
                self._check_sla_compliance()
                
                # Update enterprise metrics
                self._update_enterprise_metrics()
                
                # Check for alerts
                self._check_alert_conditions()
                
                # Sleep interval based on real-time setting
                sleep_interval = 1 if self.real_time_enabled else 5
                time.sleep(sleep_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)
    
    def _update_performance_metrics(self):
        """Update system performance metrics."""
        # Simulate real-time metrics (in production, get from actual system)
        import random
        
        self.performance_metrics.update({
            'processing_delay_ms': random.uniform(2.0, 8.0),
            'trading_error_percentage': random.uniform(0.01, 0.15),
            'ram_usage_percentage': random.uniform(60.0, 90.0),
            'cpu_usage_percentage': random.uniform(30.0, 80.0),
            'active_positions': random.randint(0, 10),
            'trades_per_second': random.uniform(0.5, 5.0),
            'api_response_time_ms': random.uniform(50.0, 200.0)
        })
    
    def _update_enterprise_metrics(self):
        """Update enterprise-specific metrics."""
        if enterprise_config:
            self.enterprise_metrics.update({
                'institutional_clients': enterprise_config.institutional_clients,
                'total_aum': enterprise_config.target_aum * 0.3,  # Simulated current AUM
                'hedge_fund_pilots': enterprise_config.get('enterprise.deployment.hedge_fund_pilots', 0),
                'compliance_score': 98.5,  # Simulated compliance score
                'risk_utilization': 0.65  # Simulated risk utilization
            })
    
    def _check_sla_compliance(self):
        """Check SLA compliance and record metrics."""
        current_time = datetime.now()
        
        for metric_name, threshold in self.sla_thresholds.items():
            current_value = self.performance_metrics.get(metric_name, 0.0)
            compliant = current_value <= threshold
            
            sla_metric = SLAMetric(
                name=metric_name,
                current_value=current_value,
                threshold=threshold,
                unit=self._get_metric_unit(metric_name),
                compliant=compliant,
                timestamp=current_time
            )
            
            self.metrics_history.append(sla_metric)
            
            # Generate alert if SLA violated
            if not compliant:
                self._generate_alert(
                    level="WARNING",
                    category="SLA",
                    message=f"SLA violation: {metric_name}",
                    details={
                        'metric': metric_name,
                        'current': current_value,
                        'threshold': threshold,
                        'violation_percentage': ((current_value - threshold) / threshold) * 100
                    }
                )
        
        # Keep only last 1000 metrics
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def _get_metric_unit(self, metric_name: str) -> str:
        """Get unit for metric."""
        unit_map = {
            'processing_delay_ms': 'ms',
            'trading_error_percentage': '%',
            'ram_usage_percentage': '%',
            'cpu_usage_percentage': '%',
            'api_response_time_ms': 'ms',
            'trades_per_second': 'tps'
        }
        return unit_map.get(metric_name, '')
    
    def _check_alert_conditions(self):
        """Check various alert conditions."""
        # Check for critical performance issues
        if self.performance_metrics['ram_usage_percentage'] > 95:
            self._generate_alert(
                level="CRITICAL",
                category="PERFORMANCE",
                message="Critical RAM usage detected",
                details={'ram_usage': self.performance_metrics['ram_usage_percentage']}
            )
        
        # Check for trading issues
        if self.performance_metrics['trading_error_percentage'] > 0.2:
            self._generate_alert(
                level="CRITICAL",
                category="TRADING",
                message="High trading error rate detected",
                details={'error_rate': self.performance_metrics['trading_error_percentage']}
            )
        
        # Check enterprise metrics
        if self.enterprise_metrics['compliance_score'] < 95:
            self._generate_alert(
                level="WARNING",
                category="COMPLIANCE",
                message="Compliance score below threshold",
                details={'score': self.enterprise_metrics['compliance_score']}
            )
    
    def _generate_alert(self, level: str, category: str, message: str, details: Dict[str, Any]):
        """Generate and log an alert event."""
        alert = AlertEvent(
            timestamp=datetime.now(),
            level=level,
            category=category,
            message=message,
            details=details
        )
        
        self.alert_history.append(alert)
        
        # Log alert
        log_method = {
            'INFO': self.logger.info,
            'WARNING': self.logger.warning,
            'CRITICAL': self.logger.critical
        }.get(level, self.logger.info)
        
        log_method(f"🚨 [{level}] {category}: {message}")
        
        # Keep only last 500 alerts
        if len(self.alert_history) > 500:
            self.alert_history = self.alert_history[-500:]
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring system status."""
        return {
            'timestamp': datetime.now().isoformat(),
            'monitoring_enabled': self.monitoring_enabled,
            'real_time_enabled': self.real_time_enabled,
            'system_running': self.is_running,
            'performance_metrics': self.performance_metrics.copy(),
            'enterprise_metrics': self.enterprise_metrics.copy(),
            'sla_thresholds': self.sla_thresholds.copy(),
            'recent_alerts': [
                {
                    'timestamp': alert.timestamp.isoformat(),
                    'level': alert.level,
                    'category': alert.category,
                    'message': alert.message,
                    'details': alert.details
                }
                for alert in self.alert_history[-10:]  # Last 10 alerts
            ],
            'sla_compliance': self._get_current_sla_compliance()
        }
    
    def _get_current_sla_compliance(self) -> Dict[str, Any]:
        """Get current SLA compliance status."""
        compliance_status = {}
        
        for metric_name, threshold in self.sla_thresholds.items():
            current_value = self.performance_metrics.get(metric_name, 0.0)
            compliance_status[metric_name] = {
                'current': current_value,
                'threshold': threshold,
                'compliant': current_value <= threshold,
                'compliance_percentage': min(100, (threshold / current_value) * 100) if current_value > 0 else 100
            }
        
        return compliance_status
    
    def get_enterprise_dashboard(self) -> Dict[str, Any]:
        """Get enterprise dashboard data."""
        return {
            'platform_status': 'OPERATIONAL' if self.is_running else 'STOPPED',
            'enterprise_metrics': self.enterprise_metrics.copy(),
            'sla_summary': {
                'total_metrics': len(self.sla_thresholds),
                'compliant_metrics': sum(
                    1 for name, threshold in self.sla_thresholds.items()
                    if self.performance_metrics.get(name, 0) <= threshold
                ),
                'overall_compliance': self._calculate_overall_compliance()
            },
            'alert_summary': {
                'total_alerts_24h': len([
                    a for a in self.alert_history 
                    if a.timestamp > datetime.now() - timedelta(hours=24)
                ]),
                'critical_alerts_24h': len([
                    a for a in self.alert_history 
                    if a.timestamp > datetime.now() - timedelta(hours=24) and a.level == 'CRITICAL'
                ]),
                'latest_alert': self.alert_history[-1] if self.alert_history else None
            },
            'deployment_status': {
                'mode': enterprise_config.deployment_mode if enterprise_config else 'staging',
                'phase': enterprise_config.deployment_phase if enterprise_config else 'initial',
                'target_aum': enterprise_config.target_aum if enterprise_config else 0
            }
        }
    
    def _calculate_overall_compliance(self) -> float:
        """Calculate overall SLA compliance percentage."""
        if not self.sla_thresholds:
            return 100.0
        
        compliant_count = sum(
            1 for name, threshold in self.sla_thresholds.items()
            if self.performance_metrics.get(name, 0) <= threshold
        )
        
        return (compliant_count / len(self.sla_thresholds)) * 100
    
    def configure_thresholds(self, thresholds: Dict[str, float]):
        """Update SLA thresholds."""
        self.sla_thresholds.update(thresholds)
        self.logger.info(f"SLA thresholds updated: {thresholds}")
        
        if enterprise_config:
            enterprise_config.configure_sla_thresholds(**thresholds)
    
    def export_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Export metrics for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_metrics = [
            {
                'timestamp': metric.timestamp.isoformat(),
                'name': metric.name,
                'value': metric.current_value,
                'threshold': metric.threshold,
                'compliant': metric.compliant
            }
            for metric in self.metrics_history
            if metric.timestamp > cutoff_time
        ]
        
        filtered_alerts = [
            {
                'timestamp': alert.timestamp.isoformat(),
                'level': alert.level,
                'category': alert.category,
                'message': alert.message
            }
            for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]
        
        return {
            'export_timestamp': datetime.now().isoformat(),
            'period_hours': hours,
            'metrics': filtered_metrics,
            'alerts': filtered_alerts,
            'summary': {
                'total_metrics': len(filtered_metrics),
                'total_alerts': len(filtered_alerts),
                'alert_levels': {
                    level: len([a for a in filtered_alerts if a['level'] == level])
                    for level in ['INFO', 'WARNING', 'CRITICAL']
                }
            }
        }


# Global enterprise monitoring instance
enterprise_monitor = EnterpriseMonitoringSystem()