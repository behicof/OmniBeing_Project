"""
Enterprise Configuration Extensions for OmniBeing Platform.
Provides enterprise-grade settings and institutional configurations.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import logging

from config import Config


class EnterpriseConfig(Config):
    """Enhanced configuration for enterprise deployment scenarios."""
    
    def __init__(self, config_file: str = "config.yaml"):
        """Initialize enterprise configuration with extended capabilities."""
        super().__init__(config_file)
        self._setup_enterprise_defaults()
        self.logger = logging.getLogger(__name__)
    
    def _setup_enterprise_defaults(self):
        """Setup default enterprise configuration values."""
        enterprise_defaults = {
            'enterprise': {
                'deployment': {
                    'mode': 'production',
                    'phase': 'initial',
                    'target_aum': 50000000,  # $50M target
                    'institutional_clients': 0,
                    'hedge_fund_pilots': 0
                },
                'monitoring': {
                    'enabled': False,
                    'real_time': False,
                    'processing_delay_ms': 5,
                    'trading_error_percentage': 0.1,
                    'ram_usage_percentage': 85,
                    'sla_monitoring': True
                },
                'security': {
                    'penetration_testing': False,
                    'honeypot_enabled': False,
                    'geographic_sharding': False,
                    'iso27001_compliance': False
                },
                'optimization': {
                    'auto_scaling': False,
                    'cost_reduction_target': 30,  # 30% cost reduction
                    'resource_allocation': {
                        'market_data_processing': 35,
                        'risk_engine': 25,
                        'ai_prediction': 20,
                        'compliance': 15,
                        'reporting': 5
                    }
                },
                'integrations': {
                    'prime_broker_api': False,
                    'bloomberg_terminal': False,
                    'primexm_liquidity': False,
                    'chainalysis_api': False
                },
                'modules': {
                    'risk_manager': {
                        'enabled': False,
                        'sla_monitoring': False,
                        'institutional_grade': True
                    },
                    'compliance': {
                        'enabled': False,
                        'iso27001_mode': False,
                        'regulatory_reporting': True
                    },
                    'live_trading': {
                        'enabled': False,
                        'institutional_mode': False,
                        'high_frequency': True
                    },
                    'arbitrage': {
                        'enabled': False,
                        'multi_exchange': False,
                        'latency_optimization': True
                    },
                    'ai_strategies': {
                        'enabled': False,
                        'deep_learning': False,
                        'ensemble_models': True
                    },
                    'reporting': {
                        'enabled': False,
                        'institutional_format': False,
                        'real_time_analytics': True
                    }
                }
            }
        }
        
        # Merge defaults with existing config
        for section, values in enterprise_defaults.items():
            if section not in self._config:
                self._config[section] = {}
            
            self._merge_dict(self._config[section], values)
    
    def _merge_dict(self, base: Dict[str, Any], overlay: Dict[str, Any]):
        """Recursively merge configuration dictionaries."""
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_dict(base[key], value)
            elif key not in base:
                base[key] = value
    
    # Enterprise Deployment Properties
    @property
    def deployment_mode(self) -> str:
        """Get deployment mode (staging/production)."""
        return self.get('enterprise.deployment.mode', 'staging')
    
    @property
    def deployment_phase(self) -> str:
        """Get current deployment phase."""
        return self.get('enterprise.deployment.phase', 'initial')
    
    @property
    def target_aum(self) -> int:
        """Get target assets under management."""
        return self.get('enterprise.deployment.target_aum', 50000000)
    
    @property
    def institutional_clients(self) -> int:
        """Get number of institutional clients."""
        return self.get('enterprise.deployment.institutional_clients', 0)
    
    # Enterprise Monitoring Properties
    @property
    def monitoring_enabled(self) -> bool:
        """Check if enterprise monitoring is enabled."""
        return self.get('enterprise.monitoring.enabled', False)
    
    @property
    def real_time_monitoring(self) -> bool:
        """Check if real-time monitoring is enabled."""
        return self.get('enterprise.monitoring.real_time', False)
    
    @property
    def processing_delay_threshold(self) -> float:
        """Get processing delay SLA threshold in milliseconds."""
        return self.get('enterprise.monitoring.processing_delay_ms', 5)
    
    @property
    def trading_error_threshold(self) -> float:
        """Get trading error SLA threshold as percentage."""
        return self.get('enterprise.monitoring.trading_error_percentage', 0.1)
    
    @property
    def ram_usage_threshold(self) -> float:
        """Get RAM usage SLA threshold as percentage."""
        return self.get('enterprise.monitoring.ram_usage_percentage', 85)
    
    # Enterprise Security Properties
    @property
    def penetration_testing_enabled(self) -> bool:
        """Check if penetration testing is configured."""
        return self.get('enterprise.security.penetration_testing', False)
    
    @property
    def honeypot_enabled(self) -> bool:
        """Check if honeypot security system is enabled."""
        return self.get('enterprise.security.honeypot_enabled', False)
    
    @property
    def geographic_sharding_enabled(self) -> bool:
        """Check if geographic data sharding is enabled."""
        return self.get('enterprise.security.geographic_sharding', False)
    
    @property
    def iso27001_compliance(self) -> bool:
        """Check if ISO 27001 compliance mode is enabled."""
        return self.get('enterprise.security.iso27001_compliance', False)
    
    # Enterprise Optimization Properties
    @property
    def auto_scaling_enabled(self) -> bool:
        """Check if auto-scaling is enabled."""
        return self.get('enterprise.optimization.auto_scaling', False)
    
    @property
    def cost_reduction_target(self) -> int:
        """Get cost reduction target percentage."""
        return self.get('enterprise.optimization.cost_reduction_target', 30)
    
    @property
    def resource_allocation(self) -> Dict[str, int]:
        """Get resource allocation percentages."""
        return self.get('enterprise.optimization.resource_allocation', {})
    
    # Enterprise Integration Properties
    @property
    def prime_broker_api_enabled(self) -> bool:
        """Check if Prime Broker API integration is enabled."""
        return self.get('enterprise.integrations.prime_broker_api', False)
    
    @property
    def bloomberg_terminal_enabled(self) -> bool:
        """Check if Bloomberg Terminal integration is enabled."""
        return self.get('enterprise.integrations.bloomberg_terminal', False)
    
    @property
    def primexm_liquidity_enabled(self) -> bool:
        """Check if PrimeXM liquidity bridge is enabled."""
        return self.get('enterprise.integrations.primexm_liquidity', False)
    
    @property
    def chainalysis_api_enabled(self) -> bool:
        """Check if Chainalysis API integration is enabled."""
        return self.get('enterprise.integrations.chainalysis_api', False)
    
    # Enterprise Module Status Methods
    def is_module_enabled(self, module: str) -> bool:
        """Check if enterprise module is enabled."""
        return self.get(f'enterprise.modules.{module}.enabled', False)
    
    def enable_enterprise_module(self, module: str, **kwargs):
        """Enable enterprise module with specific settings."""
        base_key = f'enterprise.modules.{module}'
        self.set(f'{base_key}.enabled', True)
        
        for setting, value in kwargs.items():
            self.set(f'{base_key}.{setting}', value)
    
    def get_enterprise_status(self) -> Dict[str, Any]:
        """Get comprehensive enterprise configuration status."""
        return {
            'deployment': {
                'mode': self.deployment_mode,
                'phase': self.deployment_phase,
                'target_aum': self.target_aum,
                'institutional_clients': self.institutional_clients
            },
            'monitoring': {
                'enabled': self.monitoring_enabled,
                'real_time': self.real_time_monitoring,
                'thresholds': {
                    'processing_delay_ms': self.processing_delay_threshold,
                    'trading_error_percentage': self.trading_error_threshold,
                    'ram_usage_percentage': self.ram_usage_threshold
                }
            },
            'security': {
                'penetration_testing': self.penetration_testing_enabled,
                'honeypot': self.honeypot_enabled,
                'geographic_sharding': self.geographic_sharding_enabled,
                'iso27001_compliance': self.iso27001_compliance
            },
            'optimization': {
                'auto_scaling': self.auto_scaling_enabled,
                'cost_reduction_target': self.cost_reduction_target,
                'resource_allocation': self.resource_allocation
            },
            'integrations': {
                'prime_broker_api': self.prime_broker_api_enabled,
                'bloomberg_terminal': self.bloomberg_terminal_enabled,
                'primexm_liquidity': self.primexm_liquidity_enabled,
                'chainalysis_api': self.chainalysis_api_enabled
            },
            'modules': {
                module: self.is_module_enabled(module)
                for module in ['risk_manager', 'compliance', 'live_trading', 
                              'arbitrage', 'ai_strategies', 'reporting']
            }
        }
    
    def update_deployment_phase(self, phase: str):
        """Update deployment phase with validation."""
        valid_phases = ['initial', 'phase1', 'phase2', 'phase3', 'production']
        if phase in valid_phases:
            self.set('enterprise.deployment.phase', phase)
            self.logger.info(f"Deployment phase updated to: {phase}")
        else:
            raise ValueError(f"Invalid deployment phase: {phase}")
    
    def configure_sla_thresholds(self, **thresholds):
        """Configure SLA monitoring thresholds."""
        for threshold, value in thresholds.items():
            if threshold in ['processing_delay_ms', 'trading_error_percentage', 'ram_usage_percentage']:
                self.set(f'enterprise.monitoring.{threshold}', value)
                self.logger.info(f"SLA threshold updated - {threshold}: {value}")
    
    def enable_enterprise_features(self, features: Dict[str, bool]):
        """Enable/disable multiple enterprise features."""
        for feature, enabled in features.items():
            if '.' in feature:
                self.set(f'enterprise.{feature}', enabled)
            else:
                self.set(f'enterprise.{feature}.enabled', enabled)
        
        self.save()


# Global enterprise configuration instance
enterprise_config = EnterpriseConfig()