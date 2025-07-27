"""
Enterprise Configuration Management for Production Deployment.
Extends the existing config system with production-grade features.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import hashlib
import base64
from cryptography.fernet import Fernet
from config import Config

class Environment(Enum):
    """Environment types for deployment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class DatabaseConfig:
    """Database configuration for production."""
    host: str
    port: int
    database: str
    username: str
    password_encrypted: str
    ssl_mode: str = "require"
    pool_size: int = 20
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600

@dataclass
class RedisConfig:
    """Redis configuration for caching."""
    host: str
    port: int = 6379
    database: int = 0
    password_encrypted: str = ""
    ssl: bool = True
    socket_timeout: int = 5
    max_connections: int = 100

@dataclass
class SecurityConfig:
    """Security configuration."""
    jwt_secret_key_encrypted: str
    api_rate_limit: int = 1000
    session_timeout: int = 3600
    max_login_attempts: int = 5
    password_min_length: int = 12
    require_mfa: bool = True
    allowed_origins: List[str] = None

@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration."""
    prometheus_port: int = 9090
    grafana_port: int = 3000
    alert_email: str = ""
    alert_slack_webhook_encrypted: str = ""
    log_level: str = "INFO"
    metrics_retention_days: int = 30

@dataclass
class ScalingConfig:
    """Auto-scaling configuration."""
    min_instances: int = 2
    max_instances: int = 10
    cpu_threshold: float = 70.0
    memory_threshold: float = 80.0
    scale_up_cooldown: int = 300
    scale_down_cooldown: int = 600
    health_check_interval: int = 30

class ProductionConfig(Config):
    """
    Production-grade configuration management with security and scaling.
    Extends the base Config class with enterprise features.
    """
    
    def __init__(self, 
                 config_file: str = "config.yaml",
                 environment: Environment = Environment.PRODUCTION,
                 encryption_key: Optional[str] = None):
        """
        Initialize production configuration.
        
        Args:
            config_file: Base configuration file
            environment: Deployment environment
            encryption_key: Encryption key for sensitive data
        """
        super().__init__(config_file)
        self.environment = environment
        self.encryption_key = encryption_key or self._get_encryption_key()
        self.cipher = Fernet(self.encryption_key.encode()) if self.encryption_key else None
        
        # Load environment-specific configuration
        self.env_config = self._load_environment_config()
        
        # Initialize production components
        self.database = self._load_database_config()
        self.redis = self._load_redis_config()
        self.security = self._load_security_config()
        self.monitoring = self._load_monitoring_config()
        self.scaling = self._load_scaling_config()
        
        # Validate configuration
        self._validate_production_config()
    
    def _get_encryption_key(self) -> Optional[str]:
        """Get encryption key from environment or generate new one."""
        key = os.getenv('OMNIBEING_ENCRYPTION_KEY')
        if not key:
            key = Fernet.generate_key().decode()
            print(f"Generated new encryption key. Set OMNIBEING_ENCRYPTION_KEY={key}")
        return key
    
    def _load_environment_config(self) -> Dict[str, Any]:
        """Load environment-specific configuration."""
        env_file = f"config_{self.environment.value}.yaml"
        if os.path.exists(env_file):
            with open(env_file, 'r') as file:
                return yaml.safe_load(file)
        return {}
    
    def _load_database_config(self) -> DatabaseConfig:
        """Load database configuration."""
        db_config = self.env_config.get('database', {})
        return DatabaseConfig(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 5432),
            database=db_config.get('database', 'omnibeing_trading'),
            username=db_config.get('username', 'omnibeing'),
            password_encrypted=db_config.get('password_encrypted', ''),
            ssl_mode=db_config.get('ssl_mode', 'require'),
            pool_size=db_config.get('pool_size', 20),
            max_overflow=db_config.get('max_overflow', 10),
            pool_timeout=db_config.get('pool_timeout', 30),
            pool_recycle=db_config.get('pool_recycle', 3600)
        )
    
    def _load_redis_config(self) -> RedisConfig:
        """Load Redis configuration."""
        redis_config = self.env_config.get('redis', {})
        return RedisConfig(
            host=redis_config.get('host', 'localhost'),
            port=redis_config.get('port', 6379),
            database=redis_config.get('database', 0),
            password_encrypted=redis_config.get('password_encrypted', ''),
            ssl=redis_config.get('ssl', True),
            socket_timeout=redis_config.get('socket_timeout', 5),
            max_connections=redis_config.get('max_connections', 100)
        )
    
    def _load_security_config(self) -> SecurityConfig:
        """Load security configuration."""
        security_config = self.env_config.get('security', {})
        return SecurityConfig(
            jwt_secret_key_encrypted=security_config.get('jwt_secret_key_encrypted', ''),
            api_rate_limit=security_config.get('api_rate_limit', 1000),
            session_timeout=security_config.get('session_timeout', 3600),
            max_login_attempts=security_config.get('max_login_attempts', 5),
            password_min_length=security_config.get('password_min_length', 12),
            require_mfa=security_config.get('require_mfa', True),
            allowed_origins=security_config.get('allowed_origins', ['https://omnibeing.com'])
        )
    
    def _load_monitoring_config(self) -> MonitoringConfig:
        """Load monitoring configuration."""
        monitoring_config = self.env_config.get('monitoring', {})
        return MonitoringConfig(
            prometheus_port=monitoring_config.get('prometheus_port', 9090),
            grafana_port=monitoring_config.get('grafana_port', 3000),
            alert_email=monitoring_config.get('alert_email', ''),
            alert_slack_webhook_encrypted=monitoring_config.get('alert_slack_webhook_encrypted', ''),
            log_level=monitoring_config.get('log_level', 'INFO'),
            metrics_retention_days=monitoring_config.get('metrics_retention_days', 30)
        )
    
    def _load_scaling_config(self) -> ScalingConfig:
        """Load auto-scaling configuration."""
        scaling_config = self.env_config.get('scaling', {})
        return ScalingConfig(
            min_instances=scaling_config.get('min_instances', 2),
            max_instances=scaling_config.get('max_instances', 10),
            cpu_threshold=scaling_config.get('cpu_threshold', 70.0),
            memory_threshold=scaling_config.get('memory_threshold', 80.0),
            scale_up_cooldown=scaling_config.get('scale_up_cooldown', 300),
            scale_down_cooldown=scaling_config.get('scale_down_cooldown', 600),
            health_check_interval=scaling_config.get('health_check_interval', 30)
        )
    
    def encrypt_value(self, value: str) -> str:
        """Encrypt a sensitive value."""
        if not self.cipher:
            raise ValueError("Encryption not available - no key provided")
        return self.cipher.encrypt(value.encode()).decode()
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a sensitive value."""
        if not self.cipher:
            raise ValueError("Decryption not available - no key provided")
        return self.cipher.decrypt(encrypted_value.encode()).decode()
    
    def get_database_url(self) -> str:
        """Get database connection URL with decrypted password."""
        password = self.decrypt_value(self.database.password_encrypted) if self.database.password_encrypted else ''
        return f"postgresql://{self.database.username}:{password}@{self.database.host}:{self.database.port}/{self.database.database}?sslmode={self.database.ssl_mode}"
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL with decrypted password."""
        password = self.decrypt_value(self.redis.password_encrypted) if self.redis.password_encrypted else ''
        protocol = "rediss" if self.redis.ssl else "redis"
        auth = f":{password}@" if password else ""
        return f"{protocol}://{auth}{self.redis.host}:{self.redis.port}/{self.redis.database}"
    
    def get_jwt_secret(self) -> str:
        """Get JWT secret key (decrypted)."""
        return self.decrypt_value(self.security.jwt_secret_key_encrypted) if self.security.jwt_secret_key_encrypted else ''
    
    def get_slack_webhook(self) -> str:
        """Get Slack webhook URL (decrypted)."""
        return self.decrypt_value(self.monitoring.alert_slack_webhook_encrypted) if self.monitoring.alert_slack_webhook_encrypted else ''
    
    def _validate_production_config(self):
        """Validate production configuration."""
        errors = []
        
        # Database validation
        if not self.database.host:
            errors.append("Database host is required")
        if not self.database.database:
            errors.append("Database name is required")
        if not self.database.username:
            errors.append("Database username is required")
        
        # Security validation
        if self.environment == Environment.PRODUCTION:
            if not self.security.jwt_secret_key_encrypted:
                errors.append("JWT secret key is required for production")
            if not self.security.require_mfa:
                errors.append("MFA should be required for production")
        
        # Redis validation
        if not self.redis.host:
            errors.append("Redis host is required")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {', '.join(errors)}")
    
    def export_config(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        config_dict = {
            'environment': self.environment.value,
            'database': {
                'host': self.database.host,
                'port': self.database.port,
                'database': self.database.database,
                'username': self.database.username,
                'ssl_mode': self.database.ssl_mode,
                'pool_size': self.database.pool_size,
                'max_overflow': self.database.max_overflow,
                'pool_timeout': self.database.pool_timeout,
                'pool_recycle': self.database.pool_recycle
            },
            'redis': {
                'host': self.redis.host,
                'port': self.redis.port,
                'database': self.redis.database,
                'ssl': self.redis.ssl,
                'socket_timeout': self.redis.socket_timeout,
                'max_connections': self.redis.max_connections
            },
            'security': {
                'api_rate_limit': self.security.api_rate_limit,
                'session_timeout': self.security.session_timeout,
                'max_login_attempts': self.security.max_login_attempts,
                'password_min_length': self.security.password_min_length,
                'require_mfa': self.security.require_mfa,
                'allowed_origins': self.security.allowed_origins
            },
            'monitoring': {
                'prometheus_port': self.monitoring.prometheus_port,
                'grafana_port': self.monitoring.grafana_port,
                'alert_email': self.monitoring.alert_email,
                'log_level': self.monitoring.log_level,
                'metrics_retention_days': self.monitoring.metrics_retention_days
            },
            'scaling': {
                'min_instances': self.scaling.min_instances,
                'max_instances': self.scaling.max_instances,
                'cpu_threshold': self.scaling.cpu_threshold,
                'memory_threshold': self.scaling.memory_threshold,
                'scale_up_cooldown': self.scaling.scale_up_cooldown,
                'scale_down_cooldown': self.scaling.scale_down_cooldown,
                'health_check_interval': self.scaling.health_check_interval
            }
        }
        
        if include_secrets:
            config_dict['database']['password_encrypted'] = self.database.password_encrypted
            config_dict['redis']['password_encrypted'] = self.redis.password_encrypted
            config_dict['security']['jwt_secret_key_encrypted'] = self.security.jwt_secret_key_encrypted
            config_dict['monitoring']['alert_slack_webhook_encrypted'] = self.monitoring.alert_slack_webhook_encrypted
        
        return config_dict
    
    def save_environment_config(self, file_path: Optional[str] = None):
        """Save current configuration to environment-specific file."""
        if not file_path:
            file_path = f"config_{self.environment.value}.yaml"
        
        config_dict = self.export_config(include_secrets=True)
        with open(file_path, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False)


# Create global production config instance
production_config = None

def get_production_config(environment: Environment = Environment.PRODUCTION) -> ProductionConfig:
    """Get the global production configuration instance."""
    global production_config
    if production_config is None:
        production_config = ProductionConfig(environment=environment)
    return production_config

def initialize_production_config(config_file: str = "config.yaml",
                                environment: Environment = Environment.PRODUCTION,
                                encryption_key: Optional[str] = None) -> ProductionConfig:
    """Initialize production configuration with custom parameters."""
    global production_config
    production_config = ProductionConfig(
        config_file=config_file,
        environment=environment,
        encryption_key=encryption_key
    )
    return production_config