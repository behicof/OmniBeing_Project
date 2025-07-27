"""
Centralized configuration management for the OmniBeing Trading System.
Loads configuration from YAML files and provides easy access to settings.
"""

import yaml
import os
from typing import Dict, Any, Optional


class Config:
    """Configuration manager that loads settings from YAML files."""
    
    def __init__(self, config_file: str = "config.yaml"):
        """
        Initialize configuration with default or specified config file.
        
        Args:
            config_file: Path to the YAML configuration file
        """
        self.config_file = config_file
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file {self.config_file} not found")
        
        with open(self.config_file, 'r') as file:
            return yaml.safe_load(file)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key with optional default.
        
        Args:
            key: Configuration key (supports nested keys with '.')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key.
        
        Args:
            key: Configuration key (supports nested keys with '.')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self) -> None:
        """Save current configuration to file."""
        with open(self.config_file, 'w') as file:
            yaml.dump(self._config, file, default_flow_style=False)
    
    # Trading Configuration Properties
    @property
    def trading_instrument(self) -> str:
        """Get trading instrument."""
        return self.get('trading.instrument', 'XAUUSD')
    
    @property
    def trading_timeframe(self) -> str:
        """Get trading timeframe."""
        return self.get('trading.timeframe', 'H1')
    
    @property
    def initial_capital(self) -> float:
        """Get initial capital."""
        return self.get('trading.initial_capital', 10000.0)
    
    @property
    def risk_percentage(self) -> float:
        """Get risk percentage."""
        return self.get('trading.risk_percentage', 1.5)
    
    @property
    def max_positions(self) -> int:
        """Get maximum positions."""
        return self.get('trading.max_positions', 3)
    
    # Model Configuration Properties
    @property
    def sequence_length(self) -> int:
        """Get model sequence length."""
        return self.get('model.sequence_length', 60)
    
    @property
    def epochs(self) -> int:
        """Get training epochs."""
        return self.get('model.epochs', 50)
    
    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self.get('model.batch_size', 32)
    
    # API Keys
    @property
    def market_data_api_key(self) -> Optional[str]:
        """Get market data API key."""
        return self.get('api_keys.market_data_api_key')
    
    @property
    def binance_api_key(self) -> Optional[str]:
        """Get Binance API key."""
        return self.get('api_keys.binance_api_key')
    
    @property
    def binance_secret_key(self) -> Optional[str]:
        """Get Binance secret key."""
        return self.get('api_keys.binance_secret_key')
    
    # Risk Management Settings
    @property
    def stop_loss_percentage(self) -> float:
        """Get stop loss percentage."""
        return self.get('risk_management.stop_loss_percentage', 2.0)
    
    @property
    def take_profit_percentage(self) -> float:
        """Get take profit percentage."""
        return self.get('risk_management.take_profit_percentage', 3.0)
    
    @property
    def volatility_threshold(self) -> float:
        """Get volatility threshold."""
        return self.get('risk_management.volatility_threshold', 0.8)


# Global configuration instance
config = Config()