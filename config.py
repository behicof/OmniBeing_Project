"""
Centralized configuration management for the OmniBeing Trading System.
Loads configuration from YAML files and provides easy access to settings.
Enhanced with environment variables and basic logging setup.
"""

import yaml
import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


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
            config_data = yaml.safe_load(file)
        
        # Override with environment variables if they exist
        self._load_environment_overrides(config_data)
        
        return config_data
    
    def _load_environment_overrides(self, config_data: Dict[str, Any]):
        """Load environment variable overrides for sensitive data."""
        env_mappings = {
            'MARKET_DATA_API_KEY': 'api_keys.market_data_api_key',
            'BINANCE_API_KEY': 'api_keys.binance_api_key', 
            'BINANCE_SECRET_KEY': 'api_keys.binance_secret_key',
            'TRADING_INSTRUMENT': 'trading.instrument',
            'INITIAL_CAPITAL': 'trading.initial_capital',
            'RISK_PERCENTAGE': 'trading.risk_percentage',
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                # Convert numeric values
                if env_var in ['INITIAL_CAPITAL', 'RISK_PERCENTAGE']:
                    try:
                        env_value = float(env_value)
                    except ValueError:
                        continue
                
                # Set the nested config value
                self.set(config_path, env_value)
    
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
    
    def setup_logging(self, log_level: str = 'INFO') -> logging.Logger:
        """
        Setup basic logging configuration.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            
        Returns:
            Configured logger instance
        """
        # Create logs directory if it doesn't exist
        log_dir = self.get('paths.logs_path', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'trading_system.log')),
                logging.StreamHandler()
            ]
        )
        
        logger = logging.getLogger('OmniBeing_Trading')
        logger.info(f"Logging initialized - Level: {log_level}")
        
        return logger
    
    def get_trading_parameters(self) -> Dict[str, Any]:
        """Get all trading parameters in one dictionary."""
        return {
            'instrument': self.trading_instrument,
            'timeframe': self.trading_timeframe,
            'initial_capital': self.initial_capital,
            'risk_percentage': self.risk_percentage,
            'max_positions': self.max_positions,
            'stop_loss_percentage': self.stop_loss_percentage,
            'take_profit_percentage': self.take_profit_percentage,
            'volatility_threshold': self.volatility_threshold
        }


# Global configuration instance
config = Config()