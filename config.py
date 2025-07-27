"""
Complete Configuration Management for Advanced Trading System
Handles environment variables, API keys, trading parameters, and system settings
"""

import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class TradingConfig:
    """Central configuration management for the trading system"""
    
    def __init__(self):
        self.load_config()
    
    def load_config(self):
        """Load all configuration settings"""
        
        # API Keys and Exchange Settings
        self.BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
        self.BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
        self.BINANCE_TESTNET = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
        
        # Trading Parameters
        self.DEFAULT_RISK_LEVEL = float(os.getenv('DEFAULT_RISK_LEVEL', '0.02'))  # 2% risk per trade
        self.MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '0.1'))    # 10% of portfolio
        self.STOP_LOSS_PERCENTAGE = float(os.getenv('STOP_LOSS_PERCENTAGE', '0.05'))  # 5% stop loss
        self.TAKE_PROFIT_PERCENTAGE = float(os.getenv('TAKE_PROFIT_PERCENTAGE', '0.1'))  # 10% take profit
        
        # ML Model Hyperparameters
        self.ML_LEARNING_RATE = float(os.getenv('ML_LEARNING_RATE', '0.001'))
        self.ML_BATCH_SIZE = int(os.getenv('ML_BATCH_SIZE', '32'))
        self.ML_EPOCHS = int(os.getenv('ML_EPOCHS', '100'))
        self.ML_VALIDATION_SPLIT = float(os.getenv('ML_VALIDATION_SPLIT', '0.2'))
        
        # Risk Management
        self.MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', '0.05'))  # 5% max daily loss
        self.VOLATILITY_THRESHOLD = float(os.getenv('VOLATILITY_THRESHOLD', '0.8'))
        self.NEWS_IMPACT_WEIGHT = float(os.getenv('NEWS_IMPACT_WEIGHT', '0.6'))
        
        # Database and Redis Settings
        self.REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
        self.REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
        self.REDIS_DB = int(os.getenv('REDIS_DB', '0'))
        
        # API Server Settings
        self.API_HOST = os.getenv('API_HOST', '0.0.0.0')
        self.API_PORT = int(os.getenv('API_PORT', '8000'))
        self.API_DEBUG = os.getenv('API_DEBUG', 'false').lower() == 'true'
        
        # Logging Configuration
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        self.LOG_FILE = os.getenv('LOG_FILE', 'trading_system.log')
        self.LOG_MAX_BYTES = int(os.getenv('LOG_MAX_BYTES', '10485760'))  # 10MB
        self.LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', '5'))
        
        # Data Feed Settings
        self.UPDATE_INTERVAL = int(os.getenv('UPDATE_INTERVAL', '1'))  # seconds
        self.SYMBOLS = os.getenv('SYMBOLS', 'BTCUSDT,ETHUSDT,ADAUSDT').split(',')
        self.TIMEFRAMES = os.getenv('TIMEFRAMES', '1m,5m,15m,1h').split(',')
        
        # Sentiment Analysis
        self.NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
        self.TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN', '')
        
        # Dashboard Settings
        self.DASHBOARD_HOST = os.getenv('DASHBOARD_HOST', '0.0.0.0')
        self.DASHBOARD_PORT = int(os.getenv('DASHBOARD_PORT', '8050'))
        self.DASHBOARD_DEBUG = os.getenv('DASHBOARD_DEBUG', 'false').lower() == 'true'
    
    def get_exchange_config(self) -> Dict[str, Any]:
        """Get exchange configuration for CCXT"""
        return {
            'apiKey': self.BINANCE_API_KEY,
            'secret': self.BINANCE_SECRET_KEY,
            'sandbox': self.BINANCE_TESTNET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'  # spot, margin, future
            }
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                },
            },
            'handlers': {
                'default': {
                    'level': self.LOG_LEVEL,
                    'formatter': 'standard',
                    'class': 'logging.StreamHandler',
                },
                'rotate_file': {
                    'level': self.LOG_LEVEL,
                    'formatter': 'standard',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': self.LOG_FILE,
                    'maxBytes': self.LOG_MAX_BYTES,
                    'backupCount': self.LOG_BACKUP_COUNT,
                }
            },
            'loggers': {
                '': {
                    'handlers': ['default', 'rotate_file'],
                    'level': self.LOG_LEVEL,
                    'propagate': False
                }
            }
        }
    
    def validate_config(self) -> bool:
        """Validate critical configuration settings"""
        errors = []
        
        if not self.BINANCE_API_KEY and not self.BINANCE_TESTNET:
            errors.append("BINANCE_API_KEY is required for live trading")
        
        if not self.BINANCE_SECRET_KEY and not self.BINANCE_TESTNET:
            errors.append("BINANCE_SECRET_KEY is required for live trading")
        
        if self.DEFAULT_RISK_LEVEL <= 0 or self.DEFAULT_RISK_LEVEL > 1:
            errors.append("DEFAULT_RISK_LEVEL must be between 0 and 1")
        
        if self.MAX_POSITION_SIZE <= 0 or self.MAX_POSITION_SIZE > 1:
            errors.append("MAX_POSITION_SIZE must be between 0 and 1")
        
        if errors:
            for error in errors:
                logging.error(f"Configuration error: {error}")
            return False
        
        return True
    
    def get_persona_config(self) -> Dict[str, Any]:
        """Get OmniPersona configuration"""
        return {
            'default_persona': os.getenv('DEFAULT_PERSONA', 'neutral'),
            'adaptive_mode': os.getenv('ADAPTIVE_PERSONA', 'true').lower() == 'true',
            'risk_tolerance': self.DEFAULT_RISK_LEVEL,
        }
    
    def get_ml_config(self) -> Dict[str, Any]:
        """Get machine learning configuration"""
        return {
            'learning_rate': self.ML_LEARNING_RATE,
            'batch_size': self.ML_BATCH_SIZE,
            'epochs': self.ML_EPOCHS,
            'validation_split': self.ML_VALIDATION_SPLIT,
            'model_save_path': os.getenv('MODEL_SAVE_PATH', './models/'),
            'feature_count': int(os.getenv('FEATURE_COUNT', '20')),
        }

# Global configuration instance
config = TradingConfig()

def get_config() -> TradingConfig:
    """Get the global configuration instance"""
    return config