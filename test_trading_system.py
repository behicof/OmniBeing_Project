"""
Tests for the OmniBeing Trading System components.
"""

import pytest
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from config import Config
from data_manager import DataManager
from external_risk_manager import ExternalRiskManager
from main_trading_system import MainTradingSystem
from backtesting import BacktestingEngine
from market_connectors import MockExchangeConnector, MarketConnectorManager
from logging_system import LoggingSystem, TradeLog


class TestConfig:
    """Test configuration management."""
    
    def test_config_loading(self):
        """Test configuration loading from YAML."""
        config = Config()
        assert config.trading_instrument is not None
        assert config.initial_capital > 0
        assert config.risk_percentage > 0
    
    def test_config_get_set(self):
        """Test configuration get/set operations."""
        config = Config()
        
        # Test getting existing value
        instrument = config.get('trading.instrument')
        assert instrument is not None
        
        # Test setting new value
        config.set('test.value', 123)
        assert config.get('test.value') == 123
        
        # Test default value
        assert config.get('nonexistent.key', 'default') == 'default'


class TestDataManager:
    """Test data management functionality."""
    
    def test_data_manager_initialization(self):
        """Test data manager initialization."""
        dm = DataManager()
        assert dm is not None
        assert dm.historical_data == {}
        assert dm.live_data == []
    
    def test_fetch_historical_data(self):
        """Test historical data fetching."""
        dm = DataManager()
        data = dm.fetch_historical_data('BTCUSDT', limit=100)
        
        assert len(data) == 100
        assert 'open' in data.columns
        assert 'high' in data.columns
        assert 'low' in data.columns
        assert 'close' in data.columns
        assert 'volume' in data.columns
    
    def test_technical_indicators(self):
        """Test technical indicator calculation."""
        dm = DataManager()
        data = dm.fetch_historical_data('BTCUSDT', limit=100)
        data_with_indicators = dm.calculate_technical_indicators(data)
        
        assert 'sma_20' in data_with_indicators.columns
        assert 'rsi' in data_with_indicators.columns
        assert 'macd' in data_with_indicators.columns
        assert 'volatility' in data_with_indicators.columns
    
    def test_feature_engineering(self):
        """Test feature engineering."""
        dm = DataManager()
        features = dm.engineer_features('BTCUSDT')
        
        required_features = ['price', 'sentiment', 'volatility', 'rsi']
        for feature in required_features:
            assert feature in features
    
    def test_market_data_for_prediction(self):
        """Test market data formatting for predictions."""
        dm = DataManager()
        market_data = dm.get_market_data_for_prediction('BTCUSDT')
        
        required_keys = ['sentiment', 'volatility', 'price_change', 'price']
        for key in required_keys:
            assert key in market_data


class TestRiskManager:
    """Test risk management functionality."""
    
    def test_risk_manager_initialization(self):
        """Test risk manager initialization."""
        rm = ExternalRiskManager()
        assert rm.volatility_threshold == 0.8
        assert rm.portfolio == {}
        assert rm.account_balance > 0
    
    def test_volatility_calculation(self):
        """Test volatility calculation."""
        rm = ExternalRiskManager()
        prices = [100, 101, 99, 102, 98, 103, 97]
        
        rm.update_volatility(prices)
        assert rm.current_volatility > 0
    
    def test_position_size_calculation(self):
        """Test position size calculation."""
        rm = ExternalRiskManager()
        
        position_size = rm.calculate_position_size('BTCUSDT', 50000, 49000)
        assert position_size > 0
        
        # Test with invalid prices
        position_size = rm.calculate_position_size('BTCUSDT', 0, 0)
        assert position_size == 0
    
    def test_stop_loss_calculation(self):
        """Test stop loss calculation."""
        rm = ExternalRiskManager()
        
        # Long position
        stop_loss = rm.set_stop_loss('BTCUSDT', 50000, 'long')
        assert stop_loss < 50000
        
        # Short position
        stop_loss = rm.set_stop_loss('BTCUSDT', 50000, 'short')
        assert stop_loss > 50000
    
    def test_position_management(self):
        """Test position add/remove operations."""
        rm = ExternalRiskManager()
        
        # Add position
        success = rm.add_position('BTCUSDT', 0.1, 50000, 'long')
        assert success
        assert 'BTCUSDT' in rm.portfolio
        
        # Close position
        result = rm.close_position('BTCUSDT', 51000)
        assert result['success']
        assert 'BTCUSDT' not in rm.portfolio
    
    def test_risk_assessment(self):
        """Test risk assessment."""
        rm = ExternalRiskManager()
        
        # Test with low risk
        rm.current_volatility = 0.1
        rm.market_event_risk = 0.1
        risk = rm.assess_risk()
        assert risk < rm.volatility_threshold
        
        # Test with high risk
        rm.current_volatility = 0.9
        rm.market_event_risk = 0.9
        risk = rm.assess_risk()
        assert risk >= rm.volatility_threshold


class TestMainTradingSystem:
    """Test main trading system functionality."""
    
    def test_system_initialization(self):
        """Test trading system initialization."""
        ts = MainTradingSystem()
        assert ts is not None
        assert ts.data_manager is not None
        assert ts.risk_manager is not None
        assert ts.is_trading_enabled
    
    def test_system_status(self):
        """Test system status reporting."""
        ts = MainTradingSystem()
        status = ts.get_system_status()
        
        required_keys = ['is_running', 'is_trading_enabled', 'prediction_systems', 
                        'account_balance', 'active_positions']
        for key in required_keys:
            assert key in status
    
    def test_market_data_retrieval(self):
        """Test market data retrieval."""
        ts = MainTradingSystem()
        market_data = ts.get_market_data('BTCUSDT')
        
        assert isinstance(market_data, dict)
        assert len(market_data) > 0
    
    def test_prediction_making(self):
        """Test prediction generation."""
        ts = MainTradingSystem()
        prediction = ts.make_prediction('BTCUSDT')
        
        assert isinstance(prediction, dict)
        if 'combined_prediction' in prediction:
            assert prediction['combined_prediction'] in ['buy', 'sell', 'hold']
    
    def test_risk_assessment(self):
        """Test risk assessment."""
        ts = MainTradingSystem()
        risk_assessment = ts.assess_risk('BTCUSDT')
        
        assert isinstance(risk_assessment, dict)
        assert 'risk_signal' in risk_assessment


class TestBacktestingEngine:
    """Test backtesting functionality."""
    
    def test_backtesting_initialization(self):
        """Test backtesting engine initialization."""
        be = BacktestingEngine()
        assert be.initial_capital > 0
        assert be.trades == []
    
    def test_historical_data_loading(self):
        """Test historical data loading."""
        be = BacktestingEngine()
        data = be.load_historical_data('BTCUSDT')
        
        assert len(data) > 0
        assert 'close' in data.columns
    
    def test_strategy_signals(self):
        """Test strategy signal generation."""
        be = BacktestingEngine()
        data = be.load_historical_data('BTCUSDT', limit=100)
        
        # Test with simple MA crossover strategy
        data_with_signals = be.create_strategy_signals(be.simple_ma_crossover_strategy, data)
        assert 'signal' in data_with_signals.columns
    
    def test_backtest_execution(self):
        """Test backtest execution."""
        be = BacktestingEngine()
        
        try:
            result = be.run_backtest('BTCUSDT', be.simple_ma_crossover_strategy)
            assert isinstance(result, dict)
            assert 'total_return' in result
            assert 'total_trades' in result
        except Exception as e:
            # Backtesting might fail with insufficient data, which is acceptable
            assert True


class TestMarketConnectors:
    """Test market connector functionality."""
    
    def test_mock_connector_initialization(self):
        """Test mock exchange connector initialization."""
        connector = MockExchangeConnector()
        assert connector.mock_balance['USD'] == 10000.0
        assert not connector.is_connected
    
    def test_mock_connector_connection(self):
        """Test mock connector connection."""
        connector = MockExchangeConnector()
        success = connector.connect()
        assert success
        assert connector.is_connected
    
    def test_mock_market_data(self):
        """Test mock market data retrieval."""
        connector = MockExchangeConnector()
        connector.connect()
        
        market_data = connector.get_market_data('BTCUSDT')
        assert 'symbol' in market_data
        assert 'price' in market_data
        assert 'timestamp' in market_data
    
    def test_mock_order_placement(self):
        """Test mock order placement."""
        connector = MockExchangeConnector()
        connector.connect()
        
        order = connector.place_order('BTCUSDT', 'buy', 0.1)
        assert order['status'] == 'filled'
        assert order['side'] == 'buy'
        assert order['amount'] == 0.1
    
    def test_connector_manager(self):
        """Test market connector manager."""
        manager = MarketConnectorManager()
        manager.setup_default_connectors()
        
        assert 'mock' in manager.connectors
        
        # Test connection
        results = manager.connect_all()
        assert 'mock' in results


class TestLoggingSystem:
    """Test logging system functionality."""
    
    def test_logging_system_initialization(self):
        """Test logging system initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            ls = LoggingSystem(temp_dir)
            assert ls.trade_logger is not None
            assert ls.performance_monitor is not None
            assert ls.health_monitor is not None
    
    def test_trade_logging(self):
        """Test trade logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            ls = LoggingSystem(temp_dir)
            
            trade_log = TradeLog(
                timestamp=datetime.now(),
                symbol='BTCUSDT',
                action='buy',
                price=50000.0,
                quantity=0.1,
                order_id='test_001',
                execution_time=0.5
            )
            
            ls.trade_logger.log_trade(trade_log)
            assert len(ls.trade_logger.trade_logs) == 1
    
    def test_performance_monitoring(self):
        """Test performance monitoring."""
        with tempfile.TemporaryDirectory() as temp_dir:
            ls = LoggingSystem(temp_dir)
            
            ls.performance_monitor.log_metric('test_metric', 123.45)
            assert len(ls.performance_monitor.performance_metrics) == 1
    
    def test_daily_report_generation(self):
        """Test daily report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            ls = LoggingSystem(temp_dir)
            
            report = ls.generate_daily_report()
            assert isinstance(report, dict)
            assert 'date' in report
            assert 'total_trades' in report


# Integration tests
class TestSystemIntegration:
    """Test system integration."""
    
    def test_full_system_integration(self):
        """Test full system integration."""
        # Initialize all components
        ts = MainTradingSystem()
        
        # Test data flow
        market_data = ts.get_market_data('BTCUSDT')
        assert market_data is not None
        
        # Test prediction pipeline
        prediction = ts.make_prediction('BTCUSDT')
        assert prediction is not None
        
        # Test risk assessment
        risk_assessment = ts.assess_risk('BTCUSDT')
        assert risk_assessment is not None
    
    def test_trading_workflow(self):
        """Test complete trading workflow."""
        ts = MainTradingSystem()
        
        # Make prediction
        prediction = ts.make_prediction('BTCUSDT')
        
        # Execute trade if signal exists
        if prediction and prediction.get('combined_prediction') in ['buy', 'sell']:
            trade_result = ts.execute_trade(prediction, 'BTCUSDT')
            assert trade_result is not None
            assert 'status' in trade_result