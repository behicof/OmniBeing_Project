#!/usr/bin/env python3
"""
Test script to validate the OmniBeing Trading System
Tests all core components and integration
"""

import sys
import traceback
from datetime import datetime

def test_basic_imports():
    """Test that all basic modules can be imported"""
    print("Testing basic imports...")
    try:
        from config import config
        from gut_trader import IntuitiveDecisionCore
        from external_risk_manager import ExternalRiskManager
        from data_manager import DataManager
        from market_connectors import MarketConnectorManager
        from main_trading_system import MainTradingSystem
        print("✓ All basic imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_configuration():
    """Test configuration system"""
    print("\nTesting configuration system...")
    try:
        from config import config
        
        # Test basic config access
        params = config.get_trading_parameters()
        assert params['instrument'] == 'XAUUSD'
        assert params['initial_capital'] > 0
        
        # Test logging setup
        logger = config.setup_logging('INFO')
        logger.info("Configuration test successful")
        
        print("✓ Configuration system working")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_core_modules():
    """Test individual core modules"""
    print("\nTesting core modules...")
    try:
        # Test IntuitiveDecisionCore
        from gut_trader import IntuitiveDecisionCore
        core = IntuitiveDecisionCore()
        decision = core.decide(0.7, 0.6, 0.5)
        assert decision in ['buy', 'sell', 'wait']
        print("✓ IntuitiveDecisionCore working")
        
        # Test ExternalRiskManager
        from external_risk_manager import ExternalRiskManager
        risk_manager = ExternalRiskManager()
        risk_signal = risk_manager.generate_signal()
        assert 'action' in risk_signal
        print("✓ ExternalRiskManager working")
        
        # Test DataManager
        from data_manager import DataManager
        data_manager = DataManager()
        market_data = data_manager.get_market_data_for_prediction('XAUUSD')
        assert 'price' in market_data
        print("✓ DataManager working")
        
        return True
    except Exception as e:
        print(f"✗ Core modules error: {e}")
        return False

def test_market_connectors():
    """Test market connector system"""
    print("\nTesting market connectors...")
    try:
        from market_connectors import MarketConnectorManager
        
        manager = MarketConnectorManager()
        manager.setup_default_connectors()
        
        # Test mock connector
        connections = manager.connect_all()
        assert 'mock' in connections
        assert connections['mock'] == True
        
        # Test market data retrieval
        market_data = manager.get_market_data('BTCUSDT', 'mock')
        assert 'price' in market_data
        
        print("✓ Market connectors working")
        return True
    except Exception as e:
        print(f"✗ Market connectors error: {e}")
        return False

def test_main_trading_system():
    """Test the main trading system integration"""
    print("\nTesting main trading system...")
    try:
        from main_trading_system import MainTradingSystem
        
        # Initialize system
        ts = MainTradingSystem()
        
        # Connect to markets
        connections = ts.connect_to_markets()
        assert 'mock' in connections
        
        # Test system status
        status = ts.get_system_status()
        assert status['is_trading_enabled'] == True
        assert 'IntuitiveDecisionCore' in status['core_modules']
        
        # Test market data
        market_data = ts.get_market_data('XAUUSD')
        assert 'price' in market_data
        
        # Test prediction
        prediction = ts.make_prediction('XAUUSD')
        assert 'prediction' in prediction
        
        # Test risk assessment
        risk_assessment = ts.assess_risk('XAUUSD')
        assert 'risk_signal' in risk_assessment
        
        print("✓ Main trading system working")
        return True
    except Exception as e:
        print(f"✗ Main trading system error: {e}")
        traceback.print_exc()
        return False

def test_complete_workflow():
    """Test a complete trading workflow"""
    print("\nTesting complete trading workflow...")
    try:
        from main_trading_system import MainTradingSystem
        
        # Initialize and setup
        ts = MainTradingSystem()
        ts.connect_to_markets()
        
        # Complete workflow
        market_data = ts.get_market_data('XAUUSD')
        prediction = ts.make_prediction('XAUUSD')
        risk_assessment = ts.assess_risk('XAUUSD')
        
        # Try to execute trade (should work even if no signal)
        if prediction.get('prediction') in ['buy', 'sell']:
            trade_result = ts.execute_trade(prediction, 'XAUUSD')
            print(f"  Trade executed: {trade_result.get('status', 'unknown')}")
        else:
            print("  No actionable signal (this is normal)")
        
        # Get performance report
        performance = ts.get_performance_report()
        assert isinstance(performance, dict)
        
        print("✓ Complete workflow working")
        return True
    except Exception as e:
        print(f"✗ Complete workflow error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=== OmniBeing Trading System Validation ===")
    print(f"Test started at: {datetime.now()}")
    
    tests = [
        test_basic_imports,
        test_configuration,
        test_core_modules,
        test_market_connectors,
        test_main_trading_system,
        test_complete_workflow
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! Trading system is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())