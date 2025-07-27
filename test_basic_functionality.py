#!/usr/bin/env python3
"""
Simple test script to verify basic trading system functionality.
Tests core components without complex dependencies.
"""

import sys
import traceback
from datetime import datetime

def test_core_imports():
    """Test that all core modules can be imported."""
    print("=== Testing Core Imports ===")
    
    try:
        from gut_trader import IntuitiveDecisionCore
        print("✓ IntuitiveDecisionCore imported successfully")
    except Exception as e:
        print(f"✗ IntuitiveDecisionCore import failed: {e}")
        return False
    
    try:
        from external_risk_manager import ExternalRiskManager
        print("✓ ExternalRiskManager imported successfully")
    except Exception as e:
        print(f"✗ ExternalRiskManager import failed: {e}")
        return False
    
    try:
        from config import config
        print("✓ Config imported successfully")
    except Exception as e:
        print(f"✗ Config import failed: {e}")
        return False
    
    try:
        from data_manager import DataManager
        print("✓ DataManager imported successfully")
    except Exception as e:
        print(f"✗ DataManager import failed: {e}")
        return False
    
    try:
        from market_connectors import MockExchangeConnector
        print("✓ MockExchangeConnector imported successfully")
    except Exception as e:
        print(f"✗ MockExchangeConnector import failed: {e}")
        return False
    
    try:
        from main_trading_system import MainTradingSystem
        print("✓ MainTradingSystem imported successfully")
    except Exception as e:
        print(f"✗ MainTradingSystem import failed: {e}")
        return False
    
    return True

def test_intuitive_decision_core():
    """Test IntuitiveDecisionCore functionality."""
    print("\n=== Testing IntuitiveDecisionCore ===")
    
    try:
        from gut_trader import IntuitiveDecisionCore
        
        # Test initialization
        core = IntuitiveDecisionCore()
        print("✓ IntuitiveDecisionCore initialized")
        
        # Test decision making
        decision = core.decide(0.8, 0.7, 0.6)  # High values should trigger buy
        print(f"✓ Decision made: {decision}")
        
        # Test memory functionality
        memory = core.get_memory()
        print(f"✓ Memory retrieved: {len(memory)} entries")
        
        # Test multiple decisions
        for i in range(3):
            decision = core.decide(0.5 + i*0.1, 0.4 + i*0.1, 0.3 + i*0.1)
            print(f"  Decision {i+1}: {decision}")
        
        memory = core.get_memory()
        print(f"✓ Memory after multiple decisions: {len(memory)} entries")
        
        return True
        
    except Exception as e:
        print(f"✗ IntuitiveDecisionCore test failed: {e}")
        traceback.print_exc()
        return False

def test_risk_manager():
    """Test ExternalRiskManager functionality."""
    print("\n=== Testing ExternalRiskManager ===")
    
    try:
        from external_risk_manager import ExternalRiskManager
        
        # Test initialization
        risk_manager = ExternalRiskManager()
        print("✓ ExternalRiskManager initialized")
        
        # Test volatility update
        prices = [100, 102, 98, 105, 95, 103, 99]
        risk_manager.update_volatility(prices)
        print(f"✓ Volatility updated: {risk_manager.current_volatility:.4f}")
        
        # Test risk assessment
        risk_score = risk_manager.assess_risk()
        print(f"✓ Risk assessed: {risk_score:.4f}")
        
        # Test position size calculation
        position_size = risk_manager.calculate_position_size('BTCUSDT', 50000, 48000)
        print(f"✓ Position size calculated: {position_size:.6f}")
        
        # Test signal generation
        signal = risk_manager.generate_signal()
        print(f"✓ Signal generated: {signal['action']}")
        
        # Test risk report
        report = risk_manager.get_risk_report()
        print(f"✓ Risk report generated with {len(report)} metrics")
        
        return True
        
    except Exception as e:
        print(f"✗ ExternalRiskManager test failed: {e}")
        traceback.print_exc()
        return False

def test_data_manager():
    """Test DataManager functionality."""
    print("\n=== Testing DataManager ===")
    
    try:
        from data_manager import DataManager
        
        # Test initialization
        dm = DataManager()
        print("✓ DataManager initialized")
        
        # Test historical data fetching
        data = dm.fetch_historical_data('BTCUSDT', limit=50)
        print(f"✓ Historical data fetched: {len(data)} records")
        print(f"  Columns: {list(data.columns)}")
        
        # Test technical indicators
        data_with_indicators = dm.calculate_technical_indicators(data)
        print(f"✓ Technical indicators calculated: {len(data_with_indicators.columns)} columns")
        
        # Test feature engineering
        features = dm.engineer_features('BTCUSDT')
        print(f"✓ Features engineered: {len(features)} features")
        print(f"  Sample features: {list(features.keys())[:5]}")
        
        # Test live data
        live_data = dm.get_live_data('BTCUSDT')
        print(f"✓ Live data retrieved: price=${live_data['price']:.2f}")
        
        # Test market data for prediction
        market_data = dm.get_market_data_for_prediction('BTCUSDT')
        print(f"✓ Market data for prediction: {len(market_data)} fields")
        
        return True
        
    except Exception as e:
        print(f"✗ DataManager test failed: {e}")
        traceback.print_exc()
        return False

def test_market_connector():
    """Test MockExchangeConnector functionality."""
    print("\n=== Testing MockExchangeConnector ===")
    
    try:
        from market_connectors import MockExchangeConnector
        
        # Test initialization
        connector = MockExchangeConnector()
        print("✓ MockExchangeConnector initialized")
        
        # Test connection
        success = connector.connect()
        print(f"✓ Connection established: {success}")
        
        # Test market data
        market_data = connector.get_market_data('BTCUSDT')
        print(f"✓ Market data retrieved: price=${market_data['price']:.2f}")
        
        # Test account balance
        balance = connector.get_account_balance()
        print(f"✓ Account balance: {balance}")
        
        # Test order placement
        order = connector.place_order('BTCUSDT', 'buy', 0.1)
        print(f"✓ Order placed: {order['status']} - {order['id']}")
        
        # Test balance after order
        new_balance = connector.get_account_balance()
        print(f"✓ Balance after order: {new_balance}")
        
        connector.disconnect()
        print("✓ Disconnected successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ MockExchangeConnector test failed: {e}")
        traceback.print_exc()
        return False

def test_main_trading_system():
    """Test MainTradingSystem functionality."""
    print("\n=== Testing MainTradingSystem ===")
    
    try:
        from main_trading_system import MainTradingSystem
        
        # Test initialization
        ts = MainTradingSystem()
        print("✓ MainTradingSystem initialized")
        
        # Test system status
        status = ts.get_system_status()
        print(f"✓ System status retrieved: running={status['is_running']}, trading={status['is_trading_enabled']}")
        
        # Test market data retrieval
        market_data = ts.get_market_data('BTCUSDT')
        print(f"✓ Market data retrieved: {len(market_data)} fields")
        
        # Test prediction
        prediction = ts.make_prediction('BTCUSDT')
        print(f"✓ Prediction made: {prediction.get('combined_prediction', 'N/A')}")
        
        # Test risk assessment
        risk_assessment = ts.assess_risk('BTCUSDT')
        print(f"✓ Risk assessment: {risk_assessment['risk_signal']['action']}")
        
        # Test trade execution (if prediction exists)
        if prediction and 'combined_prediction' in prediction:
            if prediction['combined_prediction'] in ['buy', 'sell']:
                trade_result = ts.execute_trade(prediction, 'BTCUSDT')
                print(f"✓ Trade executed: {trade_result['status']}")
            else:
                print("✓ No trade signal, trade execution skipped")
        
        # Test performance report
        performance = ts.get_performance_report()
        print(f"✓ Performance report: {performance}")
        
        return True
        
    except Exception as e:
        print(f"✗ MainTradingSystem test failed: {e}")
        traceback.print_exc()
        return False

def test_integration():
    """Test integration of components working together."""
    print("\n=== Testing Integration ===")
    
    try:
        from main_trading_system import MainTradingSystem
        from gut_trader import IntuitiveDecisionCore
        from external_risk_manager import ExternalRiskManager
        
        # Initialize components
        ts = MainTradingSystem()
        intuitive_core = IntuitiveDecisionCore()
        
        print("✓ All components initialized")
        
        # Test data flow
        market_data = ts.get_market_data('BTCUSDT')
        
        # Make intuitive decision
        sentiment = market_data.get('sentiment', 0.5)
        volatility = market_data.get('volatility', 0.5)
        price_change = market_data.get('price_change', 0)
        
        intuitive_decision = intuitive_core.decide(
            pattern_rarity=abs(price_change) * 10,  # Convert to 0-1 scale
            memory_match_score=sentiment + 0.5,
            emotional_pressure=volatility
        )
        
        print(f"✓ Intuitive decision: {intuitive_decision}")
        
        # Make system prediction
        prediction = ts.make_prediction('BTCUSDT')
        print(f"✓ System prediction: {prediction.get('combined_prediction', 'hold')}")
        
        # Assess risk
        risk_assessment = ts.assess_risk('BTCUSDT')
        print(f"✓ Risk assessment: {risk_assessment['risk_signal']['action']}")
        
        # Check if decisions align or differ
        system_decision = prediction.get('combined_prediction', 'hold')
        if intuitive_decision == system_decision:
            print("✓ Intuitive and system decisions align")
        else:
            print(f"⚠ Decisions differ: intuitive={intuitive_decision}, system={system_decision}")
        
        print("✓ Integration test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 OmniBeing Trading System - Basic Functionality Test")
    print("=" * 60)
    
    tests = [
        test_core_imports,
        test_intuitive_decision_core,
        test_risk_manager,
        test_data_manager,
        test_market_connector,
        test_main_trading_system,
        test_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✅ {test.__name__} PASSED")
            else:
                failed += 1
                print(f"❌ {test.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} FAILED with exception: {e}")
        
        print("-" * 40)
    
    print(f"\n📊 Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! The trading system is working correctly.")
        return 0
    else:
        print(f"\n⚠️ {failed} tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())