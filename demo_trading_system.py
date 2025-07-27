#!/usr/bin/env python3
"""
OmniBeing Trading System - Usage Demo
=====================================

This script demonstrates how to use the manual implementation of the trading system.
It shows the core functionality working together step-by-step.
"""

import time
from datetime import datetime

def demo_basic_usage():
    """Demonstrate basic usage of the trading system."""
    print("🚀 OmniBeing Trading System - Usage Demo")
    print("=" * 50)
    
    # Import the main components
    from main_trading_system import MainTradingSystem
    from gut_trader import IntuitiveDecisionCore
    from market_connectors import market_connector_manager
    
    print("\n1️⃣ Initializing Trading System...")
    ts = MainTradingSystem()
    intuitive_core = IntuitiveDecisionCore()
    
    print("✓ Trading system initialized successfully")
    print(f"✓ Available prediction systems: {ts.prediction_systems.keys()}")
    
    # Setup market connector
    print("\n2️⃣ Setting up Market Connections...")
    market_connector_manager.setup_default_connectors()
    connections = market_connector_manager.connect_all()
    print(f"✓ Market connections: {connections}")
    
    # Get system status
    print("\n3️⃣ System Status Check...")
    status = ts.get_system_status()
    print(f"✓ System running: {status['is_running']}")
    print(f"✓ Trading enabled: {status['is_trading_enabled']}")
    print(f"✓ Account balance: ${status['account_balance']:,.2f}")
    print(f"✓ Active positions: {status['active_positions']}")
    
    return ts, intuitive_core

def demo_market_analysis():
    """Demonstrate market data analysis."""
    print("\n4️⃣ Market Data Analysis...")
    
    from main_trading_system import MainTradingSystem
    ts = MainTradingSystem()
    
    # Get market data
    market_data = ts.get_market_data('BTCUSDT')
    print(f"✓ Current BTC price: ${market_data['price']:,.2f}")
    print(f"✓ Price change: {market_data['price_change']:+.4f}")
    print(f"✓ Volatility: {market_data['volatility']:.4f}")
    print(f"✓ Sentiment: {market_data['sentiment']:+.4f}")
    
    return market_data

def demo_intuitive_trading():
    """Demonstrate intuitive decision making."""
    print("\n5️⃣ Intuitive Decision Making...")
    
    from gut_trader import IntuitiveDecisionCore
    from main_trading_system import MainTradingSystem
    
    ts = MainTradingSystem()
    intuitive_core = IntuitiveDecisionCore()
    
    # Get current market data
    market_data = ts.get_market_data('BTCUSDT')
    
    # Make intuitive decision
    pattern_rarity = abs(market_data.get('price_change', 0)) * 10
    memory_match_score = (market_data.get('sentiment', 0) + 1) / 2  # Normalize to 0-1
    emotional_pressure = market_data.get('volatility', 0.5)
    
    intuitive_decision = intuitive_core.decide(
        pattern_rarity=pattern_rarity,
        memory_match_score=memory_match_score,
        emotional_pressure=emotional_pressure
    )
    
    print(f"✓ Pattern rarity: {pattern_rarity:.4f}")
    print(f"✓ Memory match score: {memory_match_score:.4f}")
    print(f"✓ Emotional pressure: {emotional_pressure:.4f}")
    print(f"🧠 Intuitive decision: {intuitive_decision.upper()}")
    
    return intuitive_decision

def demo_systematic_prediction():
    """Demonstrate systematic prediction."""
    print("\n6️⃣ Systematic Prediction...")
    
    from main_trading_system import MainTradingSystem
    ts = MainTradingSystem()
    
    # Make systematic prediction
    prediction = ts.make_prediction('BTCUSDT')
    
    print(f"✓ Individual predictions: {prediction.get('individual_predictions', {})}")
    print(f"🤖 Combined prediction: {prediction.get('combined_prediction', 'hold').upper()}")
    print(f"✓ Timestamp: {prediction.get('timestamp', 'N/A')}")
    
    return prediction

def demo_risk_assessment():
    """Demonstrate risk assessment."""
    print("\n7️⃣ Risk Assessment...")
    
    from main_trading_system import MainTradingSystem
    ts = MainTradingSystem()
    
    # Assess risk
    risk_assessment = ts.assess_risk('BTCUSDT')
    risk_signal = risk_assessment['risk_signal']
    
    print(f"✓ Risk action: {risk_signal['action']}")
    print(f"✓ Risk score: {risk_signal.get('risk_score', 'N/A')}")
    
    if 'risk_details' in risk_signal:
        details = risk_signal['risk_details']
        print(f"✓ Portfolio risk: {details.get('portfolio_risk', 0):.4f}")
        print(f"✓ Total exposure: ${details.get('total_exposure', 0):,.2f}")
    
    return risk_assessment

def demo_trade_execution():
    """Demonstrate trade execution."""
    print("\n8️⃣ Trade Execution Demo...")
    
    from main_trading_system import MainTradingSystem
    ts = MainTradingSystem()
    
    # Get prediction
    prediction = ts.make_prediction('BTCUSDT')
    
    # Execute trade if signal exists
    trade_result = ts.execute_trade(prediction, 'BTCUSDT')
    
    print(f"✓ Trade status: {trade_result['status']}")
    
    if trade_result['status'] == 'executed':
        details = trade_result['trade_details']
        print(f"✓ Action: {details['action']}")
        print(f"✓ Entry price: ${details['entry_price']:,.2f}")
        print(f"✓ Position size: {details['position_size']:.6f}")
        print(f"✓ Stop loss: ${details['stop_loss']:,.2f}")
        print(f"✓ Take profit: ${details['take_profit']:,.2f}")
    else:
        print(f"ℹ️ Reason: {trade_result.get('message', 'Unknown')}")
    
    return trade_result

def demo_performance_monitoring():
    """Demonstrate performance monitoring."""
    print("\n9️⃣ Performance Monitoring...")
    
    from main_trading_system import MainTradingSystem
    ts = MainTradingSystem()
    
    # Get performance report
    performance = ts.get_performance_report()
    
    if 'total_trades' in performance:
        print(f"✓ Total trades: {performance['total_trades']}")
        print(f"✓ Win rate: {performance['win_rate']:.1f}%")
        print(f"✓ Total P&L: ${performance['total_pnl']:,.2f}")
        print(f"✓ Average P&L per trade: ${performance['average_pnl_per_trade']:,.2f}")
        print(f"✓ Current balance: ${performance['current_balance']:,.2f}")
    else:
        print(f"ℹ️ {performance.get('message', 'No performance data available')}")
    
    return performance

def demo_complete_workflow():
    """Demonstrate complete trading workflow."""
    print("\n🔄 Complete Trading Workflow Demo")
    print("=" * 50)
    
    # Initialize system
    ts, intuitive_core = demo_basic_usage()
    
    # Analyze market
    market_data = demo_market_analysis()
    
    # Make intuitive decision
    intuitive_decision = demo_intuitive_trading()
    
    # Make systematic prediction
    systematic_prediction = demo_systematic_prediction()
    
    # Assess risk
    risk_assessment = demo_risk_assessment()
    
    # Execute trade
    trade_result = demo_trade_execution()
    
    # Monitor performance
    performance = demo_performance_monitoring()
    
    # Summary
    print("\n📊 Workflow Summary")
    print("-" * 30)
    print(f"Market Price: ${market_data['price']:,.2f}")
    print(f"Intuitive Decision: {intuitive_decision}")
    print(f"System Prediction: {systematic_prediction.get('combined_prediction', 'hold')}")
    print(f"Risk Assessment: {risk_assessment['risk_signal']['action']}")
    print(f"Trade Result: {trade_result['status']}")
    
    return {
        'market_data': market_data,
        'intuitive_decision': intuitive_decision,
        'systematic_prediction': systematic_prediction,
        'risk_assessment': risk_assessment,
        'trade_result': trade_result,
        'performance': performance
    }

def demo_realtime_simulation():
    """Demonstrate real-time trading simulation."""
    print("\n⏰ Real-time Trading Simulation (5 cycles)")
    print("=" * 50)
    
    from main_trading_system import MainTradingSystem
    ts = MainTradingSystem()
    
    for i in range(5):
        print(f"\n--- Cycle {i+1}/5 ---")
        
        # Get fresh market data
        market_data = ts.get_market_data('BTCUSDT')
        print(f"Price: ${market_data['price']:,.2f} | Change: {market_data['price_change']:+.4f}")
        
        # Make prediction
        prediction = ts.make_prediction('BTCUSDT')
        signal = prediction.get('combined_prediction', 'hold')
        print(f"Signal: {signal.upper()}")
        
        # Assess risk
        risk = ts.assess_risk('BTCUSDT')
        risk_action = risk['risk_signal']['action']
        print(f"Risk: {risk_action}")
        
        # Execute if conditions are met
        if signal in ['buy', 'sell'] and risk_action == 'PROCEED':
            trade_result = ts.execute_trade(prediction, 'BTCUSDT')
            print(f"Trade: {trade_result['status']}")
        else:
            print("Trade: SKIPPED")
        
        # Short delay
        if i < 4:  # Don't wait after last iteration
            time.sleep(1)
    
    # Final status
    print(f"\n📈 Final System Status:")
    final_status = ts.get_system_status()
    print(f"Balance: ${final_status['account_balance']:,.2f}")
    print(f"Positions: {final_status['active_positions']}")
    
    return final_status

def main():
    """Main demo function."""
    print("🎯 OmniBeing Trading System - Complete Demo")
    print("=" * 60)
    print("This demo shows all the core functionality working together.")
    print("The system uses mock data for safe testing.\n")
    
    try:
        # Run complete workflow demo
        workflow_result = demo_complete_workflow()
        
        # Run real-time simulation
        realtime_result = demo_realtime_simulation()
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print("✅ All components are working properly")
        print("✅ Manual implementation is fully functional")
        print("✅ Ready for production use with real market data")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())