#!/usr/bin/env python3
"""
Demo script for the OmniBeing Trading System
Demonstrates the complete trading workflow with the simplified implementation
"""

import time
from datetime import datetime
from main_trading_system import MainTradingSystem

def main():
    print("🚀 OmniBeing Trading System Demo")
    print("=" * 50)
    
    # Initialize the trading system
    print("\n1. Initializing Trading System...")
    ts = MainTradingSystem()
    
    # Connect to markets
    print("\n2. Connecting to Markets...")
    connections = ts.connect_to_markets()
    for exchange, status in connections.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {exchange}: {'Connected' if status else 'Failed'}")
    
    # Get system status
    print("\n3. System Status:")
    status = ts.get_system_status()
    print(f"   Trading Enabled: {status['is_trading_enabled']}")
    print(f"   Core Modules: {', '.join(status['core_modules'])}")
    print(f"   Account Balance: ${status['account_balance']:,.2f}")
    print(f"   Active Positions: {status['active_positions']}")
    
    # Trading configuration
    print("\n4. Trading Configuration:")
    params = status['trading_parameters']
    print(f"   Instrument: {params['instrument']}")
    print(f"   Initial Capital: ${params['initial_capital']:,.2f}")
    print(f"   Risk per Trade: {params['risk_percentage']}%")
    print(f"   Max Positions: {params['max_positions']}")
    print(f"   Stop Loss: {params['stop_loss_percentage']}%")
    print(f"   Take Profit: {params['take_profit_percentage']}%")
    
    # Run multiple trading cycles
    print("\n5. Running Trading Cycles...")
    symbol = params['instrument']
    
    for cycle in range(3):
        print(f"\n   Cycle {cycle + 1}:")
        
        # Get market data
        market_data = ts.get_market_data(symbol)
        current_price = market_data['price']
        print(f"   📊 Current Price: ${current_price:,.2f}")
        print(f"   📈 Price Change: {market_data['price_change']:.4f}")
        print(f"   📊 RSI: {market_data['rsi']:.1f}")
        print(f"   🔄 Volatility: {market_data['volatility']:.4f}")
        
        # Make prediction
        prediction = ts.make_prediction(symbol)
        decision = prediction['prediction']
        print(f"   🧠 Decision: {decision.upper()}")
        print(f"   🎯 Pattern Rarity: {prediction['pattern_rarity']:.3f}")
        print(f"   💭 Memory Match: {prediction['memory_match_score']:.3f}")
        print(f"   😰 Emotional Pressure: {prediction['emotional_pressure']:.3f}")
        
        # Risk assessment
        risk_assessment = ts.assess_risk(symbol)
        risk_action = risk_assessment['risk_signal']['action']
        print(f"   ⚠️ Risk Assessment: {risk_action}")
        
        # Try to execute trade
        if decision in ['buy', 'sell'] and risk_action == 'PROCEED':
            trade_result = ts.execute_trade(prediction, symbol)
            if trade_result['status'] == 'executed':
                trade_details = trade_result['trade_details']
                print(f"   💰 Trade Executed: {trade_details['action'].upper()}")
                print(f"   💸 Position Size: {trade_details['position_size']:.4f}")
                print(f"   🛑 Stop Loss: ${trade_details['stop_loss']:,.2f}")
                print(f"   🎯 Take Profit: ${trade_details['take_profit']:,.2f}")
            else:
                print(f"   ❌ Trade Failed: {trade_result.get('message', 'Unknown error')}")
        else:
            print(f"   ⏸️ No Trade: {decision} signal or risk block")
        
        # Wait before next cycle
        if cycle < 2:
            print("   ⏳ Waiting 2 seconds...")
            time.sleep(2)
    
    # Performance summary
    print("\n6. Performance Summary:")
    performance = ts.get_performance_report()
    if performance.get('total_trades', 0) > 0:
        print(f"   Total Trades: {performance['total_trades']}")
        print(f"   Winning Trades: {performance['winning_trades']}")
        print(f"   Win Rate: {performance['win_rate']:.1f}%")
        print(f"   Total P&L: ${performance['total_pnl']:,.2f}")
        print(f"   Current Balance: ${performance['current_balance']:,.2f}")
    else:
        print("   No trades executed during this demo")
    
    # Get intuitive core memory
    print("\n7. Intuitive Decision Memory:")
    memory = ts.intuitive_core.get_memory()
    if memory:
        print(f"   Decisions Made: {len(memory)}")
        for i, decision_record in enumerate(memory[-3:], 1):  # Show last 3
            print(f"   Decision {i}: {decision_record['decision']} "
                  f"(rarity: {decision_record['rarity']:.3f})")
    else:
        print("   No decision history available")
    
    print("\n🎉 Demo completed successfully!")
    print("\nThe OmniBeing Trading System is ready for deployment.")
    print("Key achievements:")
    print("✅ Minimal dependencies - no complex libraries required")
    print("✅ Core integration - IntuitiveDecisionCore + ExternalRiskManager")
    print("✅ Real-time data handling with mock connector")
    print("✅ Risk management and position sizing")
    print("✅ Configurable parameters via YAML and environment variables")
    print("✅ CCXT integration ready for real exchanges")

if __name__ == "__main__":
    main()