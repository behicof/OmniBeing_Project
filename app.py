"""
Main Application Entry Point for the OmniBeing Trading System.
Demonstrates the integrated trading system with all components working together.
"""

from main_trading_system import MainTradingSystem
from config import config
import time


def demo_trading_system():
    """Demonstrate the complete trading system functionality."""
    print("=" * 60)
    print("OmniBeing Trading System - Complete Integration Demo")
    print("=" * 60)
    print("Created by behicof")
    print()
    
    # Initialize the main trading system
    print("1. Initializing Trading System...")
    trading_system = MainTradingSystem()
    
    # Get system status
    print("\n2. System Status:")
    status = trading_system.get_system_status()
    print(f"   System Running: {status['is_running']}")
    print(f"   Trading Enabled: {status['is_trading_enabled']}")
    print(f"   Available Prediction Systems: {status['prediction_systems']}")
    print(f"   Account Balance: ${status['account_balance']:,.2f}")
    
    # Demonstrate market data retrieval
    print("\n3. Market Data Retrieval:")
    symbol = config.trading_instrument
    market_data = trading_system.get_market_data(symbol)
    if market_data:
        print(f"   Symbol: {symbol}")
        print(f"   Current Price: ${market_data.get('price', 'N/A'):,.2f}")
        print(f"   Sentiment: {market_data.get('sentiment', 'N/A'):.3f}")
        print(f"   Volatility: {market_data.get('volatility', 'N/A'):.3f}")
    
    # Demonstrate prediction making
    print("\n4. Making Trading Predictions:")
    prediction = trading_system.make_prediction(symbol)
    if prediction and 'combined_prediction' in prediction:
        print(f"   Combined Prediction: {prediction['combined_prediction']}")
        if 'individual_predictions' in prediction:
            print(f"   Individual Predictions: {prediction['individual_predictions']}")
    
    # Demonstrate risk assessment
    print("\n5. Risk Assessment:")
    risk_assessment = trading_system.assess_risk(symbol)
    if risk_assessment and 'risk_signal' in risk_assessment:
        risk_signal = risk_assessment['risk_signal']
        print(f"   Risk Action: {risk_signal.get('action', 'N/A')}")
        print(f"   Risk Score: {risk_signal.get('risk_score', 'N/A'):.3f}")
    
    # Demonstrate trade execution (simulation)
    print("\n6. Trade Execution (Simulation):")
    if prediction and prediction.get('combined_prediction') in ['buy', 'sell']:
        trade_result = trading_system.execute_trade(prediction, symbol)
        print(f"   Trade Status: {trade_result.get('status', 'N/A')}")
        if trade_result.get('status') == 'executed':
            trade_details = trade_result.get('trade_details', {})
            print(f"   Action: {trade_details.get('action', 'N/A')}")
            print(f"   Entry Price: ${trade_details.get('entry_price', 0):.2f}")
            print(f"   Position Size: {trade_details.get('position_size', 0):.4f}")
    else:
        print("   No trade signal generated")
    
    # Show performance report
    print("\n7. Performance Report:")
    performance_report = trading_system.get_performance_report()
    if 'total_trades' in performance_report:
        print(f"   Total Trades: {performance_report['total_trades']}")
        print(f"   Current Balance: ${performance_report['current_balance']:,.2f}")
        print(f"   Active Positions: {performance_report['active_positions']}")
    
    print("\n" + "=" * 60)
    print("Trading System Demo Complete!")
    print("The system successfully integrates:")
    print("- Configuration Management")
    print("- Data Pipeline with Technical Analysis")
    print("- Advanced Risk Management")
    print("- Machine Learning Predictions")
    print("- Portfolio Management")
    print("- Comprehensive Logging")
    print("=" * 60)
    
    print("\n8. Cleanup:")
    trading_system.stop_real_time_trading()
    print("   System shutdown complete")


def main():
    """Main entry point."""
    demo_trading_system()


if __name__ == "__main__":
    main()
