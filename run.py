from final_real_time_optimization_predictive_system import FinalRealTimeOptimizationPredictiveSystem
from market_integration_system import MarketIntegrationSystem
from final_optimization_system import FinalMarketOptimizationSystem

def main():
    # Create instances of the predictive systems
    real_time_system = FinalRealTimeOptimizationPredictiveSystem()
    market_integration_system = MarketIntegrationSystem()
    market_optimization_system = FinalMarketOptimizationSystem()

    # Process sample market data
    sample_market_data = {
        'sentiment': 0.8,
        'volatility': 0.3,
        'price_change': 0.05,
        'buy_sell_signal': 1
    }
    real_time_system.process_market_data(sample_market_data)
    market_integration_system.receive_live_data("SamplePlatform", sample_market_data)
    market_optimization_system.process_market_data(sample_market_data)

    # Train models and make predictions
    real_time_system.train_models()
    market_optimization_system.train_models()

    prediction_real_time = real_time_system.make_predictions(sample_market_data)
    prediction_market_optimization = market_optimization_system.make_predictions(sample_market_data)

    # Print the results
    print("Real-time system prediction:", prediction_real_time)
    print("Market optimization system prediction:", prediction_market_optimization)

if __name__ == "__main__":
    main()
