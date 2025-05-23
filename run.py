from final_real_time_optimization_predictive_system import FinalRealTimeOptimizationPredictiveSystem
from market_integration_system import MarketIntegrationSystem
from final_market_optimization_system import FinalMarketOptimizationSystem

def main():
    # Create instances of the predictive systems
    """
    Demonstrates the usage of predictive and integration systems with sample market data.
    
    Creates instances of real-time optimization, market integration, and market optimization systems. Processes sample market data through each system, trains predictive models, generates predictions, and prints the results.
    """
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

    # Get and print all predictions made so far by the real-time system
    all_predictions_real_time = real_time_system.get_predictions()
    print("All predictions made by the real-time system:", all_predictions_real_time)

if __name__ == "__main__":
    main()
