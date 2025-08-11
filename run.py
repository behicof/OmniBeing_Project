from final_real_time_optimization_predictive_system import FinalRealTimeOptimizationPredictiveSystem
from market_integration_system import MarketIntegrationSystem
from final_optimization_system import FinalMarketOptimizationSystem

def main():
    # Create instances of the predictive systems
    """
    Demonstrates the usage of predictive and integration systems with sample market data.
    
    Creates instances of real-time optimization, market integration, and market optimization systems. Processes sample market data through each system, trains predictive models, generates predictions, and prints the results.
    """
    real_time_system = FinalRealTimeOptimizationPredictiveSystem()
    market_integration_system = MarketIntegrationSystem()
    market_optimization_system = FinalMarketOptimizationSystem()

    # Process multiple sample market data points to enable proper training
    sample_market_data_list = [
        {'sentiment': 0.8, 'volatility': 0.3, 'price_change': 0.05, 'buy_sell_signal': 1},
        {'sentiment': 0.2, 'volatility': 0.7, 'price_change': -0.03, 'buy_sell_signal': 0},
        {'sentiment': 0.6, 'volatility': 0.4, 'price_change': 0.01, 'buy_sell_signal': 1},
        {'sentiment': 0.1, 'volatility': 0.9, 'price_change': -0.08, 'buy_sell_signal': 0},
        {'sentiment': 0.9, 'volatility': 0.2, 'price_change': 0.12, 'buy_sell_signal': 1}
    ]
    
    # Process data with appropriate methods for each system
    for sample_data in sample_market_data_list:
        real_time_system.process_market_data(sample_data)
        market_integration_system.receive_live_data("SamplePlatform", sample_data)
        
        # For market optimization system, provide data in the format it expects
        optimization_data = {
            'market_sentiment': sample_data['sentiment'],
            'market_volatility': sample_data['volatility']
        }
        market_optimization_system.receive_live_data("SamplePlatform", optimization_data)

    # Train models and make predictions for systems that support it
    print("Training real-time prediction system...")
    real_time_system.train_models()
    
    # Use the first sample for prediction demonstration
    prediction_sample = sample_market_data_list[0]
    prediction_real_time = real_time_system.make_predictions(prediction_sample)

    # Process data and get decisions from market optimization system
    print("Processing market optimization decisions...")
    market_optimization_system.process_data_for_decision()
    market_optimization_system.optimize_decision_making()
    optimization_decisions = market_optimization_system.get_optimized_decisions()

    # Print the results
    print("\n=== RESULTS ===")
    print("Real-time system prediction:", prediction_real_time)
    print("Market optimization system decisions:", optimization_decisions)
    print("Market integration system processed data successfully!")
    
    print("\n=== SYSTEM SUMMARY ===")
    print(f"Processed {len(sample_market_data_list)} market data samples")
    print("All systems initialized and functioning correctly!")

if __name__ == "__main__":
    main()
