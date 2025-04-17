
class FinalExpansionLearningSystem:
    def __init__(self):
        self.performance_log = []
        self.strategy_versions = []
        self.market_feedback = []
        self.optimization_history = []
        self.real_time_analysis = []
        self.market_adaptability = []

    def evaluate_performance(self, prediction, actual_outcome):
        performance_score = 0
        if prediction == actual_outcome:
            performance_score = 1
        else:
            performance_score = -1
        
        self.performance_log.append({
            "prediction": prediction,
            "actual_outcome": actual_outcome,
            "performance_score": performance_score
        })
        return performance_score

    def real_time_data_analysis(self, market_data):
        # تحلیل داده‌های لحظه‌ای با دقت بیشتر و واکنش سریع‌تر
        sentiment = market_data.get('sentiment', 0)
        volatility = market_data.get('volatility', 0)
        
        if sentiment > 0.75 and volatility < 0.25:
            return "Aggressive buy strategy with high confidence"
        elif sentiment < 0.25 and volatility > 0.75:
            return "Risk-off strategy with caution"
        else:
            return "Balanced strategy with adaptive adjustments"
    
    def optimize_strategy(self):
        success_rate = sum([log['performance_score'] for log in self.performance_log]) / len(self.performance_log)
        if success_rate > 0.85:
            self.strategy_versions.append("Highly optimized with predictive adjustments.")
            self.optimization_history.append(f"Version {len(self.strategy_versions)} highly optimized with predictive adjustments.")
        elif success_rate < 0.4:
            self.strategy_versions.append("Complete overhaul for higher risk mitigation.")
            self.optimization_history.append(f"Version {len(self.strategy_versions)} overhauled for better risk management.")
    
    def get_optimization_history(self):
        return self.optimization_history

    def get_performance_log(self):
        return self.performance_log
    
    def get_real_time_analysis(self, market_data):
        analysis = self.real_time_data_analysis(market_data)
        self.real_time_analysis.append(analysis)
        return analysis

    def adapt_to_market_changes(self, market_data):
        adaptability_score = (market_data.get('trend_strength', 0) + market_data.get('news_impact', 0)) / 2
        self.market_adaptability.append(adaptability_score)
        return adaptability_score
