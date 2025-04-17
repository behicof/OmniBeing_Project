
class ExpandedEvolutionaryLearningSystem:
    def __init__(self):
        self.performance_log = []
        self.strategy_versions = []
        self.market_feedback = []
        self.optimization_history = []
        self.real_time_analysis = []

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
        # تحلیل داده‌های لحظه‌ای و تطبیق استراتژی‌ها بر اساس تغییرات سریع
        sentiment = market_data.get('sentiment', 0)
        volatility = market_data.get('volatility', 0)
        
        if sentiment > 0.7 and volatility < 0.3:
            return "Aggressive buy strategy"
        elif sentiment < 0.3 and volatility > 0.7:
            return "Risk-off strategy"
        else:
            return "Moderate strategy"
    
    def optimize_strategy(self):
        success_rate = sum([log['performance_score'] for log in self.performance_log]) / len(self.performance_log)
        if success_rate > 0.8:
            self.strategy_versions.append("Optimized for higher success rate with dynamic adjustments.")
            self.optimization_history.append(f"Version {len(self.strategy_versions)} optimized with dynamic adjustments.")
        elif success_rate < 0.5:
            self.strategy_versions.append("Recalibrated strategy for risk mitigation.")
            self.optimization_history.append(f"Version {len(self.strategy_versions)} recalibrated.")
    
    def get_optimization_history(self):
        return self.optimization_history

    def get_performance_log(self):
        return self.performance_log
    
    def get_real_time_analysis(self, market_data):
        analysis = self.real_time_data_analysis(market_data)
        self.real_time_analysis.append(analysis)
        return analysis
