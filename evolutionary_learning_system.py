
class EvolutionaryLearningSystem:
    def __init__(self):
        self.performance_log = []
        self.strategy_versions = []
        self.market_feedback = []
        self.optimization_history = []

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
    
    def optimize_strategy(self):
        success_rate = sum([log['performance_score'] for log in self.performance_log]) / len(self.performance_log)
        if success_rate > 0.7:
            self.strategy_versions.append("Optimized for higher success rate")
            self.optimization_history.append(f"Version {len(self.strategy_versions)} optimized based on performance.")
        elif success_rate < 0.5:
            self.strategy_versions.append("Adjusted for better risk management")
            self.optimization_history.append(f"Version {len(self.strategy_versions)} adjusted for better risk management.")
    
    def get_optimization_history(self):
        return self.optimization_history

    def get_performance_log(self):
        return self.performance_log
