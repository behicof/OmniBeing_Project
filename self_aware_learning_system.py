
class SelfAwareLearningSystem:
    def __init__(self):
        self.market_data = []
        self.market_feedback = []
        self.optimized_strategies = []
        self.performance_log = []

    def process_market_feedback(self, feedback_data):
        # پردازش بازخوردهای بازار
        if feedback_data['result'] == 'profit':
            self.market_feedback.append({'feedback': 'positive', 'strategy': feedback_data['strategy']})
        else:
            self.market_feedback.append({'feedback': 'negative', 'strategy': feedback_data['strategy']})

    def adapt_to_feedback(self):
        # سازگاری با بازخوردهای بازار و بهینه‌سازی استراتژی‌ها
        positive_feedback = [f for f in self.market_feedback if f['feedback'] == 'positive']
        negative_feedback = [f for f in self.market_feedback if f['feedback'] == 'negative']
        
        if len(positive_feedback) > len(negative_feedback):
            self.optimized_strategies.append("Enhanced strategy based on positive feedback")
        elif len(negative_feedback) > len(positive_feedback):
            self.optimized_strategies.append("Adjusted strategy for better risk management")
        else:
            self.optimized_strategies.append("Stable strategy maintained")

    def evaluate_strategy_performance(self):
        # ارزیابی عملکرد استراتژی‌ها
        for strategy in self.optimized_strategies:
            if strategy == "Enhanced strategy based on positive feedback":
                self.performance_log.append("Optimized for higher success rate.")
            elif strategy == "Adjusted strategy for better risk management":
                self.performance_log.append("Risk-managed strategy applied.")
            else:
                self.performance_log.append("Standard strategy applied.")

    def get_performance_log(self):
        return self.performance_log

    def get_optimized_strategies(self):
        return self.optimized_strategies
