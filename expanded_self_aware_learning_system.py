
class ExpandedSelfAwareLearningSystem:
    def __init__(self):
        self.market_data = []
        self.market_feedback = []
        self.optimized_strategies = []
        self.performance_log = []
        self.adaptive_feedback = []
        self.strategy_evolution = []

    def process_market_feedback(self, feedback_data):
        # پردازش بازخوردهای بازار برای استراتژی‌های بهینه
        if feedback_data['result'] == 'profit':
            self.market_feedback.append({'feedback': 'positive', 'strategy': feedback_data['strategy']})
        else:
            self.market_feedback.append({'feedback': 'negative', 'strategy': feedback_data['strategy']})

    def adapt_to_feedback(self):
        # سازگاری با بازخوردهای بازار برای بهینه‌سازی استراتژی‌ها
        positive_feedback = [f for f in self.market_feedback if f['feedback'] == 'positive']
        negative_feedback = [f for f in self.market_feedback if f['feedback'] == 'negative']
        
        if len(positive_feedback) > len(negative_feedback):
            self.optimized_strategies.append("Evolved strategy with higher success rate")
            self.strategy_evolution.append("Strategy evolved based on positive feedback.")
        elif len(negative_feedback) > len(positive_feedback):
            self.optimized_strategies.append("Refined strategy for improved risk management")
            self.strategy_evolution.append("Strategy refined to reduce risk.")
        else:
            self.optimized_strategies.append("Stable strategy maintained")
            self.strategy_evolution.append("Maintaining stability in strategy.")

    def evaluate_strategy_performance(self):
        # ارزیابی عملکرد استراتژی‌ها برای تعیین اثربخشی
        for strategy in self.optimized_strategies:
            if strategy == "Evolved strategy with higher success rate":
                self.performance_log.append("Optimized for higher success rate.")
            elif strategy == "Refined strategy for improved risk management":
                self.performance_log.append("Refined strategy for risk reduction.")
            else:
                self.performance_log.append("Standard strategy maintained.")

    def get_performance_log(self):
        return self.performance_log

    def get_optimized_strategies(self):
        return self.optimized_strategies
    
    def get_strategy_evolution(self):
        return self.strategy_evolution
