
class SelfEvolutionEngine:
    def __init__(self):
        self.history = []
        self.version = 1.0

    def evaluate_performance(self, metrics):
        score = metrics.get("profit", 0) - metrics.get("drawdown", 0)
        if score < 0:
            self.version += 0.1
            return self.rewrite_strategy("adjust_for_loss")
        elif score > 10:
            self.version += 0.05
            return self.rewrite_strategy("optimize_for_profit")
        else:
            return "stable_no_change"

    def rewrite_strategy(self, mode):
        self.history.append((self.version, mode))
        return f"Strategy evolved to version {self.version:.2f} with mode: {mode}"

    def get_evolution_log(self):
        return self.history
