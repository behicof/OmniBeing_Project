
class BehavioralEconomicIntegrator:
    def __init__(self):
        self.bias_signals = {
            "loss_aversion": False,
            "overconfidence": False,
            "herding": False,
            "confirmation_bias": False
        }

    def evaluate_trading_data(self, drawdown, win_rate, sentiment_alignment, cluster_behavior):
        self.bias_signals["loss_aversion"] = drawdown > 10
        self.bias_signals["overconfidence"] = win_rate > 80
        self.bias_signals["herding"] = cluster_behavior > 5
        self.bias_signals["confirmation_bias"] = sentiment_alignment

    def interpret_bias_effect(self):
        active_biases = [k for k, v in self.bias_signals.items() if v]
        if "herding" in active_biases and "overconfidence" in active_biases:
            return "bubble_risk"
        elif "loss_aversion" in active_biases:
            return "panic_sell_risk"
        elif not active_biases:
            return "rational_market"
        return "mixed_bias_effect"

    def bias_snapshot(self):
        return self.bias_signals
