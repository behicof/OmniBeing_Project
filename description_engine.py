
class DescriptiveIntelligenceEngine:
    def __init__(self):
        self.events = []

    def describe(self, data_point):
        text = ""
        if data_point["action"] == "buy":
            text = f"A buy signal was generated because the trend is upward and momentum is strong."
        elif data_point["action"] == "sell":
            text = f"A sell decision was triggered due to a bearish divergence and increased volatility."
        else:
            text = f"The system is waiting for a clearer pattern before executing."

        self.events.append((data_point, text))
        return text

    def explanation_history(self):
        return self.events
