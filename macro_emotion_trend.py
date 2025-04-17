
class MacroEmotionalTrendMapper:
    def __init__(self):
        self.trends = {"fear": 0, "hope": 0, "confidence": 0, "uncertainty": 0}

    def track_trend(self, trend_type):
        if trend_type in self.trends:
            self.trends[trend_type] += 1
        else:
            self.trends[trend_type] = 1

    def dominant_trend(self):
        return max(self.trends, key=self.trends.get)

    def reset(self):
        self.trends = {"fear": 0, "hope": 0, "confidence": 0, "uncertainty": 0}
