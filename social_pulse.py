
class SocialPulseReader:
    def __init__(self):
        self.keywords = ["bullish", "bearish", "crash", "moon", "fear", "buy", "sell"]
        self.trend_count = {k: 0 for k in self.keywords}

    def scan_post(self, text):
        lowered = text.lower()
        for word in self.keywords:
            if word in lowered:
                self.trend_count[word] += 1

    def trending_sentiment(self):
        sorted_trends = sorted(self.trend_count.items(), key=lambda x: x[1], reverse=True)
        return sorted_trends[:3]

    def reset(self):
        self.trend_count = {k: 0 for k in self.keywords}
