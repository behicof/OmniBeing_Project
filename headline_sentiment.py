
class HeadlineSentimentDecoder:
    def __init__(self):
        self.history = []

    def analyze(self, headline):
        text = headline.lower()
        sentiment = "neutral"

        if any(word in text for word in ["crash", "collapse", "panic", "fear"]):
            sentiment = "negative"
        elif any(word in text for word in ["rally", "soar", "gain", "optimism"]):
            sentiment = "positive"
        elif any(word in text for word in ["mixed", "uncertain", "conflict", "volatile"]):
            sentiment = "confused"

        self.history.append({"headline": headline, "sentiment": sentiment})
        return sentiment

    def summary(self):
        return {s["sentiment"]: sum(1 for h in self.history if h["sentiment"] == s["sentiment"]) for s in self.history}
