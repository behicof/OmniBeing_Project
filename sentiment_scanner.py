
class SocialSentimentScanner:
    def __init__(self):
        self.feeds = []

    def scan_feed(self, text):
        text_lower = text.lower()
        if any(word in text_lower for word in ["fear", "crash", "panic", "bearish", "نزول", "ریزش", "ترس"]):
            sentiment = "negative"
        elif any(word in text_lower for word in ["bull", "rally", "profit", "moon", "صعود", "پرواز", "خرید"]):
            sentiment = "positive"
        else:
            sentiment = "neutral"
        self.feeds.append((text, sentiment))
        return sentiment

    def get_sentiment_log(self):
        return self.feeds
