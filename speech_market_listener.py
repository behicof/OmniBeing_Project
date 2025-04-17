
class SpeechMarketListener:
    def __init__(self):
        self.news_headlines = []

    def analyze_headline(self, headline):
        sentiment = "neutral"
        h = headline.lower()
        if "crash" in h or "fall" in h:
            sentiment = "bearish"
        elif "rally" in h or "surge" in h:
            sentiment = "bullish"
        elif "uncertain" in h or "doubt" in h:
            sentiment = "confused"
        self.news_headlines.append((headline, sentiment))
        return sentiment

    def get_sentiment_log(self):
        return self.news_headlines
