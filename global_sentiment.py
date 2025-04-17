
class GlobalSentimentIntegrator:
    def __init__(self):
        self.global_sentiment_scores = {"positive": 0, "negative": 0, "neutral": 0}

    def integrate_sentiment(self, sentiment):
        if sentiment == "positive":
            self.global_sentiment_scores["positive"] += 1
        elif sentiment == "negative":
            self.global_sentiment_scores["negative"] += 1
        else:
            self.global_sentiment_scores["neutral"] += 1

    def dominant_sentiment(self):
        return max(self.global_sentiment_scores, key=self.global_sentiment_scores.get)

    def reset(self):
        self.global_sentiment_scores = {"positive": 0, "negative": 0, "neutral": 0}
