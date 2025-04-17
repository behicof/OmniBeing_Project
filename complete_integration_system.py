
import requests
from sklearn.neural_network import MLPClassifier

class CompleteIntegrationSystem:
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
        self.market_data = []
        self.news_data = []
        self.social_data = []
        self.labels = []
        self.decisions = []

    def process_market_data(self, market_data):
        # پردازش داده‌های بازار
        features = [market_data['sentiment'], market_data['volatility'], market_data['price_change']]
        self.market_data.append(features)
        self.labels.append(market_data['buy_sell_signal'])
        return features

    def fetch_and_analyze_news(self, news_url):
        # دریافت داده‌های خبری و تحلیل آن‌ها
        response = requests.get(news_url)
        news_content = response.text
        sentiment = self.analyze_sentiment(news_content)
        self.news_data.append(sentiment)
        return sentiment

    def analyze_sentiment(self, text):
        # تحلیل احساسات از متن خبری
        positive_words = ["growth", "profit", "bullish", "optimistic"]
        negative_words = ["crash", "loss", "bearish", "pessimistic"]
        sentiment_score = 0
        for word in positive_words:
            if word in text.lower():
                sentiment_score += 1
        for word in negative_words:
            if word in text.lower():
                sentiment_score -= 1
        return sentiment_score

    def process_social_data(self, social_data):
        # پردازش داده‌های اجتماعی
        sentiment = self.analyze_sentiment(social_data['content'])
        self.social_data.append(sentiment)
        return sentiment

    def train_model(self):
        # آموزش مدل یادگیری عمیق
        if len(self.market_data) > 0:
            X = np.array(self.market_data)
            y = np.array(self.labels)
            self.model.fit(X, y)

    def make_predictions(self, market_data):
        # پیش‌بینی سیگنال خرید یا فروش
        features = [market_data['sentiment'], market_data['volatility'], market_data['price_change']]
        prediction = self.model.predict([features])
        return prediction[0]

    def get_decisions(self):
        # دریافت تصمیمات نهایی
        return self.decisions

    def generate_decision(self, market_data, news_sentiment, social_sentiment):
        # ترکیب داده‌های بازار، خبری و اجتماعی برای تصمیم‌گیری
        combined_sentiment = (market_data['sentiment'] + news_sentiment + social_sentiment) / 3
        decision = "hold"
        if combined_sentiment > 0.7:
            decision = "buy"
        elif combined_sentiment < 0.3:
            decision = "sell"
        self.decisions.append(decision)
        return decision
