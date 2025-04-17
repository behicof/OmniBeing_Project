
import requests
import json
from sklearn.neural_network import MLPClassifier

class MultiMarketDeepLearning:
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
        self.market_data = []
        self.labels = []
        self.social_data = []
        self.news_data = []

    def process_market_data(self, market_data):
        # پردازش داده‌های بازار و آماده‌سازی برای آموزش مدل
        features = [market_data['sentiment'], market_data['volatility'], market_data['price_change']]
        self.market_data.append(features)
        self.labels.append(market_data['buy_sell_signal'])  # سیگنال خرید یا فروش
        return features

    def fetch_news_sentiment(self, news_url):
        # دریافت احساسات از اخبار و تجزیه‌وتحلیل برای پیش‌بینی
        response = requests.get(news_url)
        news_content = response.text
        sentiment = self.analyze_sentiment(news_content)
        self.news_data.append(sentiment)
        return sentiment

    def analyze_sentiment(self, text):
        # تحلیل احساسات از متن
        positive_words = ["profit", "growth", "bull", "positive"]
        negative_words = ["crash", "fall", "bear", "negative"]
        sentiment_score = 0
        for word in positive_words:
            if word in text.lower():
                sentiment_score += 1
        for word in negative_words:
            if word in text.lower():
                sentiment_score -= 1
        return sentiment_score

    def process_social_data(self, social_data):
        # پردازش داده‌های اجتماعی برای پیش‌بینی
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
        # پیش‌بینی با استفاده از مدل یادگیری عمیق
        features = [market_data['sentiment'], market_data['volatility'], market_data['price_change']]
        prediction = self.model.predict([features])
        return prediction[0]  # بازگشت سیگنال خرید یا فروش

    def get_model_accuracy(self):
        # ارزیابی دقت مدل
        correct_predictions = 0
        total_predictions = len(self.labels)
        for i, data in enumerate(self.market_data):
            if self.model.predict([data]) == self.labels[i]:
                correct_predictions += 1
        return correct_predictions / total_predictions if total_predictions > 0 else 0
