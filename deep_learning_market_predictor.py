
import numpy as np
from sklearn.neural_network import MLPClassifier

class DeepLearningMarketPredictor:
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
        self.market_data = []
        self.labels = []

    def process_market_data(self, market_data):
        # پردازش داده‌های بازار و آماده‌سازی برای آموزش مدل
        features = np.array([market_data['sentiment'], market_data['volatility'], market_data['price_change']]).reshape(1, -1)
        self.market_data.append(features)
        self.labels.append(market_data['buy_sell_signal'])  # سیگنال خرید یا فروش
        return features

    def train_model(self):
        # آموزش مدل یادگیری عمیق
        if len(self.market_data) > 0:
            X = np.concatenate(self.market_data, axis=0)
            y = np.array(self.labels)
            self.model.fit(X, y)
    
    def make_predictions(self, market_data):
        # پیش‌بینی با استفاده از مدل یادگیری عمیق
        features = np.array([market_data['sentiment'], market_data['volatility'], market_data['price_change']]).reshape(1, -1)
        prediction = self.model.predict(features)
        return prediction[0]  # بازگشت سیگنال خرید یا فروش
    
    def get_model_accuracy(self):
        # ارزیابی دقت مدل
        correct_predictions = 0
        total_predictions = len(self.labels)
        for i, data in enumerate(self.market_data):
            if self.model.predict(data) == self.labels[i]:
                correct_predictions += 1
        return correct_predictions / total_predictions if total_predictions > 0 else 0
