
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class AdvancedPredictiveSystem:
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100)
        self.lr_model = LogisticRegression()
        self.market_data = []
        self.labels = []
        self.multi_stage_predictions = []

    def process_market_data(self, market_data):
        # پردازش داده‌های بازار برای هر مرحله از پیش‌بینی
        features = [market_data['sentiment'], market_data['volatility'], market_data['price_change']]
        self.market_data.append(features)
        self.labels.append(market_data['buy_sell_signal'])  # سیگنال خرید یا فروش
        return features

    def train_models(self):
        # آموزش مدل‌های پیش‌بینی
        if len(self.market_data) > 0:
            X = np.array(self.market_data)
            y = np.array(self.labels)
            self.rf_model.fit(X, y)
            self.lr_model.fit(X, y)

    def make_predictions(self, market_data):
        # پیش‌بینی با استفاده از مدل‌های پیشرفته
        features = [market_data['sentiment'], market_data['volatility'], market_data['price_change']]
        rf_prediction = self.rf_model.predict([features])
        lr_prediction = self.lr_model.predict([features])
        
        # ترکیب پیش‌بینی‌ها از هر مدل
        combined_prediction = "hold"
        if rf_prediction == 1 and lr_prediction == 1:
            combined_prediction = "buy"
        elif rf_prediction == 0 and lr_prediction == 0:
            combined_prediction = "sell"
        
        self.multi_stage_predictions.append(combined_prediction)
        return combined_prediction

    def get_predictions(self):
        return self.multi_stage_predictions
