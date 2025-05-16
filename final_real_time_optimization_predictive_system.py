from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import requests
import numpy as np
from sklearn.metrics import accuracy_score

class FinalRealTimeOptimizationPredictiveSystem:
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=500)
        self.lr_model = LogisticRegression(max_iter=10000)
        self.svm_model = SVC(kernel='poly')
        self.gb_model = GradientBoostingClassifier(n_estimators=400)
        self.ada_model = AdaBoostClassifier(n_estimators=250)
        self.dt_model = DecisionTreeClassifier(max_depth=15)
        
        self.voting_classifier = VotingClassifier(estimators=[
            ('rf', self.rf_model),
            ('lr', self.lr_model),
            ('svm', self.svm_model),
            ('gb', self.gb_model),
            ('ada', self.ada_model),
            ('dt', self.dt_model)
        ], voting='hard')
        
        self.market_data = []
        self.labels = []
        self.multi_stage_predictions = []
        self.model_accuracy = 0
        self.live_data = []

    def receive_live_data(self, platform_name, data):
        # دریافت داده‌های زنده از پلتفرم‌های مختلف
        self.live_data.append({"platform": platform_name, "data": data})

    def process_market_data(self, market_data):
        # پردازش داده‌های بازار برای تحلیل و پیش‌بینی
        features = [market_data['sentiment'], market_data['volatility'], market_data['price_change']]
        self.market_data.append(features)
        self.labels.append(market_data['buy_sell_signal'])
        return features

    def train_models(self):
        # آموزش مدل‌ها با داده‌های بازار
        if len(self.market_data) > 0:
            X = np.array(self.market_data)
            y = np.array(self.labels)
            self.voting_classifier.fit(X, y)
    
    def make_predictions(self, market_data):
        # پیش‌بینی با مدل‌های ترکیبی
        features = [market_data['sentiment'], market_data['volatility'], market_data['price_change']]
        prediction = self.voting_classifier.predict([features])
        combined_prediction = "hold"
        
        if prediction == 1:
            combined_prediction = "buy"
        elif prediction == 0:
            combined_prediction = "sell"
        
        self.multi_stage_predictions.append(combined_prediction)
        return combined_prediction
    
    def evaluate_model_accuracy(self):
        # ارزیابی دقت مدل‌ها
        X = np.array(self.market_data)
        y = np.array(self.labels)
        predictions = self.voting_classifier.predict(X)
        
        self.model_accuracy = accuracy_score(y, predictions)
        return self.model_accuracy
    
    def get_predictions(self):
        return self.multi_stage_predictions

    def handle_multiple_sources(self, data_sources):
        # پردازش داده‌ها از منابع مختلف به صورت همزمان
        for source in data_sources:
            self.receive_live_data(source['platform'], source['data'])

    def validate_data(self, data):
        # اعتبارسنجی داده‌های ورودی
        required_keys = ['sentiment', 'volatility', 'price_change', 'buy_sell_signal']
        for key in required_keys:
            if key not in data:
                return False
        return True

    def handle_missing_data(self, data):
        # مدیریت داده‌های ناقص
        for key in data:
            if data[key] is None:
                data[key] = 0  # جایگزینی داده‌های ناقص با مقدار پیش‌فرض
        return data

    def synchronize_data_sources(self):
        # همگام‌سازی داده‌ها از منابع مختلف
        synchronized_data = []
        for entry in self.live_data:
            if self.validate_data(entry['data']):
                data = self.handle_missing_data(entry['data'])
                synchronized_data.append(data)
        return synchronized_data

    def process_live_data(self):
        # پردازش داده‌های زنده و پیش‌بینی با استفاده از مدل‌ها
        synchronized_data = self.synchronize_data_sources()
        for data in synchronized_data:
            self.process_market_data(data)
        self.train_models()
        predictions = []
        for data in synchronized_data:
            prediction = self.make_predictions(data)
            predictions.append(prediction)
        return predictions

    def audit_data(self):
        # بررسی دوره‌ای داده‌ها برای اطمینان از صحت و یکپارچگی
        for entry in self.live_data:
            if not self.validate_data(entry['data']):
                self.live_data.remove(entry)
