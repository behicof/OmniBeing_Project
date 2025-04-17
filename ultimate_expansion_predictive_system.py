
from sklearn.metrics import accuracy_score
import numpy as np

class UltimateExpansionPredictiveSystem:
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=200)
        self.lr_model = LogisticRegression(max_iter=3000)
        self.svm_model = SVC(kernel='linear')
        self.market_data = []
        self.labels = []
        self.multi_stage_predictions = []
        self.model_accuracy = 0
    
    def process_market_data(self, market_data):
        # پردازش داده‌های بازار و آماده‌سازی برای مدل‌های مختلف
        features = [market_data['sentiment'], market_data['volatility'], market_data['price_change']]
        self.market_data.append(features)
        self.labels.append(market_data['buy_sell_signal'])
        return features
    
    def train_models(self):
        # آموزش مدل‌ها با استفاده از داده‌های بازار
        if len(self.market_data) > 0:
            X = np.array(self.market_data)
            y = np.array(self.labels)
            self.rf_model.fit(X, y)
            self.lr_model.fit(X, y)
            self.svm_model.fit(X, y)
    
    def make_predictions(self, market_data):
        # پیش‌بینی با استفاده از مدل‌های مختلف
        features = [market_data['sentiment'], market_data['volatility'], market_data['price_change']]
        rf_prediction = self.rf_model.predict([features])
        lr_prediction = self.lr_model.predict([features])
        svm_prediction = self.svm_model.predict([features])
        
        # ترکیب پیش‌بینی‌ها از مدل‌های مختلف
        combined_prediction = "hold"
        if rf_prediction == 1 and lr_prediction == 1 and svm_prediction == 1:
            combined_prediction = "buy"
        elif rf_prediction == 0 and lr_prediction == 0 and svm_prediction == 0:
            combined_prediction = "sell"
        
        self.multi_stage_predictions.append(combined_prediction)
        return combined_prediction
    
    def evaluate_model_accuracy(self):
        # ارزیابی دقت مدل‌ها
        X = np.array(self.market_data)
        y = np.array(self.labels)
        rf_predictions = self.rf_model.predict(X)
        lr_predictions = self.lr_model.predict(X)
        svm_predictions = self.svm_model.predict(X)
        
        rf_accuracy = accuracy_score(y, rf_predictions)
        lr_accuracy = accuracy_score(y, lr_predictions)
        svm_accuracy = accuracy_score(y, svm_predictions)
        
        self.model_accuracy = (rf_accuracy + lr_accuracy + svm_accuracy) / 3
        return self.model_accuracy
    
    def get_predictions(self):
        return self.multi_stage_predictions
