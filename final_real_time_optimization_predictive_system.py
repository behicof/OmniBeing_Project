import requests
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
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
        self.live_data.append(data)
        return f"Data from {platform_name} received."

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

    def integrate_data_from_multiple_platforms(self, urls):
        # دریافت داده‌ها از پلتفرم‌های مختلف
        for url in urls:
            response = requests.get(url)
            data = response.json()
            self.receive_live_data(url, data)
