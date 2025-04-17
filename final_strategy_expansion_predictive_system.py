
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier

class FinalStrategyExpansionPredictiveSystem:
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=400)
        self.lr_model = LogisticRegression(max_iter=7000)
        self.svm_model = SVC(kernel='poly')
        self.gb_model = GradientBoostingClassifier(n_estimators=350)
        self.ada_model = AdaBoostClassifier(n_estimators=200)
        self.stacking_classifier = StackingClassifier(estimators=[
            ('rf', self.rf_model),
            ('lr', self.lr_model),
            ('svm', self.svm_model),
            ('gb', self.gb_model),
            ('ada', self.ada_model)
        ], final_estimator=LogisticRegression())
        
        self.market_data = []
        self.labels = []
        self.multi_stage_predictions = []
        self.model_accuracy = 0
    
    def process_market_data(self, market_data):
        # پردازش داده‌های بازار
        features = [market_data['sentiment'], market_data['volatility'], market_data['price_change']]
        self.market_data.append(features)
        self.labels.append(market_data['buy_sell_signal'])
        return features
    
    def train_models(self):
        # آموزش مدل‌ها با داده‌های موجود
        if len(self.market_data) > 0:
            X = np.array(self.market_data)
            y = np.array(self.labels)
            self.stacking_classifier.fit(X, y)
    
    def make_predictions(self, market_data):
        # پیش‌بینی با استفاده از مدل‌های مختلف
        features = [market_data['sentiment'], market_data['volatility'], market_data['price_change']]
        prediction = self.stacking_classifier.predict([features])
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
        predictions = self.stacking_classifier.predict(X)
        
        self.model_accuracy = accuracy_score(y, predictions)
        return self.model_accuracy
    
    def get_predictions(self):
        return self.multi_stage_predictions
