from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier

class FinalExpansionAdvancedPredictiveSystem:
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=350)
        self.lr_model = LogisticRegression(max_iter=7000)
        self.svm_model = SVC(kernel='sigmoid')
        self.gb_model = GradientBoostingClassifier(n_estimators=300)
        self.ada_model = AdaBoostClassifier(n_estimators=200)
        
        self.voting_classifier = VotingClassifier(estimators=[
            ('rf', self.rf_model),
            ('lr', self.lr_model),
            ('svm', self.svm_model),
            ('gb', self.gb_model),
            ('ada', self.ada_model)
        ], voting='hard')
        
        self.market_data = []
        self.labels = []
        self.social_data = []
        self.global_data = []
        self.multi_stage_predictions = []
        self.model_accuracy = 0
    
    def process_market_data(self, market_data):
        # پردازش داده‌های بازار و آماده‌سازی برای مدل‌ها
        features = [market_data['sentiment'], market_data['volatility'], market_data['price_change']]
        self.market_data.append(features)
        self.labels.append(market_data['buy_sell_signal'])
        return features
    
    def fetch_global_data(self, url):
        # دریافت داده‌های جهانی (مثلاً اخبار اقتصادی و رویدادهای جهانی)
        response = requests.get(url)
        global_data = response.json()
        self.global_data.append(global_data)
        return global_data
    
    def fetch_social_data(self, social_media_url):
        # دریافت داده‌های اجتماعی (مثلاً تحلیل احساسات از پلتفرم‌های اجتماعی)
        response = requests.get(social_media_url)
        social_data = response.json()
        self.social_data.append(social_data)
        return social_data
    
    def train_models(self):
        # آموزش مدل‌ها با داده‌های موجود
        if len(self.market_data) > 0:
            X = np.array(self.market_data)
            y = np.array(self.labels)
            self.voting_classifier.fit(X, y)
    
    def make_predictions(self, market_data):
        # پیش‌بینی با استفاده از مدل‌های مختلف
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
        
        self.model_accuracy = classification_report(y, predictions, output_dict=True)
        return self.model_accuracy
    
    def get_predictions(self):
        return self.multi_stage_predictions
