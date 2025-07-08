from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import requests

class FinalRealTimeOptimizationPredictiveSystem:
    def __init__(self):
        """
        Initializes the ensemble predictive system with multiple machine learning classifiers.
        
        Sets up six individual classifiers and combines them into a hard voting ensemble. Initializes storage for market data, labels, predictions, model accuracy, and live data entries.
        """
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
        """
        Appends incoming live market data from a specified platform to the internal storage.
        
        Args:
        	platform_name: The name of the platform providing the live data.
        	data: The live market data to be stored.
        """
        self.live_data.append({"platform": platform_name, "data": data})

    def process_market_data(self, market_data):
        # پردازش داده‌های بازار برای تحلیل و پیش‌بینی
        """
        Extracts features and label from market data and stores them for training.
        
        Args:
        	market_data: A dictionary containing 'sentiment', 'volatility', 'price_change', and 'buy_sell_signal' keys.
        
        Returns:
        	A list of extracted feature values: [sentiment, volatility, price_change].
        """
        features = [market_data['sentiment'], market_data['volatility'], market_data['price_change']]
        self.market_data.append(features)
        self.labels.append(market_data['buy_sell_signal'])
        return features

    def train_models(self):
        # آموزش مدل‌ها با داده‌های بازار
        """
        Trains the ensemble voting classifier using accumulated market data and labels.
        
        The method fits the voting classifier on all stored feature vectors and their corresponding labels if any data is available.
        """
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
        """
        Returns the list of all predictions made by the ensemble system so far.
        """
        return self.multi_stage_predictions

    def fetch_live_data(self, platform_name, url):
        """
        Fetches live market data from a specified URL and stores it with the associated platform name.
        
        Args:
            platform_name: The name of the platform providing the data.
            url: The URL endpoint to fetch live market data from.
        
        Returns:
            The parsed JSON data retrieved from the URL.
        """
        response = requests.get(url)
        data = response.json()
        self.receive_live_data(platform_name, data)
        return data

    def process_live_data(self):
        """
        Processes all stored live data entries and generates predictions for each.
        
        Iterates over the collected live data, extracts features from each entry, and produces predictions using the ensemble model.
        """
        for entry in self.live_data:
            platform_name = entry['platform']
            data = entry['data']
            self.process_market_data(data)
            self.make_predictions(data)
