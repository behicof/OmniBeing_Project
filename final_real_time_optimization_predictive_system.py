from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import requests

class FinalRealTimeOptimizationPredictiveSystem:
    def __init__(self):
        """
        Initializes the ensemble predictive system with multiple classifiers and data storage.
        
        Sets up six machine learning classifiers and combines them into a hard voting ensemble for market prediction. Initializes lists for storing market data, labels, predictions, live data, and a variable for model accuracy.
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
        # Receive live data from various platforms
        """
        Appends incoming live data from a specified platform to the internal live data list.
        
        Args:
            platform_name: Name of the platform providing the live data.
            data: The live market data received from the platform.
        """
        self.live_data.append({"platform": platform_name, "data": data})

    def process_market_data(self, market_data):
        # پردازش داده‌های بازار برای تحلیل و پیش‌بینی
        """
        Extracts features from market data and stores them with the corresponding label.
        
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
        
        If market data is available, fits the voting classifier on the stored features and corresponding labels.
        
        Returns:
            None
        """
        if len(self.market_data) > 0:
            X = np.array(self.market_data)
            y = np.array(self.labels)
            self.voting_classifier.fit(X, y)
        return None
    
    def make_predictions(self, market_data):
        """
        Makes predictions using the ensemble model based on the provided market data.
        
        Args:
            market_data: A dictionary containing 'sentiment', 'volatility', and 'price_change' keys.
        
        Returns:
            A string representing the combined prediction: 'buy', 'sell', or 'hold'.
        """
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
        """
        Evaluates the accuracy of the ensemble model using the stored market data and labels.
        
        Returns:
            A dictionary containing the classification report of the model's performance.
        """
        X = np.array(self.market_data)
        y = np.array(self.labels)
        predictions = self.voting_classifier.predict(X)
        
        self.model_accuracy = classification_report(y, predictions, output_dict=True)
        return self.model_accuracy
    
    def get_predictions(self):
        """
        Returns the list of all predictions made by the ensemble system so far.
        """
        return self.multi_stage_predictions
    
    def fetch_live_data(self, platform_name, url):
        """
        Fetches live market data from a specified URL and stores it with platform metadata.
        
        Args:
            platform_name: Name of the platform providing the data.
            url: The URL endpoint to fetch live market data from.
        
        Returns:
            The parsed JSON data retrieved from the URL.
        
        Raises:
            requests.exceptions.RequestException: If there is an issue with the network request.
            JSONDecodeError: If there is an issue with parsing the JSON response.
        """
        response = requests.get(url)
        response.raise_for_status()  # Raises a HTTPError for bad responses
        data = response.json()
        self.receive_live_data(platform_name, data)
        return data

    def process_live_data(self):
        """
        Processes all stored live data entries and generates predictions for each.
        
        Iterates through the collected live data, processes each entry's market data, and produces a prediction using the ensemble model.
        """
        for entry in self.live_data:
            platform_name = entry['platform']
            data = entry['data']
            self.process_market_data(data)
            self.make_predictions(data)
