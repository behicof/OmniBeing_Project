class OptimizedPredictiveSystem:
    def __init__(self):
        self.model_data = []
        self.market_signals = []
        self.predictions = []
        self.dynamic_thresholds = {
            "buy_threshold": 0.7,
            "sell_threshold": 0.3,
            "hold_threshold": 0.5
        }
    
    def process_market_data(self, market_data):
        # تحلیل داده‌ها با فیلترهای پویا و به‌روزرسانی سیگنال‌های بازار
        signal = self.analyze_data(market_data)
        self.market_signals.append(signal)
    
    def analyze_data(self, market_data):
        # شبیه‌سازی الگوهای روانشناسی و تصمیم‌گیری با فیلترهای پویا
        if market_data['price_trend'] == "up" and market_data['market_sentiment'] > self.dynamic_thresholds["buy_threshold"]:
            return "buy"
        elif market_data['price_trend'] == "down" and market_data['market_sentiment'] < self.dynamic_thresholds["sell_threshold"]:
            return "sell"
        elif market_data['market_sentiment'] < self.dynamic_thresholds["hold_threshold"]:
            return "hold"
        else:
            return "monitor"
    
    def make_predictions(self):
        # پیش‌بینی روند بازار و اصلاح استراتژی‌های تصمیم‌گیری
        for signal in self.market_signals:
            if signal == "buy":
                self.predictions.append("Prediction: Buy next.")
            elif signal == "sell":
                self.predictions.append("Prediction: Sell next.")
            elif signal == "hold":
                self.predictions.append("Prediction: Hold position.")
            else:
                self.predictions.append("Prediction: Monitor market.")
    
    def get_predictions(self):
        return self.predictions
    
    def adjust_thresholds(self, new_buy_threshold, new_sell_threshold, new_hold_threshold):
        # به‌روزرسانی فیلترهای پویا برای بهینه‌سازی سیستم
        self.dynamic_thresholds["buy_threshold"] = new_buy_threshold
        self.dynamic_thresholds["sell_threshold"] = new_sell_threshold
        self.dynamic_thresholds["hold_threshold"] = new_hold_threshold
