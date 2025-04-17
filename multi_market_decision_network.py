
class MultiMarketDecisionNetwork:
    def __init__(self):
        self.market_data = {}
        self.market_decisions = []
        self.global_predictions = []
    
    def process_market_data(self, market_name, market_data):
        self.market_data[market_name] = market_data
        signal = self.analyze_market_data(market_name, market_data)
        self.market_decisions.append({"market": market_name, "signal": signal})
    
    def analyze_market_data(self, market_name, market_data):
        if market_data['sentiment'] > 0.7 and market_data['volatility'] < 0.5:
            return "buy"
        elif market_data['sentiment'] < 0.3 and market_data['volatility'] > 0.7:
            return "sell"
        else:
            return "hold"
    
    def make_global_predictions(self):
        self.global_predictions = []
        for market_name, decision in self.market_decisions:
            if decision['signal'] == "buy":
                self.global_predictions.append(f"Buy in {market_name}")
            elif decision['signal'] == "sell":
                self.global_predictions.append(f"Sell in {market_name}")
            else:
                self.global_predictions.append(f"Hold position in {market_name}")
    
    def get_global_predictions(self):
        return self.global_predictions
    
    def get_market_decisions(self):
        return self.market_decisions
