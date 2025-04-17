
import numpy as np
import datetime

class ExternalRiskManager:
    def __init__(self, volatility_threshold=0.8, news_impact_weight=0.6):
        self.volatility_threshold = volatility_threshold
        self.news_impact_weight = news_impact_weight
        self.current_volatility = 0.0
        self.market_event_risk = 0.0
        self.last_decision = None

    def update_volatility(self, prices):
        returns = np.diff(prices) / prices[:-1]
        self.current_volatility = np.std(returns[-20:])

    def update_news_impact(self, impact_score):
        self.market_event_risk = impact_score

    def assess_risk(self):
        total_risk = (
            self.current_volatility +
            self.market_event_risk * self.news_impact_weight
        )
        return total_risk

    def should_halt_trading(self):
        total_risk = self.assess_risk()
        return total_risk >= self.volatility_threshold

    def generate_signal(self):
        if self.should_halt_trading():
            return {"action": "HOLD", "reason": "High Risk", "timestamp": str(datetime.datetime.now())}
        return {"action": "PROCEED", "risk_score": self.assess_risk()}
