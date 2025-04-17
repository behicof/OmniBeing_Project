
class SocioeconomicMarketModeler:
    def __init__(self):
        self.indicators = {
            "unemployment_rate": 0.0,
            "inflation_rate": 0.0,
            "consumer_confidence": 0.0,
            "protest_activity": 0,
            "policy_changes": 0
        }

    def update_indicators(self, unemployment, inflation, confidence, protests, policy):
        self.indicators["unemployment_rate"] = unemployment
        self.indicators["inflation_rate"] = inflation
        self.indicators["consumer_confidence"] = confidence
        self.indicators["protest_activity"] = protests
        self.indicators["policy_changes"] = policy

    def assess_market_impact(self):
        if self.indicators["inflation_rate"] > 6 or self.indicators["unemployment_rate"] > 8:
            return "bearish_outlook"
        if self.indicators["consumer_confidence"] > 75 and self.indicators["policy_changes"] < 2:
            return "bullish_outlook"
        if self.indicators["protest_activity"] > 5:
            return "risk_alert"
        return "neutral"

    def get_snapshot(self):
        return self.indicators
