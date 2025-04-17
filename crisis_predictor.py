
class GlobalCrisisPredictor:
    def __init__(self):
        self.signals = {
            "yield_curve_inversion": False,
            "bank_default_risk": 0,
            "currency_volatility": 0.0,
            "credit_spread_widening": False
        }

    def update_signals(self, inversion, default_risk, currency_vol, spread_widening):
        self.signals["yield_curve_inversion"] = inversion
        self.signals["bank_default_risk"] = default_risk
        self.signals["currency_volatility"] = currency_vol
        self.signals["credit_spread_widening"] = spread_widening

    def assess_crisis_risk(self):
        risk_score = 0
        if self.signals["yield_curve_inversion"]:
            risk_score += 1
        if self.signals["bank_default_risk"] > 7:
            risk_score += 1
        if self.signals["currency_volatility"] > 5.0:
            risk_score += 1
        if self.signals["credit_spread_widening"]:
            risk_score += 1
        return "high_risk" if risk_score >= 3 else "moderate_risk" if risk_score == 2 else "low_risk"

    def signal_status(self):
        return self.signals
