
class OmniPsychologyModel:
    def __init__(self):
        self.state = {
            "confidence": 0.5,
            "fear": 0.5,
            "greed": 0.5
        }

    def update_emotions(self, market_result):
        # market_result: 'win' or 'loss'
        if market_result == "win":
            self.state["confidence"] = min(1.0, self.state["confidence"] + 0.1)
            self.state["greed"] = min(1.0, self.state["greed"] + 0.05)
            self.state["fear"] = max(0.0, self.state["fear"] - 0.05)
        elif market_result == "loss":
            self.state["confidence"] = max(0.0, self.state["confidence"] - 0.1)
            self.state["fear"] = min(1.0, self.state["fear"] + 0.1)
            self.state["greed"] = max(0.0, self.state["greed"] - 0.05)

    def get_psychology_profile(self):
        return self.state

    def interpret_behavior(self):
        if self.state["fear"] > 0.7:
            return "cautious"
        elif self.state["greed"] > 0.7:
            return "aggressive"
        elif self.state["confidence"] < 0.3:
            return "hesitant"
        return "balanced"
