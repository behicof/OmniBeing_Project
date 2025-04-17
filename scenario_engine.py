
class ScenarioEngine:
    def __init__(self):
        self.scenarios = []

    def create_scenario(self, name, event_type, volatility_level, emotional_shift):
        scenario = {
            "name": name,
            "event_type": event_type,
            "volatility": volatility_level,
            "emotion": emotional_shift
        }
        self.scenarios.append(scenario)
        return scenario

    def list_scenarios(self):
        return self.scenarios

    def simulate_effect(self, scenario):
        if scenario["volatility"] > 7 and scenario["emotion"] == "panic":
            return "high_risk_sell_off"
        elif scenario["event_type"] == "policy_shift" and scenario["emotion"] == "optimism":
            return "potential_rally"
        return "uncertain_reaction"
