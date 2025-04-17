
class EmotionalHistoricalSimulator:
    def __init__(self):
        self.simulations = []

    def simulate_event(self, year, emotion, market_reaction):
        story = f"In {year}, emotion was '{emotion}', and it triggered a market reaction of '{market_reaction}'."
        self.simulations.append({
            "year": year,
            "emotion": emotion,
            "reaction": market_reaction,
            "narrative": story
        })
        return story

    def get_simulation_log(self):
        return self.simulations
