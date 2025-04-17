
import random

class InnovationGenerator:
    def __init__(self):
        self.generated_ideas = []

    def generate_new_strategy(self):
        base = ["momentum", "mean_reversion", "breakout", "news_reaction", "sentiment"]
        feature = ["AI filter", "volatility filter", "pattern sync", "macro overlay", "volume spike"]
        strat = f"{random.choice(base)} with {random.choice(feature)}"
        self.generated_ideas.append(strat)
        return strat

    def list_innovations(self):
        return self.generated_ideas
