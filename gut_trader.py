
import random

class IntuitiveDecisionCore:
    def __init__(self):
        self.memory_snapshots = []

    def decide(self, pattern_rarity, memory_match_score, emotional_pressure):
        weight = pattern_rarity * 0.4 + memory_match_score * 0.4 + emotional_pressure * 0.2
        action = "wait"
        if weight > 0.75:
            action = "buy"
        elif weight < 0.35:
            action = "sell"

        self.memory_snapshots.append({
            "rarity": pattern_rarity,
            "match_score": memory_match_score,
            "pressure": emotional_pressure,
            "decision": action
        })

        return action

    def get_memory(self):
        return self.memory_snapshots
