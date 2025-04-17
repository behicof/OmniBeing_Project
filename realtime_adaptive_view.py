
class RealtimeAdaptiveView:
    def __init__(self):
        self.signals = []

    def integrate(self, mood, emotion_spike, price):
        decision = "hold"
        if mood == "greedy" and emotion_spike == "strong_sentiment":
            decision = "buy"
        elif mood == "fearful" and emotion_spike == "strong_sentiment":
            decision = "sell"
        elif emotion_spike == "indecision_spike":
            decision = "wait"
        self.signals.append({
            "mood": mood,
            "emotion": emotion_spike,
            "price": price,
            "decision": decision
        })
        return decision

    def last_signal(self):
        return self.signals[-1] if self.signals else None
