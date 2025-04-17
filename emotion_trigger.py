
class EmotionBasedTradingTrigger:
    def __init__(self):
        self.trigger_events = []

    def evaluate_emotion(self, emotion_label):
        action = "none"
        if emotion_label in ["fear", "shock_or_doubt", "stress"]:
            action = "sell_alert"
        elif emotion_label in ["happy", "confidence_detected"]:
            action = "buy_alert"
        elif emotion_label == "surprise":
            action = "wait_and_watch"
        self.trigger_events.append((emotion_label, action))
        return action

    def get_trigger_log(self):
        return self.trigger_events
