
class CrowdEmotionSynthesizer:
    def __init__(self):
        self.posts = []
        self.emotion_scores = {"fear": 0, "greed": 0, "hope": 0, "anger": 0}

    def process_post(self, post_text):
        text = post_text.lower()
        if any(word in text for word in ["afraid", "sell now", "panic", "nervous"]):
            self.emotion_scores["fear"] += 1
        if any(word in text for word in ["buy now", "fomo", "moon", "fast"]):
            self.emotion_scores["greed"] += 1
        if any(word in text for word in ["hope", "wait", "recover"]):
            self.emotion_scores["hope"] += 1
        if any(word in text for word in ["angry", "scam", "loss", "hate"]):
            self.emotion_scores["anger"] += 1
        self.posts.append(post_text)

    def synthesize(self):
        dominant = max(self.emotion_scores.items(), key=lambda x: x[1])
        return {"dominant_emotion": dominant[0], "score": dominant[1], "full": self.emotion_scores}
