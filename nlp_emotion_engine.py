
class NLPEmotionEngine:
    def __init__(self):
        self.terms = {
            "encouraging": ["growth", "booming", "record high", "bullish"],
            "warning": ["recession", "crisis", "crash", "fragile"],
            "uncertain": ["maybe", "unclear", "volatile", "speculation"],
            "manipulative": ["guaranteed", "must", "no risk", "perfect"]
        }
        self.detections = []

    def scan(self, text):
        results = []
        lowered = text.lower()
        for emotion, words in self.terms.items():
            for word in words:
                if word in lowered:
                    results.append(emotion)
                    break
        final = results[0] if results else "neutral"
        self.detections.append((text, final))
        return final

    def report(self):
        summary = {}
        for _, label in self.detections:
            summary[label] = summary.get(label, 0) + 1
        return summary
