
class TraderDialogueAnalyzer:
    def __init__(self):
        self.dialogues = []

    def analyze(self, dialogue):
        # شبیه‌سازی تحلیل گفت‌وگوی معاملاتی
        sentiment = "neutral"
        dialogue = dialogue.lower()

        if "fear" in dialogue or "drop" in dialogue:
            sentiment = "negative"
        elif "buy" in dialogue or "increase" in dialogue:
            sentiment = "positive"
        elif "uncertain" in dialogue or "volatile" in dialogue:
            sentiment = "neutral"

        self.dialogues.append((dialogue, sentiment))
        return sentiment

    def summary(self):
        sentiment_count = {"positive": 0, "negative": 0, "neutral": 0}
        for _, sentiment in self.dialogues:
            sentiment_count[sentiment] += 1
        return sentiment_count
