
class ToneAnalysisInterpreter:
    def __init__(self):
        self.entries = []

    def interpret(self, sentence):
        tone = "neutral"
        text = sentence.lower()
        if any(t in text for t in ["must", "urgent", "now", "act fast"]):
            tone = "aggressive"
        elif any(t in text for t in ["may", "consider", "possibly", "perhaps"]):
            tone = "cautious"
        elif any(t in text for t in ["definitely", "assured", "certain", "guaranteed"]):
            tone = "overconfident"
        elif any(t in text for t in ["we don't know", "uncertain", "unclear", "confusing"]):
            tone = "uncertain"
        self.entries.append((sentence, tone))
        return tone

    def history(self):
        return self.entries
