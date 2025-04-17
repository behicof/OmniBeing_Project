
class LiveVisualAnalyzer:
    def __init__(self):
        self.analysis_log = []

    def analyze_behavior(self, visual_state):
        signal = "neutral"
        if "restless movement" in visual_state or "eye shifting" in visual_state:
            signal = "anxiety_detected"
        elif "nodding" in visual_state or "leaning forward" in visual_state:
            signal = "confidence_detected"
        elif "sudden freeze" in visual_state or "gaze away" in visual_state:
            signal = "shock_or_doubt"
        self.analysis_log.append((visual_state, signal))
        return signal

    def get_behavior_log(self):
        return self.analysis_log
