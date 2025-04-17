
class MetacognitiveAnalyzer:
    def __init__(self):
        self.decision_logs = []
        self.analysis_log = []

    def analyze_decision(self, decision, outcome):
        analysis = {
            "decision": decision,
            "outcome": outcome,
            "correct": "yes" if outcome == "profit" else "no"
        }
        self.decision_logs.append(analysis)
        self._evaluate_decision(analysis)

    def _evaluate_decision(self, analysis):
        if analysis["correct"] == "no":
            self.analysis_log.append(f"Review decision: {analysis['decision']} resulted in loss.")
        else:
            self.analysis_log.append(f"Decision {analysis['decision']} led to a successful outcome.")

    def get_decision_analysis(self):
        return self.analysis_log
