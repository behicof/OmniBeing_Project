
class MetacognitionCore:
    def __init__(self):
        self.logs = []
        self.self_evaluation = []

    def monitor_action(self, action, outcome, context):
        log_entry = {"action": action, "outcome": outcome, "context": context}
        self.logs.append(log_entry)

    def evaluate_last(self):
        if not self.logs:
            return "No actions logged yet."
        last = self.logs[-1]
        eval_result = self.reflect(last)
        self.self_evaluation.append(eval_result)
        return eval_result

    def reflect(self, entry):
        if entry["outcome"] == "fail":
            return f"Analyzed failure: reconsider strategy for {entry['action']} under {entry['context']}"
        return f"Confirmed success: continue with similar actions in context {entry['context']}"

    def get_summary(self):
        return {
            "total": len(self.logs),
            "failures": sum(1 for e in self.logs if e["outcome"] == "fail"),
            "successes": sum(1 for e in self.logs if e["outcome"] == "success"),
        }
