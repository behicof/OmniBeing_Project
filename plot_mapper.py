
class DecisionPlotMapper:
    def __init__(self):
        self.decision_paths = []

    def map_decision(self, context, options):
        decision_tree = {
            "context": context,
            "options": options,
            "outcomes": {opt: f"Outcome of choosing '{opt}' under {context}" for opt in options}
        }
        self.decision_paths.append(decision_tree)
        return decision_tree

    def get_all_paths(self):
        return self.decision_paths
