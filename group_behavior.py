
class GroupBehaviorModeler:
    def __init__(self):
        self.group_decisions = {"buy": 0, "sell": 0, "wait": 0}

    def observe(self, dialogue_sentiment):
        if dialogue_sentiment == "positive":
            self.group_decisions["buy"] += 1
        elif dialogue_sentiment == "negative":
            self.group_decisions["sell"] += 1
        else:
            self.group_decisions["wait"] += 1

    def predict_group_action(self):
        majority_decision = max(self.group_decisions, key=self.group_decisions.get)
        return majority_decision

    def reset(self):
        self.group_decisions = {"buy": 0, "sell": 0, "wait": 0}
