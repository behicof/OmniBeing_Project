
class ConversationalDecisionExecutor:
    def __init__(self):
        self.decision_log = []

    def execute_decision(self, predicted_action):
        action_result = "none"
        if predicted_action == "buy":
            action_result = "Executing buy order based on group sentiment."
        elif predicted_action == "sell":
            action_result = "Executing sell order based on group sentiment."
        elif predicted_action == "wait":
            action_result = "Waiting for better market conditions as per group consensus."
        self.decision_log.append((predicted_action, action_result))
        return action_result

    def get_decision_log(self):
        return self.decision_log
