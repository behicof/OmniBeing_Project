
class DescriptiveSelf:
    def __init__(self, memory_ref=None, emotion_ref=None, decision_ref=None):
        self.memory = memory_ref
        self.emotion = emotion_ref
        self.decision = decision_ref

    def describe_state(self):
        return {
            "memory": self.memory.memory_summary() if self.memory else "No memory reference",
            "emotion": self.emotion.get_state() if self.emotion else "No emotion reference",
            "last_decision": self.decision.get_last_action() if self.decision else "No decision reference"
        }

    def explain_decision(self):
        if self.decision:
            return self.decision.explain_last_action()
        return "No decision engine available."

    def summarize_self(self):
        state = self.describe_state()
        return f"My current emotional state is {state['emotion']}. I remember: {state['memory']}. My last action was: {state['last_decision']}."
