
class HumanAgentDialogueCore:
    def __init__(self):
        self.dialogue_history = []

    def process_input(self, user_text):
        intent = "unknown"
        response = "Can you clarify your request?"

        if "should i buy" in user_text.lower() or "بخرم؟" in user_text:
            intent = "buy_question"
            response = "Let's look at the current signals together before deciding."
        elif "what's your analysis" in user_text.lower() or "تحلیل تو چیه" in user_text:
            intent = "analysis_request"
            response = "I’ll check the market and share my insights."
        elif "execute now" in user_text.lower() or "الان بزن" in user_text:
            intent = "execute_trade"
            response = "Trade executing now. Monitoring closely."

        self.dialogue_history.append((user_text, intent, response))
        return {"intent": intent, "response": response}

    def get_log(self):
        return self.dialogue_history
