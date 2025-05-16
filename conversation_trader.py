from avatar_agent import TradingAvatarAgent

class ConversationalTradeEngine:
    def __init__(self):
        self.dialogue_log = []
        self.agent = TradingAvatarAgent()

    def converse(self, user_input):
        response = "I didn't understand. Please rephrase."
        action = None
        text = user_input.lower()

        try:
            if "should i buy" in text:
                response = self.agent.speak("Based on current data, a buy might be valid. Would you like me to proceed?")
                action = "suggest_buy"
                self.suggest_buy()
            elif "sell now" in text:
                response = self.agent.speak("Understood. Preparing to execute a sell order.")
                action = "execute_sell"
                self.execute_sell()
            elif "what's your view" in text:
                response = self.agent.speak("The market looks uncertain, but I’m monitoring signals in real-time.")
                action = "provide_outlook"
                self.provide_outlook()
        except Exception as e:
            response = self.agent.speak(f"An error occurred: {str(e)}")
            self.agent.action_log.append(f"Error: {str(e)}")

        self.dialogue_log.append({
            "user": user_input,
            "bot": response,
            "action": action
        })
        return response, action

    def suggest_buy(self):
        return self.agent.execute_trade("buy")

    def execute_sell(self):
        return self.agent.execute_trade("sell")

    def provide_outlook(self):
        return self.agent.get_status()

    def log(self):
        return self.dialogue_log
