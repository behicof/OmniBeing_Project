
class TradingAvatarAgent:
    def __init__(self, name="OmniBot"):
        self.name = name
        self.position = 0
        self.action_log = []

    def speak(self, message):
        return f"{self.name} says: {message}"

    def move_to(self, position):
        self.position = position
        return f"{self.name} moved to position {self.position}"

    def execute_trade(self, signal):
        result = f"{self.name} executed {signal} trade"
        self.action_log.append(result)
        return result

    def get_status(self):
        return {
            "name": self.name,
            "position": self.position,
            "last_action": self.action_log[-1] if self.action_log else "none"
        }
