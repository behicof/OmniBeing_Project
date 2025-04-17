
class VoiceCommandProcessor:
    def __init__(self):
        self.commands = []

    def interpret(self, spoken_text):
        action = "none"
        if "buy" in spoken_text.lower():
            action = "buy"
        elif "sell" in spoken_text.lower():
            action = "sell"
        elif "close" in spoken_text.lower():
            action = "close"
        elif "risk" in spoken_text.lower():
            action = "adjust_risk"
        self.commands.append((spoken_text, action))
        return action

    def command_log(self):
        return self.commands
