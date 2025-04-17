
class MultilingualSpeechInterpreter:
    def __init__(self):
        self.commands = []

    def interpret(self, text, language="en"):
        text = text.lower()
        action = "none"

        if language == "en":
            if "buy" in text:
                action = "buy"
            elif "sell" in text:
                action = "sell"
            elif "hold" in text:
                action = "hold"
        elif language == "fa":
            if "بخر" in text:
                action = "buy"
            elif "بفروش" in text:
                action = "sell"
            elif "نگه دار" in text:
                action = "hold"

        self.commands.append((text, language, action))
        return action

    def get_log(self):
        return self.commands
