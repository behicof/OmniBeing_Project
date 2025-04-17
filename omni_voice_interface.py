
class OmniVoiceInterface:
    def __init__(self):
        self.voice_mode = "neutral"

    def set_voice_mode(self, mood):
        if mood in ["calm", "direct", "intense", "soft"]:
            self.voice_mode = mood

    def speak(self, message):
        if self.voice_mode == "calm":
            return f"[Calm Voice] {message}"
        elif self.voice_mode == "direct":
            return f"[Direct Tone] {message}"
        elif self.voice_mode == "intense":
            return f"[Intense Emotion] {message.upper()}!"
        elif self.voice_mode == "soft":
            return f"[Soft and Gentle] {message}"
        return f"[Neutral] {message}"

    def alert(self, level, context=""):
        if level == "high":
            self.set_voice_mode("intense")
            return self.speak(f"Warning! {context}")
        elif level == "medium":
            self.set_voice_mode("direct")
            return self.speak(f"Notice: {context}")
        else:
            self.set_voice_mode("calm")
            return self.speak(f"Status normal. {context}")
