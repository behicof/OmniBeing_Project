
class EmotionalResponseEngine:
    def __init__(self):
        self.history = []

    def respond(self, message):
        text = message.lower()
        tone = "neutral"
        response = "I'm here to help you make better trading decisions."

        if any(word in text for word in ["scared", "worried", "panic", "afraid", "ترس", "اضطراب"]):
            tone = "reassuring"
            response = "Don't worry. Let's assess the market together calmly."
        elif any(word in text for word in ["excited", "great", "awesome", "خیلی خوب", "عالی"]):
            tone = "encouraging"
            response = "That's amazing! Let's make the most of this momentum."
        elif any(word in text for word in ["angry", "hate", "bad", "بد", "خسته شدم"]):
            tone = "soothing"
            response = "I understand your frustration. Let's take a breath and look again."
        
        self.history.append((message, tone, response))
        return {"tone": tone, "response": response}

    def get_emotional_log(self):
        return self.history
