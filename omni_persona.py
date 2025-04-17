
class OmniPersona:
    def __init__(self, persona_type="neutral"):
        self.persona_type = persona_type
        self.profile = self.define_persona(persona_type)

    def define_persona(self, persona_type):
        personas = {
            "neutral": {"risk_tolerance": 0.5, "reaction_speed": "medium", "tone": "calm"},
            "aggressive": {"risk_tolerance": 0.9, "reaction_speed": "fast", "tone": "direct"},
            "conservative": {"risk_tolerance": 0.2, "reaction_speed": "slow", "tone": "soft"},
            "emotional": {"risk_tolerance": 0.4, "reaction_speed": "fast", "tone": "intense"},
        }
        return personas.get(persona_type, personas["neutral"])

    def adjust_behavior(self, market_state):
        # Adjust persona dynamically based on market trends or user input
        if market_state == "volatile" and self.persona_type != "conservative":
            self.persona_type = "conservative"
            self.profile = self.define_persona("conservative")
        return self.profile

    def get_personality(self):
        return self.profile
