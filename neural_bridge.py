
class NeuroSyncInterface:
    def __init__(self):
        self.synced_inputs = []

    def sync_brain_input(self, speech_emotion, trade_reaction, hesitation_level):
        fusion_score = (speech_emotion + trade_reaction) / 2 - hesitation_level
        mode = "observe"
        if fusion_score > 0.5:
            mode = "co_trade"
        elif fusion_score < -0.3:
            mode = "hold_back"

        self.synced_inputs.append({
            "speech_emotion": speech_emotion,
            "trade_reaction": trade_reaction,
            "hesitation": hesitation_level,
            "sync_mode": mode
        })

        return mode

    def log_sync(self):
        return self.synced_inputs
