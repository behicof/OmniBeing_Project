
class LiveVisualInterface:
    def __init__(self):
        self.sessions = []

    def start_session(self, trader_id, facial_state, visual_response):
        if facial_state == "fear" or visual_response == "stressed":
            status = "warn_and_monitor"
        elif facial_state == "confident" and visual_response == "focused":
            status = "allow_execution"
        else:
            status = "observe_only"

        session = {
            "trader_id": trader_id,
            "facial_state": facial_state,
            "visual_response": visual_response,
            "status": status
        }
        self.sessions.append(session)
        return status

    def get_live_log(self):
        return self.sessions
