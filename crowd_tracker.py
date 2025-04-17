
class LiveBehaviorTracker:
    def __init__(self):
        self.behavior_log = []

    def track_behavior(self, buy_volume, sell_volume, headline_reaction):
        net_flow = buy_volume - sell_volume
        action = "observe"

        if net_flow > 1000 and "optimistic" in headline_reaction.lower():
            action = "align_with_buyers"
        elif net_flow < -1000 and "fear" in headline_reaction.lower():
            action = "prepare_for_selloff"
        elif "confused" in headline_reaction.lower():
            action = "reduce_exposure"

        self.behavior_log.append({
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "headline": headline_reaction,
            "recommended_action": action
        })

        return action

    def get_behavior_trend(self):
        return self.behavior_log
