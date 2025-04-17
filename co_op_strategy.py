
class CooperativeStrategyPlanner:
    def __init__(self):
        self.agent_insights = []

    def collect_insight(self, agent_id, market_region, suggestion):
        self.agent_insights.append({
            "agent_id": agent_id,
            "region": market_region,
            "suggestion": suggestion
        })

    def synthesize_plan(self):
        strategy_summary = {}
        for insight in self.agent_insights:
            region = insight["region"]
            suggestion = insight["suggestion"]
            strategy_summary.setdefault(region, []).append(suggestion)

        consensus_plan = {}
        for region, suggestions in strategy_summary.items():
            most_common = max(set(suggestions), key=suggestions.count)
            consensus_plan[region] = most_common

        return consensus_plan

    def reset(self):
        self.agent_insights = []
