
class CollectiveMemoryCore:
    def __init__(self):
        self.agent_experiences = {}

    def share_experience(self, agent_id, strategy_name, result):
        if agent_id not in self.agent_experiences:
            self.agent_experiences[agent_id] = []
        self.agent_experiences[agent_id].append({
            "strategy": strategy_name,
            "result": result
        })

    def get_successful_strategies(self):
        successful = {}
        for agent, experiences in self.agent_experiences.items():
            for e in experiences:
                if e["result"] == "profit":
                    successful.setdefault(e["strategy"], []).append(agent)
        return successful

    def view_memory(self):
        return self.agent_experiences
