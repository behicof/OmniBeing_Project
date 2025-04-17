
class CollectiveMemorySync:
    def __init__(self):
        self.shared_pool = {}

    def broadcast_experience(self, agent_id, experience):
        self.shared_pool[agent_id] = experience

    def receive_all(self):
        return self.shared_pool

    def merge_memories(self):
        merged = {}
        for exp in self.shared_pool.values():
            for key, value in exp.items():
                merged.setdefault(key, []).append(value)
        return merged
