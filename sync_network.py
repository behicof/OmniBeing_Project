
class SynchronizedDecisionNetwork:
    def __init__(self):
        self.agent_votes = {}

    def propose_decision(self, agent_id, decision):
        self.agent_votes[agent_id] = decision

    def compute_consensus(self):
        vote_count = {}
        for decision in self.agent_votes.values():
            vote_count[decision] = vote_count.get(decision, 0) + 1
        consensus = max(vote_count.items(), key=lambda x: x[1])[0] if vote_count else "no_consensus"
        return consensus

    def reset_votes(self):
        self.agent_votes = {}

    def get_votes(self):
        return self.agent_votes
