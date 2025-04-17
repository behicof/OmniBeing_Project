
class DistributedDecisionGrid:
    def __init__(self):
        self.agent_votes = {}

    def cast_vote(self, agent_id, decision):
        self.agent_votes[agent_id] = decision

    def resolve_consensus(self):
        if not self.agent_votes:
            return "No votes"
        tally = {}
        for vote in self.agent_votes.values():
            tally[vote] = tally.get(vote, 0) + 1
        decision = max(tally.items(), key=lambda x: x[1])[0]
        return {"consensus": decision, "votes": tally}

    def clear_votes(self):
        self.agent_votes = {}
