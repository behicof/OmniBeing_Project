
class ExperienceTransferUnit:
    def __init__(self):
        self.memory = []

    def log_experience(self, pattern, result):
        self.memory.append({"pattern": pattern, "result": result})

    def recommend_action(self, current_pattern):
        for mem in self.memory[::-1]:
            if mem["pattern"] == current_pattern:
                return f"Previously, this pattern resulted in '{mem['result']}' – consider similar action."
        return "No matching experience found. Proceed with analysis."

    def get_memory(self):
        return self.memory
