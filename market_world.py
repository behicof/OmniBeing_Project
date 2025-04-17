
class MarketEnvironment3D:
    def __init__(self):
        self.entities = []

    def add_element(self, name, entity_type, position):
        entity = {
            "name": name,
            "type": entity_type,
            "position": position
        }
        self.entities.append(entity)
        return entity

    def render_world_state(self):
        return [f"{e['name']} ({e['type']}) at {e['position']}" for e in self.entities]

    def simulate_interaction(self, avatar_position):
        messages = []
        for entity in self.entities:
            if abs(entity["position"] - avatar_position) <= 2:
                messages.append(f"Avatar near {entity['name']} – trigger interaction")
        return messages
