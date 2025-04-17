
import json

class ExperienceTransfer:
    def __init__(self):
        self.package = {}

    def package_experience(self, name, data):
        self.package[name] = data

    def export(self):
        return json.dumps(self.package)

    def import_package(self, data):
        try:
            self.package = json.loads(data)
            return True
        except:
            return False

    def get_package(self):
        return self.package
