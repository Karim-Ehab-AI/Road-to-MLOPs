import os

class BaseClass:
    def __init__(self):
        self._main_folder = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.artifact_folder = os.path.join(self._main_folder, "artifacts")

    def make_artifact_folder(self):
        artifact_folder = self.artifact_folder
        os.makedirs(artifact_folder, exist_ok=True)
