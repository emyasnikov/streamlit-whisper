import yaml

from yaml.loader import SafeLoader


class Config:
    file = "config.yaml"

    def __init__(self, file=None):
        if file:
            self.file = file

    def get_config(self):
        with open(self.file) as f:
            return yaml.load(f, Loader=SafeLoader)
