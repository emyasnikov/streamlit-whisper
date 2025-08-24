import ollama

from ..config import Config
from .client import Client


class ClientOllama(Client):
    def __init__(self):
        self.config = Config().get_config()
        self.model = self.config["ollama"]["model"]

    def chat(self, messages, stream=False):
        return ollama.chat(
            messages=messages,
            model=self.model,
            stream=stream,
        )
