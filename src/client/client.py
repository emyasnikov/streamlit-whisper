from ..config import Config


class Client:
    def __init__(self):
        config = Config().get_config()
        client = config["client"]
        if client == "groq":
            from .groq import ClientGroq
            self.__class__ = ClientGroq
        elif client == "lmstudio":
            from .lmstudio import ClientLMStudio
            self.__class__ = ClientLMStudio
        elif client == "ollama":
            from .ollama import ClientOllama
            self.__class__ = ClientOllama
        elif client == "openai":
            from .openai import ClientOpenAI
            self.__class__ = ClientOpenAI
        else:
            raise ValueError(f"Invalid client: {client}")
        self.__init__()

    def chat(self, messages, stream=False):
        raise NotImplementedError()
