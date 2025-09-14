from config import Config


class Client:
    temperature = None

    def __init__(self, client=None, temperature=None):
        config = Config().get_config()
        client = client or config["client"]
        if client == "groq":
            from client.groq import ClientGroq
            self.__class__ = ClientGroq
        elif client == "lmstudio":
            from client.lmstudio import ClientLMStudio
            self.__class__ = ClientLMStudio
        elif client == "ollama":
            from client.ollama import ClientOllama
            self.__class__ = ClientOllama
        elif client == "openai":
            from client.openai import ClientOpenAI
            self.__class__ = ClientOpenAI
        else:
            raise ValueError(f"Invalid client: {client}")
        self.temperature = temperature
        self.__init__()

    def chat(self, messages, stream=False):
        raise NotImplementedError()
