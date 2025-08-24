from ..config import Config
from openai import OpenAI


class ClientLMStudio():
    def __init__(self):
        self.config = Config().get_config()
        self.model = self.config["lmstudio"]["model"]
        self.temperature = self.config["lmstudio"]["temperature"]
        self.client = OpenAI(
            api_key=self.config["lmstudio"]["api_key"],
            base_url=self.config["lmstudio"]["url"],
        )

    def chat(self, messages, stream=False):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=stream,
            temperature=self.temperature,
        )
        if stream:
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                yield {
                    "message": {
                        "content": content,
                    },
                }
        else:
            content = response.choices[0].message.content
            yield {
                "message": {
                    "content": content,
                },
            }
