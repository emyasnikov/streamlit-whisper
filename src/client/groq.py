from ..config import Config
from .client import Client
from openai import OpenAI


class ClientGroq(Client):
    def __init__(self):
        self.config = Config().get_config()
        self.client = OpenAI(
            api_key=self.config["groq"]["api_key"],
            base_url=self.config["groq"]["base_url"],
        )

    def chat(self, messages, stream=False):
        response = self.client.chat.completions.create(
            model=self.config["groq"]["model"],
            messages=messages,
            stream=stream,
            temperature=self.config["groq"]["temperature"],
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
