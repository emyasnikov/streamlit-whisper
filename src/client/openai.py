from ..config import Config
from openai import OpenAI


class ClassOpenAI():
    def __init__(self):
        self.config = Config().get_config()
        self.client = OpenAI(
            api_key=self.config["openai"]["api_key"]
        )

    def chat(self, messages, stream=False):
        response = self.client.chat.completions.create(
            model=self.config["openai"]["model"],
            messages=messages,
            stream=stream,
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
