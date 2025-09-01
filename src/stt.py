import tempfile
import whisper

from config import Config


class Whisper:
    def __init__(self):
        self.config = Config().get_config()
        self.model = whisper.load_model(self.config["model"])

    def temp_file(self, file):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(file.read())
            return tmp.name

    def transcribe(self, input):
        if hasattr(input, "read"):
            file_path = self.temp_file(input)
        elif isinstance(input, str):
            file_path = input
        with open(file_path, "rb") as f:
            result = self.model.transcribe(file_path, language=self.config["language"])
        return result.get("text", "")
