import whisper
from config import Config


class Whisper:
    def __init__(self):
        self.config = Config().get_config()
        self.model = whisper.load_model(self.config["model"])

    def transcribe(self, audio_file):
        result = self.model.transcribe(audio_file)
        return result["text"]
