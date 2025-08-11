import streamlit as st
from config import Config
from stt import Whisper


class App:
    audio = None
    config = None

    def __init__(self):
        self.config = Config().get_config()
        self.stt = Whisper()

    def run(self):
        st.title("Streamlit Whisper")
        self.audio = st.audio_input("Record audio", label_visibility="hidden")
        if self.audio is not None:
            self.text = self.stt.transcribe(self.audio)
            st.text(self.text)
        self.file = st.file_uploader("Upload file", label_visibility="hidden")
        if self.file is not None:
            self.text = self.stt.transcribe(self.file)
            st.text(self.text)


if __name__ == "__main__":
    app = App()
    app.run()
