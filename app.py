import streamlit as st
from config import Config


class App:
    audio = None
    config = None

    def __init__(self):
        self.config = Config().get_config()

    def run(self):
        st.title("Streamlit Whisper")
        self.audio = st.audio_input("Record audio", label_visibility="hidden")


if __name__ == "__main__":
    app = App()
    app.run()
