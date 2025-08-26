import streamlit as st

from client.client import Client
from config import Config
from stt import Whisper


class App:

    def __init__(self):
        self.client = Client()
        self.config = Config().get_config()
        self.stt = Whisper()


    def _sidebar_settings(self, config):
        st.sidebar.header("Settings")
        client = st.sidebar.selectbox(
            "Client",
            [
                "groq",
                "lmstudio",
                "ollama",
                "openai",
            ],
            index=[
                "groq",
                "lmstudio",
                "ollama",
                "openai",
            ].index(config["client"] or "ollama"),
        )
        language = st.sidebar.selectbox(
            "Language",
            [
                "english",
                "german",
            ], index=[
                "english",
                "german",
            ].index(config["language"] or "german"),
        )
        model = st.sidebar.text_input(
            "Whisper Modell",
            value=config["model"] or "base",
        ),
        speaker_recognition = st.sidebar.checkbox(
            "Speaker recognition",
            value=False,
        )
        summary = st.sidebar.checkbox(
            "Summary generation",
            value=False,
        )
        temperature = st.sidebar.slider(
            "Temperature",
            max_value=1.0,
            min_value=0.0,
            step=0.05,
            value=float(config["groq"]["temperature"] or 0.7),
        )
        groq_api_key = st.sidebar.text_input(
            "Groq API Key",
            type="password",
            value=config["groq"]["api_key"] or "",
        )
        openai_api_key = st.sidebar.text_input(
            "OpenAI API Key",
            value=config["openai"]["api_key"] or "",
        )
        return {
            "client": client,
            "language": language,
            "model": model,
            "speaker_recognition": speaker_recognition,
            "summary": summary,
            "temperature": temperature,
            "groq_api_key": groq_api_key,
            "openai_api_key": openai_api_key,
        }

    def run(self):
        st.title("Streamlit Whisper")
        st.session_state["settings"] = self._sidebar_settings(self.config)
        audio_tab, upload_tab = st.tabs(["Audio", "Upload"])
        with audio_tab:
            self.audio = st.audio_input("Record audio", label_visibility="hidden")
            if self.audio is not None:
                self.text = self.stt.transcribe(self.audio)
                st.text(self.text)
        with upload_tab:
            self.file = st.file_uploader("Upload file", label_visibility="hidden")
            if self.file is not None:
                self.text = self.stt.transcribe(self.file)
                st.text(self.text)


if __name__ == "__main__":
    app = App()
    app.run()
