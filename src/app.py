import os
import streamlit as st
import tempfile
import torch

from client.client import Client
from config import Config
from pyannote.audio import Pipeline as PyannotePipeline
from pydub import AudioSegment
from stt import Whisper


class App:
    def __init__(self):
        self.client = Client()
        self.config = Config().get_config()
        self.stt = Whisper()
        self.pyannote_pipeline = None
        self.speaker_map = None
        self.transcription = ""

    def _chat_message(self, input):
        messages = [{
            "role": "user",
            "content": input,
        }]

        message = ""
        response = self.client.chat(messages=messages, stream=True)

        for part in response:
            content = part["message"]["content"]
            message += content
            yield content

    def _run_with_status(self):
        with st.status("") as status:
            status.update(label="Transcription ...", expanded=True, state="running")
            self._transcribe()
            status.update(label="Summary ...")
            self._summarize()
            status.update(label="Tasks ...")
            self._tasks()
            status.update(label="Complete", state="complete")

    def _sidebar_settings(self):
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
            ].index(self.config["client"] or "ollama"),
        )
        language = st.sidebar.selectbox(
            "Language",
            [
                "english",
                "german",
            ], index=[
                "english",
                "german",
            ].index(self.config["language"] or "german"),
        )
        model = st.sidebar.text_input(
            "Whisper Modell",
            value=self.config["model"] or "base",
        ),
        speaker_recognition = st.sidebar.checkbox(
            "Speaker recognition",
            value=False,
            help="Separate transcript by speakers using pyannote.audio (requires model and API token)"
        )
        summary = st.sidebar.checkbox(
            "Summary generation",
            value=self.config["summarize"] or False,
        )
        temperature = st.sidebar.slider(
            "Temperature",
            max_value=1.0,
            min_value=0.0,
            step=0.05,
            value=float(self.config["groq"]["temperature"] or 0.7),
        )
        groq_api_key = st.sidebar.text_input(
            "Groq API Key",
            type="password",
            value=self.config["groq"]["api_key"] or "",
        )
        openai_api_key = st.sidebar.text_input(
            "OpenAI API Key",
            value=self.config["openai"]["api_key"] or "",
        )
        pyannote_token = None
        if speaker_recognition:
            pyannote_token = st.sidebar.text_input(
                "pyannote API Token",
                type="password",
                value=self.config["huggingface"]["pyannote"]["token"] or "",
                help="HuggingFace API token for pyannote.audio pipeline"
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
            "pyannote_token": pyannote_token,
        }

    def _summarize(self):
        if self.settings.get("summarize", False):
            st.header("Summary")
            st.write_stream(self._chat_message("Summarize following text: " + self.transcription))

    def _tasks(self):
        if self.settings.get("speaker_recognition", False):
            st.header("Tasks")
            st.write_stream(self._chat_message("Extract tasks from the text: " + self.transcription))

    def _transcribe(self):
        pyannote_token = self.settings.get("pyannote_token")
        st.text(f"Transcription with: {self.config['model']}")
        if self.settings.get("speaker_recognition", False):
            if not pyannote_token:
                st.warning("HuggingFace API token is missing.")
            else:
                st.text(f"Speaker diarisation ...")
                try:
                    self.pyannote_pipeline = PyannotePipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=pyannote_token,
                    )
                    self.pyannote_pipeline.to(torch.device("cpu"))
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                        audio = AudioSegment.from_file(self.input)
                        audio = audio.set_frame_rate(16000).set_channels(1)
                        audio.export(tmp_wav.name, format="wav")
                        diarization = self.pyannote_pipeline(tmp_wav.name)
                        for turn, _, speaker in diarization.itertracks(yield_label=True):
                            segment_audio = audio[turn.start * 1000:turn.end * 1000]
                            segment_id = f"{turn.start:.1f}_{turn.end:.1f}_{speaker}"
                            segment_path = f"segment_{segment_id}.wav"
                            segment_audio.export(segment_path, format="wav")
                            st.write(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker={speaker}")
                            st.write(self.stt.transcribe(segment_path))
                            if os.path.exists(segment_path):
                                os.remove(segment_path)
                except Exception as e:
                    st.error(f"{e}")

    def run(self):
        st.title("Streamlit Whisper")
        st.session_state["settings"] = self._sidebar_settings()
        self.settings = st.session_state.get("settings", {})
        audio_tab, upload_tab = st.tabs(["Audio", "Upload"])
        with audio_tab:
            self.input = st.audio_input("Record audio", label_visibility="hidden")
            if self.input is not None:
                self._run_with_status()
        with upload_tab:
            self.input = st.file_uploader("Upload file", label_visibility="hidden")
            if self.input is not None:
                self._run_with_status()


if __name__ == "__main__":
    app = App()
    app.run()
