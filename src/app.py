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
        self.config = Config().get_config()
        self.pyannote_pipeline = None
        self.speaker_map = None
        self.transcription = ""
        self.huggingface_token = ""

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

    def _init_pyannote(self):
        self.pyannote_pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.settings.get("huggingface_token"),
        )
        self.pyannote_pipeline.to(torch.device("cpu"))

    def _run_with_status(self):
        with st.status("") as status:
            status.update(label="Initialization ...", expanded=True, state="running")
            st.markdown("### Transcript")
            self._transcribe(status=status)
            if self.settings.get("summary_generation", False):
                status.update(label="Summary ...", expanded=True, state="running")
                st.markdown("### Summary")
                st.write_stream(self._chat_message(self.settings["prompt_summary"] + ": " + self.transcription))
            if self.settings.get("summary_generation", False):
                status.update(label="Tasks ...", expanded=True, state="running")
                st.markdown("### Tasks")
                st.write_stream(self._chat_message(self.settings["prompt_tasks"] + ": " + self.transcription))
            status.update(label="Complete", expanded=True, state="complete")

    def _sidebar_settings(self):
        st.sidebar.markdown("## Settings")
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
        model = st.sidebar.selectbox(
            "Model",
            [
                "turbo",
                "large",
                "medium",
                "small",
                "base",
                "tiny",
            ], index=[
                "turbo",
                "large",
                "medium",
                "small",
                "base",
                "tiny",
            ].index(self.config["model"] or "turbo"),
        )
        speaker_recognition = st.sidebar.checkbox(
            "Speaker recognition",
            value=False,
            help="Separate transcript by speakers using pyannote.audio (requires model and API token)"
        )
        summary_generation = st.sidebar.checkbox(
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
        huggingface_token = st.sidebar.text_input(
            "pyannote API Token",
            type="password",
            value=self.config["huggingface"]["token"] or "",
            help="HuggingFace API token for pyannote.audio pipeline etc.",
        )
        prompt_summary = st.sidebar.text_area(
            "Prompt - Summary",
            value=self.config["summary"]["prompt"] or "",
        )
        prompt_tasks = st.sidebar.text_area(
            "Prompt - Tasks",
            value=self.config["tasks"]["prompt"] or "",
        )
        return {
            "client": client,
            "groq_api_key": groq_api_key,
            "huggingface_token": huggingface_token,
            "language": language,
            "model": model,
            "openai_api_key": openai_api_key,
            "prompt_summary": prompt_summary,
            "prompt_tasks": prompt_tasks,
            "speaker_recognition": speaker_recognition,
            "summary_generation": summary_generation,
            "temperature": temperature,
        }

    def _transcribe(self, status=None):
        if self.settings.get("speaker_recognition", False):
            if not self.settings.get("huggingface_token"):
                st.warning("HuggingFace API token is missing.")
            else:
                try:
                    self._init_pyannote()
                    status.update(label="Speaker diarisation ...", expanded=True, state="running")
                    status.update(label="Prepare files ...", expanded=True, state="running")
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                        audio = AudioSegment.from_file(self.input)
                        audio = audio.set_frame_rate(16000).set_channels(1)
                        audio.export(tmp_wav.name, format="wav")
                        diarization = self.pyannote_pipeline(tmp_wav.name)
                        status.update(label="Transcription ...", expanded=True, state="running")
                        for turn, _, speaker in diarization.itertracks(yield_label=True):
                            segment_audio = audio[turn.start * 1000:turn.end * 1000]
                            segment_path = f"segment_{turn.start}.wav"
                            segment_audio.export(segment_path, format="wav")
                            with st.chat_message("user"):
                                st.markdown(f"**{speaker}** - {turn.start:.1f}s")
                                segment = self.stt.transcribe(segment_path, self.settings["language"])
                                self.transcription += segment
                                st.write(segment)
                            if os.path.exists(segment_path):
                                os.remove(segment_path)
                except Exception as e:
                    st.error(f"{e}")
        else:
            self.transcription = self.stt.transcribe(self.input, self.settings["language"])
            st.text_area(label="Transcript", value=self.transcription, label_visibility="collapsed")

    def run(self):
        st.title("Streamlit Whisper")
        st.session_state["settings"] = self._sidebar_settings()
        self.settings = st.session_state.get("settings", {})
        self.client = Client(self.settings["client"])
        self.stt = Whisper(self.settings["model"])
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
