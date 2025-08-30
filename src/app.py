import streamlit as st
import torch

from client.client import Client
from config import Config
from pyannote.audio import Pipeline as PyannotePipeline
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

    def _compute_overlap(interval1, interval2):
        latest_start = max(interval1, interval2)
        earliest_end = min(interval1[13], interval2[13])
        overlap = max(0, earliest_end - latest_start)
        return overlap


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
            help="Separate transcript by speakers using pyannote.audio (requires model and API token)"
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
        pyannote_token = None
        if speaker_recognition:
            pyannote_token = st.sidebar.text_input(
                "pyannote API Token",
                type="password",
                value=config["huggingface"]["pyannote"]["token"] or "",
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
        st.header("Summary")
        st.write_stream(self._chat_message("Summarize following text: " + self.transcription))

    def _tasks(self):
        st.header("Tasks")
        st.write_stream(self._chat_message("Extract tasks from the text: " + self.transcription))

    def _transcribe(self):
        st.header("Transcription")
        settings = st.session_state.get("settings", {})
        speaker_recognition = settings.get("speaker_recognition", False)
        pyannote_token = settings.get("pyannote_token")
        with st.status("Transcribing ...") as status:
            st.text(f"Transcription with: {self.config['model']}")
            self.transcription, self.segments = self.stt.transcribe(self.input)
            if speaker_recognition:
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
                        diarization = self.pyannote_pipeline(self.input)
                        for segment in self.segments:
                            matching_speaker = None
                            max_overlap = 0
                            for turn, _, speaker in diarization.itertracks(yield_label=True):
                                overlap = self._compute_overlap([segment['start'], segment['end']], [turn.start, turn.end])
                                if overlap > max_overlap:
                                    max_overlap = overlap
                                    matching_speaker = speaker
                            st.write(f"{matching_speaker}: {segment['text']}")
                            st.write(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker={speaker}")
                    except Exception as e:
                        st.error(f"Speaker diarization failed: {e}")
            st.text_area("Output", self.transcription)
            status.update(label="Transcription complete", state="complete")

    def run(self):
        st.title("Streamlit Whisper")
        st.session_state["settings"] = self._sidebar_settings(self.config)
        audio_tab, upload_tab = st.tabs(["Audio", "Upload"])
        with audio_tab:
            self.input = st.audio_input("Record audio", label_visibility="hidden")
            if self.input is not None:
                self._transcribe()
                self._summarize()
                self._tasks()
        with upload_tab:
            self.input = st.file_uploader("Upload file", label_visibility="hidden")
            if self.input is not None:
                self._transcribe()
                self._summarize()
                self._tasks()


if __name__ == "__main__":
    app = App()
    app.run()
