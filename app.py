import streamlit as st


class App:
    audio = None

    def run(self):
        st.title("Streamlit Whisper")
        self.audio = st.audio_input("Record audio", label_visibility="hidden")


if __name__ == "__main__":
    app = App()
    app.run()
