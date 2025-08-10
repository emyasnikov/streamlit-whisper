import streamlit as st


class App:
    def run(self):
        st.title("Streamlit Whisper")
        audio_value = st.audio_input("Record a voice message")
        if audio_value:
            st.audio(audio_value)


if __name__ == "__main__":
    app = App()
    app.run()
