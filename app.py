"""
Professional Voice Translation App (Premium UI)
"""

import streamlit as st
import whisper
import tempfile
from gtts import gTTS
import time
from deep_translator import GoogleTranslator
import os

# Page config
st.set_page_config(
    page_title="Voice Translator",
    page_icon="🌐",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# NEW PROFESSIONAL UI CSS
st.markdown("""
<style>
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Global font & smoothing */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    /* App Background */
    .stApp {
        background: #F3F4F6;
    }

    .main-container {
        background: white;
        padding: 2.2rem 2.5rem;
        border-radius: 14px;
        max-width: 800px;
        margin: 2.5rem auto;
        box-shadow: 0 4px 25px rgba(0,0,0,0.07);
    }

    h1 {
        color: #111827;
        font-weight: 600;
        text-align: center;
        margin-bottom: 0.3rem;
        font-size: 1.9rem;
    }

    .subtitle {
        color: #6B7280;
        text-align: center;
        margin-bottom: 1.5rem;
        font-size: 1.05rem;
    }

    /* Buttons */
    .stButton>button {
        background: #2563EB !important;
        color: white !important;
        font-weight: 600;
        width: 100%;
        padding: 0.7rem;
        border-radius: 8px;
        border: 1px solid #1D4ED8;
        transition: 0.15s;
        font-size: 1rem;
    }

    .stButton>button:hover {
        background: #1D4ED8 !important;
        box-shadow: 0 5px 12px rgba(29,78,216,0.15);
        transform: translateY(-1px);
    }

    /* Info + success boxes */
    .info-box, .success-box {
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        line-height: 1.5;
        font-size: 0.95rem;
    }

    .info-box {
        background: #EFF6FF;
        border-left: 4px solid #2563EB;
    }

    .success-box {
        background: #F0FDF4;
        border-left: 4px solid #16A34A;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return whisper.load_model("base")

if "model" not in st.session_state:
    with st.spinner("Loading model..."):
        st.session_state.model = load_model()

model = st.session_state.model

# Language options
LANGUAGES = ["Hindi", "English", "Tamil", "Telugu", "Kannada", "Malayalam",
             "Marathi", "Bengali", "Gujarati", "Punjabi"]

CODES = {
    "Hindi": "hi", "English": "en", "Tamil": "ta", "Telugu": "te",
    "Kannada": "kn", "Malayalam": "ml", "Marathi": "mr", "Bengali": "bn",
    "Gujarati": "gu", "Punjabi": "pa"
}

# UI Container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.title("🌐 Voice Translator")
st.markdown('<p class="subtitle">Professional multilingual speech translation</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    src_lang = st.selectbox("Translate from", LANGUAGES)
with col2:
    tgt_lang = st.selectbox("Translate to", LANGUAGES, index=1)

st.markdown("---")

st.subheader("🎤 Input Speech")

tab1, tab2 = st.tabs(["Upload File", "Record Audio"])

with tab1:
    audio_file = st.file_uploader(
        "Upload Audio",
        type=["wav", "mp3", "m4a", "ogg"]
    )
with tab2:
    audio_bytes = st.audio_input("Record Audio")

st.markdown("---")

translate_btn = st.button("Translate Now")

# Logic
if translate_btn:
    input_audio = audio_bytes if audio_bytes else audio_file

    if input_audio is None:
        st.warning("Please upload or record audio first.")
    else:
        with st.spinner("Processing..."):
            try:
                src_code = CODES[src_lang]
                tgt_code = CODES[tgt_lang]

                # Save audio temp
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    if isinstance(input_audio, bytes):
                        tmp.write(input_audio)
                    else:
                        tmp.write(input_audio.read())
                    audio_path = tmp.name

                # Transcribe
                res = model.transcribe(
                    audio_path,
                    language=src_code,
                    task="transcribe",
                    fp16=False,
                    best_of=5,
                    beam_size=5,
                    temperature=0.0
                )
                src_text = res["text"].strip()

                if not src_text:
                    st.error("No speech detected. Try again.")
                else:
                    # Translate
                    translator = GoogleTranslator(source=src_code, target=tgt_code)
                    translated = translator.translate(src_text)

                    # TTS
                    tts = gTTS(text=translated, lang=tgt_code)
                    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    tts.save(out_file.name)

                    st.subheader("✔ Audio Output")
                    with open(out_file.name, "rb") as a:
                        st.audio(a.read())

                    # Display text sections
                    st.markdown(
                        f"<div class='info-box'><strong>Original ({src_lang})</strong><br>{src_text}</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<div class='success-box'><strong>Translation ({tgt_lang})</strong><br>{translated}</div>",
                        unsafe_allow_html=True
                    )

                    os.unlink(audio_path)
                    os.unlink(out_file.name)

            except Exception as e:
                st.error("Translation failed. Try again.")

st.markdown("</div>", unsafe_allow_html=True)
