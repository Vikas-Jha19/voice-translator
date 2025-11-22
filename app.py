import streamlit as st
import whisper
import tempfile
import asyncio
import edge_tts
from gtts import gTTS
from deep_translator import GoogleTranslator
import os

# -----------------------------------------------------------------------------
# PROFESSIONAL STYLING & MOBILE-FIRST BRANDING REMOVAL (CSS)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
/* --- Remove Streamlit default branding/footer/deploy/menu/header --- */
#MainMenu, footer, .stDeployButton, header {visibility: hidden; height: 0 !important;}
footer:after, .st-emotion-cache-6qob1r, .st-emotion-cache-1v0mbdj, [data-testid="stFooter"] {display: none !important;}
.stApp {
    background-color: #f8fafc !important;
    background-image: radial-gradient(#e2e8f0 1px, transparent 1px);
    background-size: 20px 20px;
    padding-bottom: 0 !important;
}
/* --- Mobile-first improvements --- */
@media (max-width: 500px) {
    .stApp {padding: 0 !important;}
    .stButton>button, .stSelectbox, .stTabs [data-baseweb="tab-list"], .stFileUploader, .result-card {
        width: 100% !important; min-width: 0 !important; font-size: 16px !important;
    }
    h1, h2, h3 {font-size: 1.4rem !important;}
    .result-card {padding: 0.7rem !important;}
    .footer, .custom-footer {font-size: 11px !important; margin-top: 10px !important; margin-bottom: 4px !important;}
}
.stButton>button, .stSelectbox, .stFileUploader, .result-card {
    width: 100% !important; min-width: 0 !important;
}
.result-card {
    background: #ffffff; border-radius: 12px; padding: 1.1rem; border: 1px solid #e2e8f0;
    margin-bottom: 1rem; box-shadow: 0 1px 6px rgba(37,99,235,0.06);
}
.result-label {
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em;
    color: #64748b; margin-bottom: 0.5rem; font-weight: 600;
}
.result-text {
    font-size: 1.1rem; line-height: 1.6; color: #334155;
}
.stButton>button {
    width: 100%; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: white; border: none; padding: 0.75rem 1.5rem; font-weight: 600;
    border-radius: 8px; transition: all 0.2s ease;
    box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
}
.stButton>button:hover {
    transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(37,99,235,0.1);
}
.stTabs [data-baseweb="tab-list"] {gap: 8px;}
.stTabs [data-baseweb="tab"] {padding: 10px 20px; background-color: #f1f5f9; border-radius: 8px;}
.stTabs [aria-selected="true"] {background-color: #3b82f6; color: white;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# APP CONFIG & LANG MAPS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Linguist Pro",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

VOICE_MAP = {
    "Hindi": "hi-IN-SwaraNeural",
    "English": "en-IN-NeerjaNeural",
    "Tamil": "ta-IN-PallaviNeural",
    "Telugu": "te-IN-ShrutiNeural",
    "Kannada": "kn-IN-SapnaNeural",
    "Malayalam": "ml-IN-SobhanaNeural",
    "Marathi": "mr-IN-AarohiNeural",
    "Bengali": "bn-IN-BashkarNeural",
    "Gujarati": "gu-IN-DhwaniNeural",
    "Punjabi": "pa-IN-GurleenNeural"
}
LANG_CODES = {
    "Hindi": "hi", "English": "en", "Tamil": "ta", "Telugu": "te",
    "Kannada": "kn", "Malayalam": "ml", "Marathi": "mr", "Bengali": "bn",
    "Gujarati": "gu", "Punjabi": "pa"
}

@st.cache_resource(show_spinner=False)
def load_whisper():
    return whisper.load_model("base")

async def try_edge_tts(text, voice, output_path):
    try:
        communicate = edge_tts.Communicate(text=text.strip(), voice=voice)
        await communicate.save(output_path)
        return True
    except:
        return False

def generate_speech_gtts(text, lang_code, output_path):
    try:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        tts.save(output_path)
        return True
    except:
        return False

def generate_speech(text, tgt_lang):
    if not text or not text.strip():
        return None, "Empty translation text"
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    voice = VOICE_MAP.get(tgt_lang)
    lang_code = LANG_CODES.get(tgt_lang, "en")

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(try_edge_tts(text, voice, output_path))
        loop.close()
        if success and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path, None
    except: pass

    success = generate_speech_gtts(text, lang_code, output_path)
    if success and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path, None
    return None, "TTS generation failed"

def process_audio(audio_input, src_lang, tgt_lang, model):
    audio_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            if isinstance(audio_input, bytes):
                tmp.write(audio_input)
            else:
                tmp.write(audio_input.read())
            audio_path = tmp.name

        src_code = LANG_CODES[src_lang]
        result = model.transcribe(
            audio_path,
            language=src_code,
            task="transcribe",
            fp16=False,
            best_of=5,
            beam_size=5,
            temperature=0.0
        )
        source_text = result["text"].strip()

        if not source_text:
            return None, None, None, "No speech detected"

        tgt_code = LANG_CODES[tgt_lang]
        translator = GoogleTranslator(source=src_code, target=tgt_code)
        translated_text = translator.translate(source_text)

        if not translated_text or not translated_text.strip():
            return source_text, source_text, None, "Translation failed"

        output_audio_path, tts_error = generate_speech(translated_text, tgt_lang)
        if not output_audio_path:
            return source_text, translated_text, None, tts_error

        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)
        return source_text, translated_text, output_audio_path, None

    except Exception as e:
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)
        return None, None, None, f"Error: {str(e)}"

# -----------------------------------------------------------------------------
# APP UI & LAYOUT
# -----------------------------------------------------------------------------
if 'model' not in st.session_state:
    with st.spinner("Initializing AI engine..."):
        st.session_state.model = load_whisper()

st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-size: 2.3rem; margin-bottom: 0.2rem;">🎙️ Linguist Pro</h1>
        <p style="color: #64748b; font-size: 1.08rem;">Professional Voice Translation</p>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([4, 1, 4])
with col1:
    src_lang = st.selectbox("From", list(LANG_CODES.keys()), index=1)
with col2:
    st.markdown("<div style='text-align: center; padding-top: 1.8rem; color: #94a3b8; font-size: 1.4rem;'>→</div>", unsafe_allow_html=True)
with col3:
    tgt_lang = st.selectbox("To", list(LANG_CODES.keys()), index=0)

st.markdown("###")
st.markdown("**📥 Audio Input**")
tab_record, tab_upload = st.tabs(["🎙️ Record Voice", "📁 Upload File"])

audio_data = None
with tab_record:
    recorded_audio = st.audio_input("Record", label_visibility="collapsed")
    if recorded_audio:
        audio_data = recorded_audio

with tab_upload:
    uploaded_file = st.file_uploader("Upload", type=["wav", "mp3", "m4a", "ogg"], label_visibility="collapsed")
    if uploaded_file:
        audio_data = uploaded_file

st.markdown("###")
if st.button("✨ Translate Audio"):
    if not audio_data:
        st.warning("⚠️ Please provide audio input first")
    else:
        with st.spinner("Translating..."):
            src_text, tgt_text, audio_path, error = process_audio(
                audio_data, src_lang, tgt_lang, st.session_state.model
            )
        if error:
            st.error(f"❌ {error}")
        elif not src_text:
            st.error("❌ Processing failed")
        else:
            st.success("✅ Translation completed!")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Original ({src_lang})</div>
                    <div class="result-text">{src_text}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="result-card" style="background: #f0f9ff; border-color: #bae6fd;">
                    <div class="result-label" style="color: #0284c7;">Translated ({tgt_lang})</div>
                    <div class="result-text" style="font-weight: 500; color: #0c4a6e;">{tgt_text}</div>
                </div>
                """, unsafe_allow_html=True)
            if audio_path and os.path.exists(audio_path):
                st.markdown("###")
                st.markdown("**🔊 Translated Audio:**")
                with open(audio_path, "rb") as f:
                    st.audio(f.read(), format="audio/mp3")
                os.unlink(audio_path)

# Professional minimal footer
st.markdown("""
<div style="
    text-align:center; 
    color:#94a3b8; 
    font-size:12px;
    margin-top:24px;
    margin-bottom:16px;">
    Linguist Pro © 2025
</div>
""", unsafe_allow_html=True)
