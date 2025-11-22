import streamlit as st
import whisper
import tempfile
import asyncio
import edge_tts
from deep_translator import GoogleTranslator
import os
import time

# -----------------------------------------------------------------------------
# CONFIGURATION & ASSETS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Linguist Pro",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Professional Voice Mapping (Edge TTS Neural Voices)
VOICE_MAP = {
    "Hindi": "hi-IN-SwaraNeural",
    "English": "en-IN-NeerjaNeural",
    "Tamil": "ta-IN-PallaviNeural",
    "Telugu": "te-IN-ShrutiNeural",
    "Kannada": "kn-IN-GaganNeural",
    "Malayalam": "ml-IN-SobhanaNeural",
    "Marathi": "mr-IN-AarohiNeural",
    "Bengali": "bn-IN-TanishaaNeural",
    "Gujarati": "gu-IN-DhwaniNeural",
    "Punjabi": "pa-IN-OjasNeural"
}

LANG_CODES = {
    "Hindi": "hi", "English": "en", "Tamil": "ta", "Telugu": "te",
    "Kannada": "kn", "Malayalam": "ml", "Marathi": "mr", "Bengali": "bn",
    "Gujarati": "gu", "Punjabi": "pa"
}

# -----------------------------------------------------------------------------
# PROFESSIONAL STYLING (CSS INJECTION)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Poppins:wght@500;700&display=swap');

    /* Global Reset & Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1e293b;
    }
    
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        color: #0f172a;
    }

    /* Background & Main Container */
    .stApp {
        background-color: #f8fafc;
        background-image: radial-gradient(#e2e8f0 1px, transparent 1px);
        background-size: 20px 20px;
    }

    /* Card Styling - The "Glass" Effect */
    .stBaseContainer {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #e2e8f0;
    }

    /* Custom Button Styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.2s ease;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }

    /* Result Cards */
    .result-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
        transition: all 0.2s;
    }
    
    .result-card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }

    .result-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }

    .result-text {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #334155;
    }

    /* Remove Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Audio Player */
    .stAudio {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# BACKEND FUNCTIONS
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_whisper():
    return whisper.load_model("base")

async def generate_speech(text, voice, output_path):
    """Generate High-Quality Neural Speech"""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def process_audio(audio_input, src_lang, tgt_lang, model):
    """Main Processing Pipeline"""
    try:
        # 1. Save Audio Temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            if isinstance(audio_input, bytes):
                tmp.write(audio_input)
            else:
                tmp.write(audio_input.read())
            audio_path = tmp.name

        # 2. Transcribe (ASR)
        src_code = LANG_CODES[src_lang]
        result = model.transcribe(
            audio_path,
            language=src_code, 
            fp16=False
        )
        source_text = result["text"].strip()

        if not source_text:
            return None, None, "No speech detected."

        # 3. Translate
        tgt_code = LANG_CODES[tgt_lang]
        translator = GoogleTranslator(source=src_code, target=tgt_code)
        translated_text = translator.translate(source_text)

        # 4. Synthesize (TTS) - Neural
        output_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        tgt_voice = VOICE_MAP.get(tgt_lang, "en-US-AriaNeural") # Fallback
        
        asyncio.run(generate_speech(translated_text, tgt_voice, output_audio_path))
        
        # Cleanup Input
        os.unlink(audio_path)
        
        return source_text, translated_text, output_audio_path

    except Exception as e:
        return None, None, str(e)

# -----------------------------------------------------------------------------
# UI LAYOUT
# -----------------------------------------------------------------------------

def main():
    # Load Model (Silent)
    if 'model' not in st.session_state:
        with st.spinner("Initializing AI engine..."):
            st.session_state.model = load_whisper()

    # Header Section
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">Linguist Pro</h1>
            <p style="color: #64748b; font-size: 1.1rem;">Professional Real-time Voice Translation</p>
        </div>
    """, unsafe_allow_html=True)

    # Language Controls
    with st.container():
        col1, col2, col3 = st.columns([4, 1, 4])
        with col1:
            src_lang = st.selectbox("Source Language", list(LANG_CODES.keys()), index=1)
        with col2:
            st.markdown("<div style='text-align: center; padding-top: 1.8rem; color: #94a3b8;'>➝</div>", unsafe_allow_html=True)
        with col3:
            tgt_lang = st.selectbox("Target Language", list(LANG_CODES.keys()), index=0)

    st.markdown("###") # Spacer

    # Input Section
    st.markdown("**Input Source**")
    tab_record, tab_upload = st.tabs(["🎙️ Record Voice", "📁 Upload File"])
    
    audio_data = None
    
    with tab_record:
        audio_data = st.audio_input("", label_visibility="collapsed")
    
    with tab_upload:
        uploaded_file = st.file_uploader("", type=["wav", "mp3", "m4a"], label_visibility="collapsed")
        if uploaded_file:
            audio_data = uploaded_file

    # Action Button
    st.markdown("###")
    if st.button("Translate Audio"):
        if not audio_data:
            st.toast("⚠️ Please provide an audio input first.", icon="⚠️")
            return

        # Processing State
        status_container = st.empty()
        with status_container.container():
            st.markdown("""
                <div style="background: #eff6ff; padding: 1rem; border-radius: 8px; border: 1px solid #bfdbfe; color: #1e40af; display: flex; align-items: center; gap: 10px;">
                    <div class="spinner"></div>
                    Processing high-fidelity audio translation...
                </div>
            """, unsafe_allow_html=True)

        # Execute Pipeline
        start_time = time.time()
        src_text, tgt_text, audio_path = process_audio(audio_data, src_lang, tgt_lang, st.session_state.model)
        status_container.empty()

        if audio_path and not src_text: # Error case
            st.error(f"Processing Failed: {audio_path}")
        
        elif src_text:
            # Results Display
            st.toast("Translation completed successfully!", icon="✅")
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Original ({src_lang})</div>
                    <div class="result-text">{src_text}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_res2:
                st.markdown(f"""
                <div class="result-card" style="background: #f8fafc; border-color: #cbd5e1;">
                    <div class="result-label" style="color: #2563eb;">Translated ({tgt_lang})</div>
                    <div class="result-text" style="font-weight: 500;">{tgt_text}</div>
                </div>
                """, unsafe_allow_html=True)

            # Audio Player
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            
            st.markdown("###")
            st.audio(audio_bytes, format="audio/mp3")
            
            # Cleanup
            os.unlink(audio_path)

if __name__ == "__main__":
    main()
