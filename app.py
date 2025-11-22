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

    /* Card Styling */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
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

    /* Spinner Animation */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .spinner {
        border: 3px solid #bfdbfe;
        border-top: 3px solid #2563eb;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        animation: spin 1s linear infinite;
    }

    /* Remove Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Audio Player */
    .stAudio {
        margin-top: 1rem;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f1f5f9;
        border-radius: 8px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# BACKEND FUNCTIONS
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_whisper():
    """Load Whisper model silently"""
    return whisper.load_model("base")

async def generate_speech_async(text, voice, output_path):
    """Generate High-Quality Neural Speech (Async)"""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def generate_speech(text, voice, output_path):
    """Sync wrapper for async Edge TTS"""
    try:
        # Create new event loop for Streamlit compatibility
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(generate_speech_async(text, voice, output_path))
        loop.close()
        return True
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return False

def process_audio(audio_input, src_lang, tgt_lang, model):
    """Main Processing Pipeline"""
    audio_path = None
    output_audio_path = None
    
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
            task="transcribe",
            fp16=False,
            best_of=5,
            beam_size=5,
            temperature=0.0
        )
        source_text = result["text"].strip()

        if not source_text:
            return None, None, None, "No speech detected in the audio."

        # 3. Translate
        tgt_code = LANG_CODES[tgt_lang]
        translator = GoogleTranslator(source=src_code, target=tgt_code)
        translated_text = translator.translate(source_text)

        # 4. Synthesize (TTS) - Neural Voice
        output_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        tgt_voice = VOICE_MAP.get(tgt_lang, "en-US-AriaNeural")
        
        success = generate_speech(translated_text, tgt_voice, output_audio_path)
        
        if not success:
            return source_text, translated_text, None, "TTS generation failed"
        
        # Cleanup Input Audio
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)
        
        return source_text, translated_text, output_audio_path, None

    except Exception as e:
        # Cleanup on error
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)
        if output_audio_path and os.path.exists(output_audio_path):
            os.unlink(output_audio_path)
        return None, None, None, f"Error: {str(e)}"

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
            <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎙️ Linguist Pro</h1>
            <p style="color: #64748b; font-size: 1.1rem;">Professional Real-time Voice Translation</p>
        </div>
    """, unsafe_allow_html=True)

    # Language Controls
    with st.container():
        col1, col2, col3 = st.columns([4, 1, 4])
        with col1:
            src_lang = st.selectbox("Source Language", list(LANG_CODES.keys()), index=1)
        with col2:
            st.markdown("<div style='text-align: center; padding-top: 1.8rem; color: #94a3b8; font-size: 1.5rem;'>→</div>", unsafe_allow_html=True)
        with col3:
            tgt_lang = st.selectbox("Target Language", list(LANG_CODES.keys()), index=0)

    st.markdown("###") # Spacer

    # Input Section
    st.markdown("**📥 Audio Input**")
    tab_record, tab_upload = st.tabs(["🎙️ Record Voice", "📁 Upload File"])
    
    audio_data = None
    
    with tab_record:
        recorded_audio = st.audio_input("Record your voice", label_visibility="collapsed")
        if recorded_audio:
            audio_data = recorded_audio
    
    with tab_upload:
        uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a", "ogg"], label_visibility="collapsed")
        if uploaded_file:
            audio_data = uploaded_file

    # Action Button
    st.markdown("###")
    if st.button("✨ Translate Audio"):
        if not audio_data:
            st.warning("⚠️ Please provide an audio input first.")
            return

        # Processing State
        with st.spinner(""):
            st.markdown("""
                <div style="background: #eff6ff; padding: 1rem; border-radius: 8px; border: 1px solid #bfdbfe; color: #1e40af; display: flex; align-items: center; gap: 10px;">
                    <div class="spinner"></div>
                    <span>Processing high-fidelity audio translation...</span>
                </div>
            """, unsafe_allow_html=True)

            # Execute Pipeline
            src_text, tgt_text, audio_path, error = process_audio(
                audio_data, src_lang, tgt_lang, st.session_state.model
            )

        # Handle Results
        if error:
            st.error(f"❌ {error}")
            return
        
        if not src_text:
            st.error("❌ Could not process the audio. Please try again.")
            return

        # Success Display
        st.success("✅ Translation completed successfully!")
        
        # Results in Two Columns
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
            <div class="result-card" style="background: #f0f9ff; border-color: #bae6fd;">
                <div class="result-label" style="color: #0284c7;">Translated ({tgt_lang})</div>
                <div class="result-text" style="font-weight: 500; color: #0c4a6e;">{tgt_text}</div>
            </div>
            """, unsafe_allow_html=True)

        # Audio Player
        if audio_path and os.path.exists(audio_path):
            st.markdown("###")
            st.markdown("**🔊 Translated Audio:**")
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/mp3")
            
            # Cleanup
            os.unlink(audio_path)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #94a3b8; font-size: 0.9rem;">
            <p>🌐 Supporting 10 Indian languages with neural voice synthesis<br>
            ⚡ Powered by state-of-the-art AI models</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
