import streamlit as st
import whisper
import tempfile
import asyncio
import edge_tts
from gtts import gTTS
from deep_translator import GoogleTranslator
import os

# -----------------------------------------------------------------------------
# CONFIGURATION & ASSETS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Linguist Pro",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Edge TTS Voices (will fallback to gTTS if unavailable)
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

# -----------------------------------------------------------------------------
# PROFESSIONAL STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Poppins:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1e293b;
    }
    
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        color: #0f172a;
    }

    .stApp {
        background-color: #f8fafc;
        background-image: radial-gradient(#e2e8f0 1px, transparent 1px);
        background-size: 20px 20px;
    }

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
    }

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

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
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
    return whisper.load_model("base")

async def try_edge_tts(text, voice, output_path):
    """Try Edge TTS (may be blocked on some platforms)"""
    try:
        communicate = edge_tts.Communicate(text=text.strip(), voice=voice)
        await communicate.save(output_path)
        return True
    except:
        return False

def generate_speech_gtts(text, lang_code, output_path):
    """Fallback to Google TTS (always works)"""
    try:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        tts.save(output_path)
        return True
    except:
        return False

def generate_speech(text, tgt_lang):
    """Generate speech with automatic fallback"""
    if not text or not text.strip():
        return None, "Empty text"
    
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    
    # Try Edge TTS first (better quality)
    voice = VOICE_MAP.get(tgt_lang)
    lang_code = LANG_CODES.get(tgt_lang, "en")
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(try_edge_tts(text, voice, output_path))
        loop.close()
        
        if success and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path, None  # Edge TTS worked!
    except:
        pass  # Edge TTS failed, try fallback
    
    # Fallback to Google TTS (reliable)
    success = generate_speech_gtts(text, lang_code, output_path)
    
    if success and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path, None
    
    return None, "TTS generation failed"

def process_audio(audio_input, src_lang, tgt_lang, model):
    """Main Processing Pipeline"""
    audio_path = None
    
    try:
        # 1. Save Audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            if isinstance(audio_input, bytes):
                tmp.write(audio_input)
            else:
                tmp.write(audio_input.read())
            audio_path = tmp.name

        # 2. ASR
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

        # 3. Translation
        tgt_code = LANG_CODES[tgt_lang]
        translator = GoogleTranslator(source=src_code, target=tgt_code)
        translated_text = translator.translate(source_text)
        
        if not translated_text or not translated_text.strip():
            return source_text, source_text, None, "Translation failed"

        # 4. TTS (with automatic fallback)
        output_audio_path, tts_error = generate_speech(translated_text, tgt_lang)
        
        if not output_audio_path:
            return source_text, translated_text, None, tts_error
        
        # Cleanup input
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)
        
        return source_text, translated_text, output_audio_path, None

    except Exception as e:
        # Cleanup
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)
        return None, None, None, f"Error: {str(e)}"

# -----------------------------------------------------------------------------
# UI LAYOUT
# -----------------------------------------------------------------------------

def main():
    # Load Model
    if 'model' not in st.session_state:
        with st.spinner("Initializing AI engine..."):
            st.session_state.model = load_whisper()

    # Header
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎙️ Linguist Pro</h1>
            <p style="color: #64748b; font-size: 1.1rem;">Professional Voice Translation</p>
        </div>
    """, unsafe_allow_html=True)

    # Language Controls
    col1, col2, col3 = st.columns([4, 1, 4])
    with col1:
        src_lang = st.selectbox("Source Language", list(LANG_CODES.keys()), index=1)
    with col2:
        st.markdown("<div style='text-align: center; padding-top: 1.8rem; color: #94a3b8; font-size: 1.5rem;'>→</div>", unsafe_allow_html=True)
    with col3:
        tgt_lang = st.selectbox("Target Language", list(LANG_CODES.keys()), index=0)

    st.markdown("###")

    # Input Section
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

    # Action Button
    st.markdown("###")
    if st.button("✨ Translate Audio"):
        if not audio_data:
            st.warning("⚠️ Please provide audio input first")
            return

        # Process
        with st.spinner("Processing translation..."):
            src_text, tgt_text, audio_path, error = process_audio(
                audio_data, src_lang, tgt_lang, st.session_state.model
            )

        # Handle Results
        if error:
            st.error(f"❌ {error}")
            return
        
        if not src_text:
            st.error("❌ Processing failed")
            return

        # Success
        st.success("✅ Translation completed!")
        
        # Results
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

        # Audio Player
        if audio_path and os.path.exists(audio_path):
            st.markdown("###")
            st.markdown("**🔊 Translated Audio:**")
            with open(audio_path, "rb") as f:
                st.audio(f.read(), format="audio/mp3")
            os.unlink(audio_path)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #94a3b8; font-size: 0.9rem;">
            <p>🌐 10 Indian languages • ⚡ High-quality synthesis</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
