"""
Professional Voice Translation App
"""

import streamlit as st
import whisper
import tempfile
from gtts import gTTS
import time
from deep_translator import GoogleTranslator
import os

# Professional page config
st.set_page_config(
    page_title="Voice Translator",
    page_icon="🌐",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-container {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        margin: 2rem auto;
    }
    
    h1 {
        color: #2d3748;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #718096;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
    }
    
    .success-box {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model silently
@st.cache_resource
def load_model():
    return whisper.load_model("base")

if 'model' not in st.session_state:
    with st.spinner(""):
        st.session_state.model = load_model()

model = st.session_state.model

# Languages
LANGUAGES = ["Hindi", "English", "Tamil", "Telugu", "Kannada", "Malayalam", 
             "Marathi", "Bengali", "Gujarati", "Punjabi"]

CODES = {
    "Hindi": "hi", "English": "en", "Tamil": "ta", "Telugu": "te",
    "Kannada": "kn", "Malayalam": "ml", "Marathi": "mr", "Bengali": "bn",
    "Gujarati": "gu", "Punjabi": "pa"
}

# Main UI
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header
st.title("🌐 Voice Translator")
st.markdown('<p class="subtitle">Instant voice translation for Indian languages</p>', unsafe_allow_html=True)

# Language selection
col1, col2 = st.columns(2)
with col1:
    src_lang = st.selectbox("From", LANGUAGES, label_visibility="visible")
with col2:
    tgt_lang = st.selectbox("To", LANGUAGES, index=1, label_visibility="visible")

st.markdown("---")

# Audio input section
st.markdown("### 🎤 Record or Upload Audio")

tab1, tab2 = st.tabs(["📁 Upload File", "🎙️ Record"])

with tab1:
    audio_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a", "ogg"],
        label_visibility="collapsed"
    )

with tab2:
    audio_bytes = st.audio_input("Click to record", label_visibility="collapsed")

st.markdown("---")

# Translate button
translate_btn = st.button("✨ Translate Now", type="primary")

# Processing
if translate_btn:
    audio_input = audio_bytes if audio_bytes else audio_file
    
    if audio_input is None:
        st.warning("⚠️ Please record or upload audio first")
    else:
        with st.spinner("🔄 Translating..."):
            try:
                src_code = CODES[src_lang]
                tgt_code = CODES[tgt_lang]
                
                # Save audio
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    if isinstance(audio_input, bytes):
                        tmp.write(audio_input)
                    else:
                        tmp.write(audio_input.read())
                    audio_path = tmp.name
                
                # Process
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
                    st.error("❌ Could not detect speech. Please try again with clearer audio.")
                else:
                    # Translate
                    translator = GoogleTranslator(source=src_code, target=tgt_code)
                    target_text = translator.translate(source_text)
                    
                    # TTS
                    tts = gTTS(text=target_text, lang=tgt_code, slow=False)
                    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    tts.save(output_path.name)
                    
                    # Display results
                    st.markdown("### ✅ Translation Complete")
                    
                    # Audio player
                    with open(output_path.name, "rb") as audio_out:
                        st.audio(audio_out.read(), format="audio/mp3")
                    
                    st.markdown("---")
                    
                    # Original text
                    st.markdown(f"""
                    <div class="info-box">
                        <strong>🗣️ Original ({src_lang})</strong><br>
                        {source_text}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Translation
                    st.markdown(f"""
                    <div class="success-box">
                        <strong>🌍 Translation ({tgt_lang})</strong><br>
                        {target_text}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Cleanup
                    os.unlink(audio_path)
                    os.unlink(output_path.name)
                    
            except Exception as e:
                st.error("❌ Translation failed. Please try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #718096; font-size: 0.9rem;">
    <p>🌐 Supports 10 Indian languages<br>
    ⚡ Fast & accurate translations</p>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
