"""
Multilingual Voice Translation System
Powered by Whisper Tiny + Google Translate + gTTS
"""

import streamlit as st
import whisper
import tempfile
from gtts import gTTS
import time
from googletrans import Translator
import os

# Page config
st.set_page_config(
    page_title="🌍 Voice Translation",
    page_icon="🌍",
    layout="wide"
)

# Cache model loading
@st.cache_resource
def load_model():
    """Load Whisper model (cached)"""
    return whisper.load_model("tiny")

# Initialize
if 'model' not in st.session_state:
    with st.spinner("🔄 Loading AI models..."):
        st.session_state.model = load_model()
        st.session_state.translator = Translator()

model = st.session_state.model
translator = st.session_state.translator

# Language mapping
LANGUAGES = {
    "Hindi": "hi", "English": "en", "Tamil": "ta", "Telugu": "te",
    "Kannada": "kn", "Malayalam": "ml", "Marathi": "mr", "Bengali": "bn",
    "Gujarati": "gu", "Punjabi": "pa"
}

# Header
st.title("🌍 Multilingual Voice Translation System")
st.markdown("**Real-time voice translation for Indian languages**")
st.markdown("---")

# Two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Input")
    
    # Language selection
    src_lang_name = st.selectbox("Source Language", list(LANGUAGES.keys()), index=0)
    tgt_lang_name = st.selectbox("Target Language", list(LANGUAGES.keys()), index=1)
    
    # Audio input
    audio_file = st.file_uploader(
        "🎤 Upload audio file",
        type=["wav", "mp3", "m4a", "ogg"],
        help="Upload an audio file to translate"
    )
    
    # OR record audio
    st.markdown("**Or record audio:**")
    audio_bytes = st.audio_input("Record your voice")
    
    translate_btn = st.button("🚀 Translate", type="primary", use_container_width=True)

with col2:
    st.subheader("📤 Output")
    output_container = st.container()

# Process translation
if translate_btn:
    audio_input = audio_bytes if audio_bytes else audio_file
    
    if audio_input is None:
        st.error("❌ Please provide audio input (upload or record)")
    else:
        with st.spinner("⏳ Processing..."):
            try:
                # Get language codes
                src_code = LANGUAGES[src_lang_name]
                tgt_code = LANGUAGES[tgt_lang_name]
                
                # Save audio to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    if isinstance(audio_input, bytes):
                        tmp.write(audio_input)
                    else:
                        tmp.write(audio_input.read())
                    audio_path = tmp.name
                
                # Stage 1: Speech Recognition
                start_time = time.time()
                st.info(f"🎤 Transcribing {src_lang_name}...")
                
                result = model.transcribe(
                    audio_path,
                    language=src_code,
                    task="transcribe"
                )
                source_text = result["text"].strip()
                asr_time = time.time() - start_time
                
                if not source_text:
                    st.error("❌ No speech detected in audio")
                else:
                    # Stage 2: Translation
                    st.info(f"🌍 Translating to {tgt_lang_name}...")
                    trans_start = time.time()
                    
                    translation = translator.translate(
                        source_text,
                        src=src_code,
                        dest=tgt_code
                    )
                    target_text = translation.text
                    trans_time = time.time() - trans_start
                    
                    # Stage 3: Text-to-Speech
                    st.info(f"🔊 Generating speech...")
                    tts_start = time.time()
                    
                    tts = gTTS(text=target_text, lang=tgt_code, slow=False)
                    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    tts.save(output_path.name)
                    tts_time = time.time() - tts_start
                    
                    total_time = time.time() - start_time
                    
                    # Display results
                    with output_container:
                        st.success("✅ Translation complete!")
                        
                        # Audio output
                        with open(output_path.name, "rb") as audio_out:
                            st.audio(audio_out.read(), format="audio/mp3")
                        
                        # Text outputs
                        st.markdown(f"**📝 Original ({src_lang_name}):**")
                        st.info(source_text)
                        
                        st.markdown(f"**🌍 Translation ({tgt_lang_name}):**")
                        st.success(target_text)
                        
                        # Performance metrics
                        st.markdown("**⚡ Performance:**")
                        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                        with metrics_col1:
                            st.metric("ASR", f"{asr_time:.2f}s")
                        with metrics_col2:
                            st.metric("Translation", f"{trans_time:.2f}s")
                        with metrics_col3:
                            st.metric("TTS", f"{tts_time:.2f}s")
                        with metrics_col4:
                            st.metric("Total", f"{total_time:.2f}s")
                    
                    # Cleanup
                    os.unlink(audio_path)
                    os.unlink(output_path.name)
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
**🛠️ Tech Stack:**
- ASR: OpenAI Whisper Tiny (39MB)
- Translation: Google Translate
- TTS: Google Text-to-Speech

**🌐 Supported Languages:**
Hindi, English, Tamil, Telugu, Kannada, Malayalam, Marathi, Bengali, Gujarati, Punjabi
""")
