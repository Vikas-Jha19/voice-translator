"""
Multilingual Voice Translation System
Hybrid: Local Whisper Tiny + Cloud Translation (Fits in 1GB!)
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
    page_title="🌍 Voice Translation",
    page_icon="🌍",
    layout="wide"
)

# Cache model loading
@st.cache_resource
def load_whisper():
    """Load Whisper Tiny (only 39MB!)"""
    return whisper.load_model("tiny")

# Initialize
if 'model' not in st.session_state:
    with st.spinner("🔄 Loading AI model..."):
        st.session_state.model = load_whisper()
        st.success("✅ Model loaded!")

model = st.session_state.model

# Language mapping
LANG_NAMES = ["Hindi", "English", "Tamil", "Telugu", "Kannada", "Malayalam", 
              "Marathi", "Bengali", "Gujarati", "Punjabi"]

WHISPER_CODES = {
    "Hindi": "hi", "English": "en", "Tamil": "ta", "Telugu": "te",
    "Kannada": "kn", "Malayalam": "ml", "Marathi": "mr", "Bengali": "bn",
    "Gujarati": "gu", "Punjabi": "pa"
}

# Header
st.title("🌍 Multilingual Voice Translation System")
st.markdown("**High-quality voice translation for Indian languages**")
st.markdown("---")

# Two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Input")
    
    src_lang = st.selectbox("Source Language", LANG_NAMES, index=0)
    tgt_lang = st.selectbox("Target Language", LANG_NAMES, index=1)
    
    audio_file = st.file_uploader(
        "🎤 Upload audio file",
        type=["wav", "mp3", "m4a", "ogg"],
        help="Upload an audio file to translate"
    )
    
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
        st.error("❌ Please provide audio input")
    else:
        with st.spinner("⏳ Processing..."):
            try:
                src_code = WHISPER_CODES[src_lang]
                tgt_code = WHISPER_CODES[tgt_lang]
                
                # Save audio
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    if isinstance(audio_input, bytes):
                        tmp.write(audio_input)
                    else:
                        tmp.write(audio_input.read())
                    audio_path = tmp.name
                
                # Stage 1: ASR (Local Whisper Tiny)
                start_time = time.time()
                progress = st.progress(0, text=f"🎤 Transcribing {src_lang}...")
                
                result = model.transcribe(
                    audio_path,
                    language=src_code,
                    task="transcribe"
                )
                source_text = result["text"].strip()
                asr_time = time.time() - start_time
                progress.progress(33, text="✅ Transcription complete!")
                
                if not source_text:
                    st.error("❌ No speech detected")
                else:
                    # Stage 2: Translation (Google Translate API)
                    progress.progress(33, text=f"🌍 Translating to {tgt_lang}...")
                    trans_start = time.time()
                    
                    translator = GoogleTranslator(source=src_code, target=tgt_code)
                    target_text = translator.translate(source_text)
                    
                    trans_time = time.time() - trans_start
                    progress.progress(66, text="✅ Translation complete!")
                    
                    # Stage 3: TTS (Google)
                    progress.progress(66, text=f"🔊 Generating speech...")
                    tts_start = time.time()
                    
                    tts = gTTS(text=target_text, lang=tgt_code, slow=False)
                    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    tts.save(output_path.name)
                    tts_time = time.time() - tts_start
                    progress.progress(100, text="✅ All done!")
                    
                    total_time = time.time() - start_time
                    
                    # Display results
                    with output_container:
                        st.success("✅ Translation complete!")
                        
                        with open(output_path.name, "rb") as audio_out:
                            st.audio(audio_out.read(), format="audio/mp3")
                        
                        st.markdown(f"**📝 Original ({src_lang}):**")
                        st.info(source_text)
                        
                        st.markdown(f"**🌍 Translation ({tgt_lang}):**")
                        st.success(target_text)
                        
                        # Metrics
                        st.markdown("**⚡ Performance:**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("ASR", f"{asr_time:.2f}s")
                        with col2:
                            st.metric("Translation", f"{trans_time:.2f}s")
                        with col3:
                            st.metric("TTS", f"{tts_time:.2f}s")
                        with col4:
                            st.metric("Total", f"{total_time:.2f}s")
                        
                        st.markdown("""
                        **🤖 Tech Stack:**
                        - ASR: Whisper Tiny (Local, 39MB)
                        - Translation: Google Translate API
                        - TTS: Google Text-to-Speech
                        """)
                    
                    os.unlink(audio_path)
                    os.unlink(output_path.name)
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
**🛠️ Optimized Architecture:**
- **Lightweight**: Only 39MB model loaded locally
- **Fast**: Processing in 5-10 seconds
- **Reliable**: Uses proven Google Translate API
- **Memory Efficient**: Fits comfortably in 1GB

**🌐 Supported Languages:**
Hindi, English, Tamil, Telugu, Kannada, Malayalam, Marathi, Bengali, Gujarati, Punjabi

**✨ Features:**
- File upload + browser recording
- Real-time progress tracking
- Performance metrics
- Clean, professional UI
""")
