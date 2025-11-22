"""
Multilingual Voice Translation System
Uses HF Inference API - Models run remotely (fits in 1GB!)
"""

import streamlit as st
import tempfile
from gtts import gTTS
import time
from huggingface_hub import InferenceClient
import os

# Page config
st.set_page_config(
    page_title="🌍 Voice Translation",
    page_icon="🌍",
    layout="wide"
)

# Initialize HF client (no token needed for public inference)
@st.cache_resource
def get_client():
    return InferenceClient()

client = get_client()

# Language mapping
LANGUAGES = {
    "Hindi": ("hin_Deva", "hi"), "English": ("eng_Latn", "en"), 
    "Tamil": ("tam_Taml", "ta"), "Telugu": ("tel_Telu", "te"),
    "Kannada": ("kan_Knda", "kn"), "Malayalam": ("mal_Mlym", "ml"), 
    "Marathi": ("mar_Deva", "mr"), "Bengali": ("ben_Beng", "bn"), 
    "Gujarati": ("guj_Gujr", "gu"), "Punjabi": ("pan_Guru", "pa")
}

# Header
st.title("🌍 Multilingual Voice Translation System")
st.markdown("**Powered by Hugging Face Inference API**")
st.markdown("State-of-the-art models: Whisper Large + NLLB-600M")
st.markdown("---")

# Two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Input")
    
    src_lang = st.selectbox("Source Language", list(LANGUAGES.keys()), index=0)
    tgt_lang = st.selectbox("Target Language", list(LANGUAGES.keys()), index=1)
    
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
        with st.spinner("⏳ Processing with cloud AI..."):
            try:
                nllb_code, tts_code = LANGUAGES[src_lang][0], LANGUAGES[src_lang][1]
                tgt_nllb, tgt_tts = LANGUAGES[tgt_lang][0], LANGUAGES[tgt_lang][1]
                
                # Save audio
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    if isinstance(audio_input, bytes):
                        tmp.write(audio_input)
                    else:
                        tmp.write(audio_input.read())
                    audio_path = tmp.name
                
                # Stage 1: ASR via HF API
                start_time = time.time()
                progress = st.progress(0, text=f"🎤 Transcribing {src_lang}...")
                
                with open(audio_path, "rb") as f:
                    audio_data = f.read()
                
                try:
                    asr_result = client.automatic_speech_recognition(
                        audio_data,
                        model="openai/whisper-large-v3"
                    )
                    source_text = asr_result.get("text", "").strip()
                except Exception as e:
                    st.error(f"ASR Error: {str(e)}")
                    st.info("Trying alternative model...")
                    # Fallback to smaller model
                    asr_result = client.automatic_speech_recognition(
                        audio_data,
                        model="openai/whisper-base"
                    )
                    source_text = asr_result.get("text", "").strip()
                
                asr_time = time.time() - start_time
                progress.progress(33, text="✅ Transcription complete!")
                
                if not source_text:
                    st.error("❌ No speech detected")
                else:
                    # Stage 2: Translation via HF API
                    progress.progress(33, text=f"🌍 Translating to {tgt_lang}...")
                    trans_start = time.time()
                    
                    try:
                        trans_result = client.translation(
                            source_text,
                            model="facebook/nllb-200-distilled-600M",
                            src_lang=nllb_code,
                            tgt_lang=tgt_nllb
                        )
                        target_text = trans_result.get("translation_text", "")
                    except Exception as e:
                        # Fallback to simple translation
                        from deep_translator import GoogleTranslator
                        translator = GoogleTranslator(source=tts_code, target=tgt_tts)
                        target_text = translator.translate(source_text)
                    
                    trans_time = time.time() - trans_start
                    progress.progress(66, text="✅ Translation complete!")
                    
                    # Stage 3: TTS
                    progress.progress(66, text=f"🔊 Generating speech...")
                    tts_start = time.time()
                    
                    tts = gTTS(text=target_text, lang=tgt_tts, slow=False)
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
                        **🤖 Cloud AI Models:**
                        - ASR: Whisper Large V3 (via HF API)
                        - Translation: NLLB-600M (via HF API)
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
**🛠️ Cloud-Based Architecture:**
- All AI models run on Hugging Face servers
- Zero local memory footprint
- Best available models (Whisper Large V3, NLLB-600M)
- Fast and scalable

**🌐 Supported Languages:**
Hindi, English, Tamil, Telugu, Kannada, Malayalam, Marathi, Bengali, Gujarati, Punjabi
""")
