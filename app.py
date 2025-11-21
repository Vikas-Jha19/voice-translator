"""
Multilingual Voice Translation System
Optimized for Render Free Tier - Complete Voice Pipeline
"""

import gradio as gr
import os
import whisper
import tempfile
from gtts import gTTS
import time
import logging
from googletrans import Translator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load tiny Whisper model (only 39MB - fits in free tier!)
logger.info("Loading Whisper Tiny model...")
asr_model = whisper.load_model("tiny")
logger.info("Model loaded successfully!")

# Initialize translator
translator = Translator()

LANGUAGES = {
    "hi": "Hindi", "en": "English", "ta": "Tamil", "te": "Telugu",
    "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "bn": "Bengali",
    "gu": "Gujarati", "pa": "Punjabi"
}

LANG_CODES = {
    "Hindi": "hi", "English": "en", "Tamil": "ta", "Telugu": "te",
    "Kannada": "kn", "Malayalam": "ml", "Marathi": "mr", "Bengali": "bn",
    "Gujarati": "gu", "Punjabi": "pa"
}

def translate_voice(audio, src_lang, tgt_lang):
    """Complete voice translation pipeline"""
    
    if audio is None:
        return None, "Please provide audio input", "", ""
    
    src_code = LANG_CODES.get(src_lang, "hi")
    tgt_code = LANG_CODES.get(tgt_lang, "en")
    
    # Handle microphone input (tuple format)
    if isinstance(audio, tuple):
        import soundfile as sf
        sr, data = audio
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp.name, data, sr)
        audio = tmp.name
    
    try:
        start_time = time.time()
        
        # Stage 1: Speech Recognition (Whisper Tiny)
        logger.info(f"ASR: Transcribing {src_lang}...")
        asr_result = asr_model.transcribe(
            audio,
            language=src_code,
            task="transcribe"
        )
        source_text = asr_result["text"].strip()
        
        if not source_text:
            return None, "No speech detected in audio", "", ""
        
        asr_time = time.time() - start_time
        logger.info(f"Transcription: {source_text}")
        
        # Stage 2: Translation (Google Translate)
        logger.info(f"Translating: {src_lang} → {tgt_lang}...")
        trans_start = time.time()
        
        translation_result = translator.translate(
            source_text,
            src=src_code,
            dest=tgt_code
        )
        target_text = translation_result.text
        
        trans_time = time.time() - trans_start
        logger.info(f"Translation: {target_text}")
        
        # Stage 3: Text-to-Speech (Google TTS)
        logger.info(f"TTS: Generating {tgt_lang} audio...")
        tts_start = time.time()
        
        tts = gTTS(text=target_text, lang=tgt_code, slow=False)
        output_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(output_audio.name)
        
        tts_time = time.time() - tts_start
        total_time = time.time() - start_time
        
        # Format outputs
        transcription = f"**Source ({src_lang}):**\n\n{source_text}"
        translation = f"**Target ({tgt_lang}):**\n\n{target_text}"
        timing = f"""**Processing Time:**
- Speech Recognition: {asr_time:.2f}s
- Translation: {trans_time:.2f}s
- Speech Synthesis: {tts_time:.2f}s
- **Total: {total_time:.2f}s**

**Models Used:**
- ASR: Whisper Tiny (39MB)
- Translation: Google Translate
- TTS: Google Text-to-Speech"""
        
        return output_audio.name, transcription, translation, timing
        
    except Exception as e:
        import traceback
        error = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        logger.error(error)
        return None, error, "", ""

# Build Gradio Interface
language_list = list(LANGUAGES.values())

demo = gr.Interface(
    fn=translate_voice,
    inputs=[
        gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="🎤 Record or Upload Audio"
        ),
        gr.Dropdown(
            choices=language_list,
            label="Source Language",
            value="Hindi"
        ),
        gr.Dropdown(
            choices=language_list,
            label="Target Language",
            value="English"
        )
    ],
    outputs=[
        gr.Audio(
            label="🔊 Translated Audio",
            type="filepath"
        ),
        gr.Textbox(
            label="📝 Original Transcription",
            lines=3
        ),
        gr.Textbox(
            label="🌍 Translation",
            lines=3
        ),
        gr.Textbox(
            label="⚡ Performance Metrics",
            lines=8
        )
    ],
    title="🌍 Multilingual Voice Translation System",
    description="""
    **Real-time voice translation for Indian languages**
    
    Speak or upload audio → Get instant translation with synthesized speech
    
    Supports: Hindi, English, Tamil, Telugu, Kannada, Malayalam, Marathi, Bengali, Gujarati, Punjabi
    """,
    examples=[
        [None, "Hindi", "English"],
        [None, "Tamil", "English"],
        [None, "English", "Hindi"]
    ]
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🚀 Starting server on port {port}")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )
